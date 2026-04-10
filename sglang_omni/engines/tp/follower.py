# SPDX-License-Identifier: Apache-2.0
"""Follower worker loop for TP ranks > 0.

Follower processes join the NCCL group, load their model shard, then run
a blocking loop: receive a serialization-safe batch from rank 0 via
broadcast_pyobj, build ForwardBatch, run model.forward(). Output is
discarded — rank 0 handles sampling and results.

This is a top-level module so all functions are picklable for mp.Process
with the 'spawn' start method.
"""
from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any

logger = logging.getLogger(__name__)


def register_omni_models() -> None:
    """Register sglang-omni model classes in sglang's ModelRegistry.

    Must be called in each subprocess before model loading because
    'spawn' start method doesn't inherit parent's registry state.
    Delegates to the single source of truth in sglang_omni.models.registry.
    """
    from sglang_omni.models.sglang_registry import register_omni_models_in_sglang

    register_omni_models_in_sglang()


def relocate_batch_tensors(batch, device) -> None:
    """Move all tensors in *batch* (and nested multimodal inputs) to *device*.

    After ``broadcast_pyobj`` deserializes a batch, tensors remain on rank 0's
    device.  This must be called on every follower to move them to the local GPU.
    """
    import torch

    seen: set[int] = set()

    def _move(obj):
        obj_id = id(obj)
        if obj_id in seen:
            return obj
        seen.add(obj_id)

        if isinstance(obj, torch.Tensor):
            return obj.to(device, non_blocking=True) if obj.device != device else obj
        if isinstance(obj, dict):
            for key, val in obj.items():
                obj[key] = _move(val)
            return obj
        if isinstance(obj, list):
            for i, val in enumerate(obj):
                obj[i] = _move(val)
            return obj
        if isinstance(obj, tuple):
            return tuple(_move(val) for val in obj)
        if hasattr(obj, "__dict__"):
            for attr, val in vars(obj).items():
                moved = _move(val)
                if moved is not val:
                    setattr(obj, attr, moved)
        return obj

    _move(batch)


def sync_page_table(batch, req_to_token_pool) -> None:
    """Write page table rows from rank 0's snapshot into the follower's pool.

    Must be called BEFORE ForwardBatch.init_new() so the attention backend
    can look up correct KV cache locations.
    """
    rows = getattr(batch, "tp_page_table_rows", None)
    if not rows:
        return
    pool_tensor = req_to_token_pool.req_to_token
    for i, row in enumerate(rows):
        idx = int(batch.req_pool_indices[i])
        seq_len = len(row)
        pool_tensor[idx, :seq_len] = row.to(pool_tensor.device)


def patch_batch_for_follower(batch, device, vocab_size: int = 0) -> None:
    """Fill in fields nulled by ``make_follower_batch()`` and relocate tensors.

    ``make_follower_batch()`` sets ``reqs`` and ``sampling_info`` to None to
    strip weakrefs / unpicklable objects.  This function:

    1. Moves all tensor fields to *device* (they arrive on rank 0's device).
    2. Restores minimal stubs so ``ForwardBatch.init_new()`` works.
    """
    import torch
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

    relocate_batch_tensors(batch, device)

    if batch.reqs is None:
        batch.reqs = []
    if batch.sampling_info is None:
        bs = len(batch.seq_lens)
        batch.sampling_info = SamplingBatchInfo(
            temperatures=torch.ones(bs, device=device),
            top_ps=torch.ones(bs, device=device),
            top_ks=torch.zeros(bs, dtype=torch.int32, device=device),
            min_ps=torch.zeros(bs, device=device),
            is_all_greedy=True,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            vocab_size=vocab_size,
        )


def follower_worker_loop(
    tp_rank: int,
    gpu_id: int,
    server_args: Any,
    nccl_port: int,
) -> None:
    """Entry point for a follower TP worker process.

    1. Register omni model classes
    2. Create ModelWorker (joins NCCL group — blocks until rank 0 joins too)
    3. Loop: recv ModelWorkerBatch → ForwardBatch → model.forward()
    4. Exit on None signal

    Args:
        tp_rank: This worker's TP rank (1, 2, ..., tp_size-1).
        gpu_id: CUDA device ID for this worker.
        server_args: SGLang ServerArgs (shared with rank 0).
        nccl_port: NCCL rendezvous port (must match rank 0).
    """
    import torch

    torch.cuda.set_device(gpu_id)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [TP{tp_rank}] %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(f"tp_follower.{tp_rank}")
    log.info("Starting follower on GPU %d", gpu_id)

    # 1. Register model classes before any model loading
    register_omni_models()

    # 2. Create ModelWorker — this calls init_distributed_environment internally
    #    and blocks until all TP ranks have joined the NCCL group
    from sglang_omni.engines.ar.sglang_backend.model_worker import (
        ModelWorker,
        ModelWorkerConfig,
    )

    worker = ModelWorker(
        config=ModelWorkerConfig(nccl_port=nccl_port),
        server_args=server_args,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
    )

    log.info("ModelWorker initialized, NCCL group joined")

    # 3. Get the NCCL CPU group for broadcast_pyobj
    tp_cpu_group = worker.model_runner.tp_group.cpu_group

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.utils import broadcast_pyobj

    device = torch.device("cuda", gpu_id)
    model_vocab_size = worker.model_runner.model_config.vocab_size

    # 4. Main loop: receive batch, forward, repeat
    step = 0
    while True:
        result = broadcast_pyobj([None], tp_rank, tp_cpu_group, src=0)
        batch = result[0] if result else None

        if batch is None:
            log.info("Received stop signal after %d steps", step)
            break

        patch_batch_for_follower(batch, device, vocab_size=model_vocab_size)
        sync_page_table(batch, worker.model_runner.req_to_token_pool)
        forward_batch = ForwardBatch.init_new(batch, worker.model_runner)
        worker.model_runner.forward(forward_batch=forward_batch)
        step += 1

    log.info("Follower exiting")


def spawn_followers(
    server_args: Any,
    nccl_port: int,
    base_gpu_id: int,
    tp_size: int,
) -> list[mp.Process]:
    """Spawn follower worker processes for TP ranks 1..tp_size-1.

    Must be called BEFORE rank 0's ModelWorker.__init__ because
    init_distributed_environment is a collective operation.

    Args:
        server_args: SGLang ServerArgs.
        nccl_port: NCCL rendezvous port.
        base_gpu_id: GPU ID for rank 0. Followers get base_gpu_id + rank.
        tp_size: Total TP size (including rank 0).

    Returns:
        List of mp.Process handles for followers.
    """
    processes = []

    for rank in range(1, tp_size):
        gpu_id = base_gpu_id + rank
        proc = mp.Process(
            target=follower_worker_loop,
            args=(rank, gpu_id, server_args, nccl_port),
            daemon=True,
        )
        proc.start()
        processes.append(proc)
        logger.info(
            "Spawned follower rank %d on GPU %d (pid=%d)", rank, gpu_id, proc.pid
        )

    # Don't wait here — followers block on init_distributed_environment
    # until rank 0 also calls it. The factory will create rank 0's ModelWorker
    # next, which unblocks everyone.

    return processes
