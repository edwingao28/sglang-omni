# SPDX-License-Identifier: Apache-2.0
"""Follower worker loop for TP ranks > 0."""
from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any

logger = logging.getLogger(__name__)


def register_omni_models() -> None:
    """Register omni models in SGLang's registry."""
    from sglang_omni.models.sglang_registry import register_omni_models_in_sglang

    register_omni_models_in_sglang()


def relocate_batch_tensors(batch, device) -> None:
    """Move all tensors in *batch* to *device*."""
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
    """Write rank 0's page-table snapshot into the follower pool."""
    rows = getattr(batch, "tp_page_table_rows", None)
    if not rows:
        return
    pool_tensor = req_to_token_pool.req_to_token
    for i, row in enumerate(rows):
        idx = int(batch.req_pool_indices[i])
        seq_len = len(row)
        pool_tensor[idx, :seq_len] = row.to(pool_tensor.device)


def patch_batch_for_follower(batch, device, vocab_size: int = 0) -> None:
    """Restore sanitized batch fields and relocate tensors."""
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


def _forward_with_deepstack(model_runner, forward_batch, deepstack_visual_embeds,
                            visual_pos_masks) -> None:
    """Run the deepstack path followers need to mirror rank 0."""
    import torch

    model_runner.attn_backend.init_forward_metadata(forward_batch)

    input_embeds = forward_batch.input_embeds
    device = input_embeds.device
    dtype = input_embeds.dtype

    positions = forward_batch.positions
    if forward_batch.mrope_positions is not None:
        positions = forward_batch.mrope_positions

    layer_tensors = [t.to(device=device, dtype=dtype) for t in deepstack_visual_embeds]
    ds_input = torch.cat(layer_tensors, dim=-1)
    full_ds = torch.zeros(
        input_embeds.shape[0], ds_input.shape[-1], device=device, dtype=dtype,
    )
    full_ds[visual_pos_masks] = ds_input

    model = model_runner.model
    outer = model.thinker if hasattr(model, "thinker") else model

    hidden_states = outer.model(
        input_ids=None,
        positions=positions,
        forward_batch=forward_batch,
        input_embeds=input_embeds,
        input_deepstack_embeds=full_ds,
    )

    # Followers must join the LM-head collectives too.
    outer.logits_processor(
        forward_batch.input_ids,
        hidden_states,
        outer.lm_head,
        forward_batch,
    )


def follower_worker_loop(
    tp_rank: int,
    gpu_id: int,
    server_args: Any,
    nccl_port: int,
) -> None:
    """Entry point for a follower TP worker process."""
    import torch

    torch.cuda.set_device(gpu_id)
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s [TP{tp_rank}] %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(f"tp_follower.{tp_rank}")
    log.info("Starting follower on GPU %d", gpu_id)

    register_omni_models()

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

    tp_cpu_group = worker.model_runner.tp_group.cpu_group

    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.utils import broadcast_pyobj

    device = torch.device("cuda", gpu_id)
    model_vocab_size = worker.model_runner.model_config.vocab_size

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

        ds_embeds = getattr(batch, "tp_deepstack_visual_embeds", None)
        vis_masks = getattr(batch, "tp_visual_pos_masks", None)

        if forward_batch.input_embeds is not None and ds_embeds is not None:
            _forward_with_deepstack(
                worker.model_runner, forward_batch, ds_embeds, vis_masks,
            )
        else:
            worker.model_runner.forward(forward_batch=forward_batch)
        step += 1

    log.info("Follower exiting")


def spawn_followers(
    server_args: Any,
    nccl_port: int,
    base_gpu_id: int,
    tp_size: int,
) -> list[mp.Process]:
    """Spawn TP follower processes."""
    processes = []
    ctx = mp.get_context("spawn")

    for rank in range(1, tp_size):
        gpu_id_step = getattr(server_args, "gpu_id_step", 1)
        gpu_id = base_gpu_id + rank * gpu_id_step
        proc = ctx.Process(
            target=follower_worker_loop,
            args=(rank, gpu_id, server_args, nccl_port),
            daemon=True,
        )
        proc.start()
        processes.append(proc)
        logger.info(
            "Spawned follower rank %d on GPU %d (pid=%d)", rank, gpu_id, proc.pid
        )

    return processes
