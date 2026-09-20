"""Indexed CUDA SM partitions for the evidence campaign; CUDA imported on demand.

Derived from round-1 gc-resources-v3. Adds `union2`: one Green Context built from
two adjacent groups of the same split, so replica ranks overlap pairwise
(rank i owns groups i and (i+1) mod n). Cross-process disjointness stays an
empirical gate (CUPTI TPC masks in the trace), never inferred from indices.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


def checked(result: tuple[Any, ...], operation: str) -> tuple[Any, ...]:
    if int(result[0]) != 0:
        raise RuntimeError(f"{operation}: {result}")
    return result[1:]


def validate_request(sms: int | None, group_index: int, group_count: int) -> None:
    if type(group_count) is not int or group_count < 1:
        raise ValueError("group_count must be a positive integer")
    if type(group_index) is not int or not 0 <= group_index < group_count:
        raise ValueError("group_index must select an existing requested group")
    if sms is not None and (type(sms) is not int or sms < 1):
        raise ValueError("sms must be a positive integer or None")
    if "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" in os.environ:
        raise ValueError("This experiment requires MPS active thread percentage unset")


def describe_stream(stream: Any, driver: Any) -> dict[str, int]:
    handle = driver.CUstream(stream.cuda_stream)
    context, = checked(driver.cuStreamGetCtx(handle), "cuStreamGetCtx")
    resource, = checked(driver.cuCtxGetDevResource(
        context, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
    ), "cuCtxGetDevResource")
    stream_id, = checked(driver.cuStreamGetId(handle), "cuStreamGetId")
    return {"stream_handle": int(stream.cuda_stream), "stream_id": int(stream_id),
            "context_handle": int(context), "actual_sms": int(resource.sm.smCount)}


@dataclass
class GreenStreamOwner:
    """Retain all driver resources until the owning model process exits."""

    resource: Any
    groups: Any
    descriptor: Any
    green_context: Any
    stream_handle: Any


def create_stream(device_id: int, sms: int | None = None, group_index: int = 0,
                  group_count: int = 3, union_next: bool = False) -> tuple[Any, Any, dict[str, Any]]:
    validate_request(sms, group_index, group_count)
    import torch
    from cuda.bindings import driver

    torch.cuda.set_device(device_id)
    torch.cuda.init()
    owner = None
    split_receipt: dict[str, Any] = {}
    if sms is None:
        stream = torch.cuda.Stream(device=device_id)
    else:
        device, = checked(driver.cuDeviceGet(device_id), "cuDeviceGet")
        resource, = checked(driver.cuDeviceGetDevResource(
            device, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        ), "cuDeviceGetDevResource")
        groups, actual_groups, remainder = checked(driver.cuDevSmResourceSplitByCount(
            group_count, resource, 0, sms
        ), "cuDevSmResourceSplitByCount")
        if int(actual_groups) < group_count or len(groups) < group_count:
            raise RuntimeError(f"Requested {group_count} groups of {sms} SMs; got {actual_groups}")
        selected_indices = [group_index]
        if union_next:
            selected_indices.append((group_index + 1) % group_count)
        selected = [groups[i] for i in selected_indices]
        for part in selected:
            if int(part.sm.smCount) != sms:
                raise RuntimeError(f"Requested exact {sms} SMs; split returned {part.sm.smCount}")
        descriptor, = checked(driver.cuDevResourceGenerateDesc(selected, len(selected)),
                              "cuDevResourceGenerateDesc")
        green, = checked(driver.cuGreenCtxCreate(
            descriptor, device, int(driver.CUgreenCtxCreate_flags.CU_GREEN_CTX_DEFAULT_STREAM)
        ), "cuGreenCtxCreate")
        handle, = checked(driver.cuGreenCtxStreamCreate(
            green, int(driver.CUstream_flags.CU_STREAM_NON_BLOCKING), 0
        ), "cuGreenCtxStreamCreate")
        stream = torch.cuda.ExternalStream(int(handle), device=device_id)
        owner = GreenStreamOwner(resource, groups, descriptor, green, handle)
        split_receipt = {
            "device_sms": int(resource.sm.smCount), "split_flags": 0,
            "min_partition_size": int(resource.sm.minSmPartitionSize),
            "coscheduled_alignment": int(resource.sm.smCoscheduledAlignment),
            "group_count": int(actual_groups), "group_index": group_index,
            "selected_group_indices": selected_indices,
            "group_sms": [int(r.sm.smCount) for r in groups[:int(actual_groups)]],
            "remainder_sms": int(remainder.sm.smCount),
        }
    receipt = {**describe_stream(stream, driver), **split_receipt,
               "requested_sms": sms, "union_next": union_next, "pid": os.getpid(),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
               "cuda_mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
               "globally_disjoint_replica_sms_proven": False}
    expected = None if sms is None else sms * (2 if union_next else 1)
    if expected is not None and receipt["actual_sms"] != expected:
        raise RuntimeError(f"Stream has {receipt['actual_sms']} SMs, wanted {expected}")
    return owner, stream, receipt
