"""Probe H100 SM partition granularity and the union2 Green Context path (seconds, lane GPU)."""
import json
import os

import torch
from cuda.bindings import driver

from gc_resources import checked, create_stream

torch.cuda.init()
torch.cuda.set_device(0)
device, = checked(driver.cuDeviceGet(0), 'cuDeviceGet')
resource, = checked(driver.cuDeviceGetDevResource(device, driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM), 'cuDeviceGetDevResource')
print(json.dumps(dict(cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'), device_sms=int(resource.sm.smCount),
    min_partition=int(resource.sm.minSmPartitionSize), coscheduled_alignment=int(resource.sm.smCoscheduledAlignment))))
for count, sms in [(3, 44), (3, 40), (3, 32), (4, 32), (4, 33), (4, 30), (2, 66), (2, 64), (3, 48), (4, 28), (1, 132), (2, 88)]:
    try:
        groups, n, rem = checked(driver.cuDevSmResourceSplitByCount(count, resource, 0, sms), 'split')
        print(json.dumps(dict(requested_groups=count, requested_sms=sms, groups=int(n),
            group_sms=[int(g.sm.smCount) for g in groups[:int(n)]], remainder=int(rem.sm.smCount))))
    except Exception as error:  # noqa: BLE001
        print(json.dumps(dict(requested_groups=count, requested_sms=sms, error=repr(error))))
for placement, sms, rank, count, union in [('indexed', 44, 1, 3, False), ('union2', 44, 2, 3, True), ('indexed', 32, 3, 4, False)]:
    try:
        owner, stream, receipt = create_stream(0, sms, rank, count, union)
        with torch.cuda.stream(stream):
            x = torch.ones(1024, 1024, device='cuda')
            y = (x @ x).sum()
        stream.synchronize()
        print(json.dumps(dict(placement=placement, ok=True, value=float(y), receipt={k: receipt[k] for k in (
            'actual_sms', 'group_sms', 'selected_group_indices', 'remainder_sms', 'min_partition_size', 'coscheduled_alignment')})))
    except Exception as error:  # noqa: BLE001
        print(json.dumps(dict(placement=placement, sms=sms, ok=False, error=repr(error))))
