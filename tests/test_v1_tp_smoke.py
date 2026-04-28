# SPDX-License-Identifier: Apache-2.0
"""Manual TP smoke targets that require a multi-GPU CUDA host."""
from __future__ import annotations

import os

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_TP_SMOKE") != "1",
    reason="manual smoke requires RUN_TP_SMOKE=1 and >=2 CUDA GPUs",
)
def test_two_rank_recv_round_trips() -> None:
    """Spawn a 2-rank thinker stage, send a no-op/text request, and verify
    rank 1 observes the request through the OmniScheduler.recv_requests
    broadcast path.
    """
    pytest.xfail("implement during execution")
