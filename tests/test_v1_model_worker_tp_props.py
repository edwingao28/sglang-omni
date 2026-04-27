"""ModelWorker must expose tp_cpu_group and tp_size for the OmniScheduler broadcast."""
from types import SimpleNamespace


def test_tp_cpu_group_returns_underlying_runner_cpu_group():
    from sglang_omni_v1.model_runner.model_worker import ModelWorker

    cpu_group = object()
    fake_tp_group = SimpleNamespace(cpu_group=cpu_group)
    fake_runner = SimpleNamespace(tp_group=fake_tp_group)
    worker = ModelWorker.__new__(ModelWorker)
    worker.model_runner = fake_runner
    assert worker.tp_cpu_group is cpu_group


def test_tp_size_returns_server_args_value():
    from sglang_omni_v1.model_runner.model_worker import ModelWorker

    server_args = SimpleNamespace(tp_size=4)
    worker = ModelWorker.__new__(ModelWorker)
    worker.server_args = server_args
    assert worker.tp_size == 4
