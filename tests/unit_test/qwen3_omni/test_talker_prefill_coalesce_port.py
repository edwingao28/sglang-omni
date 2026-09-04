# SPDX-License-Identifier: Apache-2.0
"""CPU contract checks executing unchanged scheduler/factory bodies with fake hardware.

Only dependency imports and upstream admission are replaced. The production gate,
Talker counter methods, and both complete factory functions execute from source.
This does not qualify SGLang/CUDA integration or generated-audio correctness.
"""

from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "sglang_omni/models/qwen3_omni"


def _function(path, name):
    return next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)


def _methods(path, cls, names):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]


def _execute(nodes, namespace):
    tree = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), "<actual-production-body>", "exec"), namespace)


def _mode(extend=True):
    return NS(is_extend=lambda: extend)


def _scheduler_types():
    clock = NS(now=100.0, perf_counter=lambda: clock.now)
    calls = []

    class Upstream:
        @staticmethod
        def get_new_batch_prefill(s, running):
            calls.append(tuple(s.waiting_queue))
            admitted = s.waiting_queue[:s.admit_limit]
            del s.waiting_queue[:len(admitted)]
            return NS(batch_to_run=NS(reqs=admitted, forward_mode=_mode()), running_batch=running)

    gate = _methods(ROOT / "sglang_omni/scheduling/omni_scheduler.py", "OmniScheduler", {"get_new_batch_prefill"})
    base = ast.ClassDef(name="OmniScheduler", bases=[], keywords=[], body=gate, decorator_list=[])
    talker = ast.ClassDef(name="QwenTalkerScheduler", bases=[ast.Name(id="OmniScheduler", ctx=ast.Load())], keywords=[], body=_methods(MODEL / "talker_scheduler.py", "QwenTalkerScheduler", {"get_new_batch_prefill", "_admin_model_info"}), decorator_list=[])
    ns = {"_Upstream": Upstream, "time": clock, "NextBatchPlan": lambda **kw: NS(**kw)}
    _execute([base, talker], ns)
    ns["OmniScheduler"]._admin_model_info = lambda self: {"success": True, "data": {"existing": 7}}
    cls = ns["QwenTalkerScheduler"]

    def make(target=4, wait=40.0, idle=False):
        s = cls()
        s.prefill_coalesce_requests = target
        s.prefill_coalesce_wait_s = wait / 1000
        s.prefill_coalesce_when_idle = idle
        s.prefill_coalesce_requires_pending_builds = False
        s.prefill_coalesce_after_builds_during_decode = False
        s.chunked_req = None
        s.waiting_queue = []
        s.admit_limit = 99
        return s

    return make, clock, calls, ns


class GateTests(unittest.TestCase):
    def setUp(self):
        self.make, self.clock, self.calls, self.ns = _scheduler_types()
        self.running = NS(is_empty=lambda: False)

    def test_disabled_has_no_hold(self):
        s = self.make(target=0)
        s.waiting_queue = [NS(_coalesce_enqueue_t=100.0)]
        self.assertEqual(len(s.get_new_batch_prefill(self.running).batch_to_run.reqs), 1)

    def test_target_releases_same_order(self):
        s = self.make()
        rows = [NS(rid=str(i), _coalesce_enqueue_t=100.0) for i in range(4)]
        s.waiting_queue = rows.copy()
        self.assertEqual(s.get_new_batch_prefill(self.running).batch_to_run.reqs, rows)
        self.assertEqual(s._prefill_batch_histogram, {4: 1})

    def test_timeout_does_not_restart_and_fifo_is_preserved(self):
        s = self.make()
        a, b = NS(rid="a"), NS(rid="b", _coalesce_enqueue_t=100.01)
        s.waiting_queue = [a, b]
        self.assertIsNone(s.get_new_batch_prefill(self.running).batch_to_run)
        self.clock.now = 100.039
        self.assertIsNone(s.get_new_batch_prefill(self.running).batch_to_run)
        self.assertEqual(a._coalesce_enqueue_t, 100.0)
        self.clock.now = 100.041
        self.assertEqual(s.get_new_batch_prefill(self.running).batch_to_run.reqs, [a, b])
        self.assertEqual(s._prefill_batch_histogram, {2: 1})

    def test_partial_admission_preserves_leftover_deadline(self):
        s = self.make()
        rows = [NS(rid=str(i), _coalesce_enqueue_t=99.0) for i in range(3)]
        s.waiting_queue = rows.copy()
        s.admit_limit = 1
        self.assertEqual(s.get_new_batch_prefill(self.running).batch_to_run.reqs, rows[:1])
        self.assertEqual(s.get_new_batch_prefill(self.running).batch_to_run.reqs, rows[1:2])
        self.assertEqual(s.waiting_queue[0]._coalesce_enqueue_t, 99.0)

    def test_aborted_oldest_does_not_release_newcomer_early(self):
        s = self.make()
        old = NS(_coalesce_enqueue_t=99.0)
        new = NS(_coalesce_enqueue_t=100.0)
        s.waiting_queue = [old, new]
        s.waiting_queue.remove(old)
        self.assertIsNone(s.get_new_batch_prefill(self.running).batch_to_run)

    def test_idle_and_chunked_prefill_bypass(self):
        for running, chunked in [(None, None), (NS(is_empty=lambda: True), None), (self.running, object())]:
            with self.subTest(running=running, chunked=chunked):
                s = self.make()
                s.chunked_req = chunked
                s.waiting_queue = [NS(_coalesce_enqueue_t=100.0)]
                self.assertIsNotNone(s.get_new_batch_prefill(running).batch_to_run)

    def test_counter_ignores_decode_and_empty_plans(self):
        s = self.make()
        plan = NS(batch_to_run=NS(reqs=[1], forward_mode=_mode(False)))
        self.ns["OmniScheduler"].get_new_batch_prefill = lambda self, running: plan
        self.assertIs(s.get_new_batch_prefill(None), plan)
        self.assertFalse(hasattr(s, "_prefill_batch_histogram"))
        plan.batch_to_run = None
        self.assertIs(s.get_new_batch_prefill(None), plan)
        self.assertFalse(hasattr(s, "_prefill_batch_histogram"))

    def test_admin_policy_is_detached_and_serializable(self):
        s = self.make()
        s._prefill_batch_histogram = {1: 3, 4: 2}
        reply = s._admin_model_info()
        self.assertEqual(reply["data"]["existing"], 7)
        info = reply["data"]["talker_prefill_batching"]
        self.assertEqual(info, {"target_requests": 4, "max_wait_ms": 40.0, "when_idle": False, "batch_histogram": {"1": 3, "4": 2}})
        self.assertEqual(json.loads(json.dumps(info)), info)
        import msgpack
        self.assertEqual(msgpack.unpackb(msgpack.packb(info), strict_map_key=True), info)
        info["batch_histogram"].clear()
        self.assertEqual(s._prefill_batch_histogram, {1: 3, 4: 2})


class FactoryTests(unittest.TestCase):
    def test_complete_stage_and_bootstrap_forward_to_scheduler_only(self):
        for target in [0, 4]:
            with self.subTest(target=target):
                self._run_factory(target)

    def test_lookahead_with_cache_and_batching_binds_only_the_talker_abort_predicate(self):
        for enabled in [False, True]:
            with self.subTest(enabled=enabled):
                self._run_factory(4, lookahead=enabled)

    def test_scratch_option_binds_before_capture_on_identical_initialization_path(self):
        for skip_scratch in [False, True]:
            for graph_enabled in [False, True]:
                with self.subTest(skip_scratch=skip_scratch, graph_enabled=graph_enabled):
                    self._run_factory(4, lookahead=True, skip_scratch=skip_scratch, graph_enabled=graph_enabled)

    def _run_factory(self, target, lookahead=None, skip_scratch=None, graph_enabled=False):
        observed = {}
        cfg = NS(model_path="test-model", hf_config=NS(
            thinker_config=NS(audio_token_id=1, image_token_id=2, video_token_id=3),
            talker_config=NS(text_config=NS(vocab_size=4096), accept_hidden_layer=4,
                codec_bos_id=5, codec_eos_token_id=6, codec_nothink_id=7,
                codec_think_bos_id=8, codec_think_eos_id=9, codec_pad_id=10, speaker_id={}),
            tts_bos_token_id=11, tts_eos_token_id=12, tts_pad_token_id=13,
            im_start_token_id=14, im_end_token_id=15, system_token_id=16,
            user_token_id=17, assistant_token_id=18))
        phases = []
        model = NS()
        def configure_scratch(*, skip_unused):
            phases.append("configure")
            observed["skip_scratch"] = skip_unused
        model.configure_predictor_scratch_writes = configure_scratch
        worker = NS(model_runner=NS(model_config=cfg, model=model, sampler=object()))
        def infrastructure(*args, **kwargs):
            phases.append("infrastructure")
            self.assertIs(kwargs["defer_cuda_graph_capture"], graph_enabled)
            kwargs["model_post_load_hook"](model)
            if not graph_enabled:
                init_graphs(worker)
            return worker, None, None, None, cfg
        def init_graphs(value):
            self.assertIs(value, worker)
            self.assertIn("skip_scratch", observed)
            self.assertEqual(hasattr(model, "_sampler"), graph_enabled)
            phases.append("init")
        def adapters(**kw):
            observed["adapters"] = kw
            return tuple(object() for _ in range(4))
        class FakeScheduler:
            def __init__(self, **kw):
                observed["scheduler"] = kw
                self.outbox = object()
                observed["scheduler_instance"] = self
            def is_request_aborted(self, request_id):
                return request_id == "cancelled"
            def bind_model_runner(self, runner):
                observed["runner"] = runner
        exports = dict(get_tokenizer=lambda *a, **kw: object(),
            current_platform=NS(enable_talker_graph=lambda: True),
            make_talker_scheduler_adapters=adapters, QwenTalkerModelRunner=lambda *a, **kw: NS(**kw),
            QwenTalkerScheduler=FakeScheduler, configure_talker_server_args=lambda *a, **kw: graph_enabled,
            create_sglang_infrastructure=infrastructure,
            init_sglang_cuda_graphs=init_graphs,
            SGLangOutputProcessor=lambda **kw: NS(**kw))
        real_import = builtins.__import__
        def controlled_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("sglang"):
                return NS(**{n: exports[n] for n in fromlist})
            return real_import(name, globals, locals, fromlist, level)
        ns = {"__builtins__": {**vars(builtins), "__import__": controlled_import}}
        _execute([_function(MODEL / "bootstrap.py", "create_talker_scheduler")], ns)
        exports["create_talker_scheduler"] = ns["create_talker_scheduler"]
        ns.update(build_generation_batch_overrides=lambda **kw: {},
            _apply_colocated_ar_memory_contract=lambda *a, **kw: None,
            build_sglang_server_args=lambda *a, **kw: NS(mem_fraction_static=0.123),
            validate_generation_batch_policy=lambda **kw: None,
            avail_gpu_mem=lambda *a: 1, get_process_gpu_memory_bytes=lambda *a: 0,
            format_bytes_gib=str, os=NS(getpid=lambda: 1), logger=NS(info=lambda *a: None))
        _execute([_function(MODEL / "stages.py", "create_talker_ar_executor_from_config")], ns)
        stage = ns["create_talker_ar_executor_from_config"]
        opts = {} if target == 0 else dict(prefill_coalesce_requests=4, prefill_coalesce_wait_ms=40.0, prefill_coalesce_when_idle=False)
        for factory in [stage, ns["create_talker_scheduler"]]:
            self.assertIs(inspect.signature(factory).parameters["code_predictor_skip_scratch_writes"].default, False)
            self.assertIs(inspect.signature(factory).parameters["enable_async_decode"].default, False)
            self.assertEqual(inspect.signature(factory).parameters["async_decode_min_batch_size"].default, 2)
        if lookahead is not None:
            opts.update(enable_async_decode=lookahead, async_decode_min_batch_size=3)
        if skip_scratch is not None:
            opts["code_predictor_skip_scratch_writes"] = skip_scratch
        stage("test-model", assistant_projection_cache_size=4096, **opts)
        self.assertEqual(phases, ["infrastructure", "configure", "init"])
        self.assertIs(observed["skip_scratch"], bool(skip_scratch))
        self.assertIs(observed["scheduler"]["enable_async_decode"], bool(lookahead))
        self.assertEqual(observed["scheduler"]["async_decode_min_batch_size"], 2 if lookahead is None else 3)
        predicate = observed["runner"].request_is_aborted
        if lookahead:
            self.assertIs(predicate.__self__, observed["scheduler_instance"])
            self.assertTrue(predicate("cancelled"))
            self.assertFalse(predicate("live"))
        else:
            self.assertIsNone(predicate)
        self.assertNotIn("enable_async_decode", observed["adapters"])
        self.assertNotIn("async_decode_min_batch_size", observed["adapters"])
        self.assertEqual(observed["scheduler"]["prefill_coalesce_requests"], target)
        self.assertEqual(observed["scheduler"]["prefill_coalesce_wait_ms"], 40.0)
        self.assertFalse(observed["scheduler"]["prefill_coalesce_when_idle"])
        self.assertEqual(observed["adapters"]["assistant_projection_cache_size"], 4096)
        self.assertFalse(any(k.startswith("prefill_coalesce") for k in observed["adapters"]))
        self.assertNotIn("prefill_coalesce_requests", observed["runner"].__dict__)


    def test_infrastructure_hook_preserves_capture_and_cache_lifecycle(self):
        for deferred in [False, True]:
            for use_hook in [False, True]:
                with self.subTest(deferred=deferred, use_hook=use_hook):
                    events = []
                    model, cfg, req_pool, kv_pool, cache = [object() for _ in range(5)]
                    runner = NS(
                        model=model,
                        alloc_memory_pool=lambda: events.append("alloc"),
                        init_attention_backends=lambda: events.append("attention"),
                    )
                    worker = NS(model_runner=runner, model_config=cfg)
                    def pools():
                        events.append("get_pool")
                        return req_pool, kv_pool
                    worker.get_memory_pool = pools
                    def make_worker(**kwargs):
                        events.append("worker")
                        return worker
                    def make_cache(args, req, kv, page):
                        self.assertIs(req, req_pool)
                        self.assertIs(kv, kv_pool)
                        events.append("tree_cache")
                        return cache
                    def hook(actual):
                        self.assertIs(actual, model)
                        events.append("hook")
                    exports = dict(
                        get_context=lambda: NS(is_config_namespace_published=lambda name: False),
                        consume_stage_kv_cache_bytes=lambda: None,
                        use_mlx=lambda: False,
                        ModelWorker=make_worker, ModelWorkerConfig=lambda **kw: NS(**kw),
                        create_tree_cache=make_cache,
                    )
                    original_import = builtins.__import__
                    def dependency_import(name, globals=None, locals=None, fromlist=(), level=0):
                        if name.startswith("sglang"):
                            return NS(**{key: exports[key] for key in fromlist})
                        return original_import(name, globals, locals, fromlist, level)
                    ns = {
                        "__builtins__": {**vars(builtins), "__import__": dependency_import},
                        "logger": NS(info=lambda *args: None),
                        "_describe_sglang_runtime_configuration": lambda *args: "test",
                        "init_sglang_cuda_graphs": lambda actual: events.append("init"),
                    }
                    source = ROOT / "sglang_omni/scheduling/bootstrap.py"
                    _execute([_function(source, "create_sglang_infrastructure")], ns)
                    result = ns["create_sglang_infrastructure"](
                        NS(page_size=1), 0, defer_cuda_graph_capture=deferred,
                        model_post_load_hook=hook if use_hook else None,
                    )
                    expected = ["worker", "alloc", "attention"]
                    if use_hook:
                        expected.append("hook")
                    if not deferred:
                        expected.append("init")
                    expected.extend(["get_pool", "tree_cache"])
                    self.assertEqual(events, expected)
                    self.assertEqual(result, (worker, cache, req_pool, kv_pool, cfg))


if __name__ == "__main__":
    unittest.main()
