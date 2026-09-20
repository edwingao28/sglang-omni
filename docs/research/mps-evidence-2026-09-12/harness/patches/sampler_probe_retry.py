"""Idempotent: make the CPU sampler retry its MPS readiness probe instead of dying on one transient failure.

  python3 sampler_probe_retry.py /work/campaigns/mps-evidence-20260912-a01/control/sample-process-cpu-v3.py
"""
import hashlib
import sys
from pathlib import Path

OLD = " daemon=client.read_daemon_identity(paths.pipe_dir)\n references=client.snapshot(paths.pipe_dir)\n"
NEW = (" # Note (wenyao): one get_server_list probe failed transiently at readiness (2026-09-13T02:46Z) while the daemon\n"
       " # and servers were alive; retry the readiness probe instead of dying before measurement.\n"
       " from sglang_omni.mps.manager import MpsControlError\n"
       " for attempt in range(1,6):\n"
       "  try:\n"
       "   daemon=client.read_daemon_identity(paths.pipe_dir)\n"
       "   references=client.snapshot(paths.pipe_dir)\n"
       "   break\n"
       "  except MpsControlError as error:\n"
       "   if attempt==5:raise\n"
       "   sys.stderr.write(f'MPS readiness probe failed ({attempt}/5): {error}; retrying in 2 s\\n');sys.stderr.flush();time.sleep(2)\n")
path = Path(sys.argv[1])
text = path.read_text()
if NEW in text:
    print('already applied', hashlib.md5(text.encode()).hexdigest())
elif OLD in text:
    path.write_text(text.replace(OLD, NEW))
    print('applied', hashlib.md5(path.read_bytes()).hexdigest())
else:
    sys.exit('anchor not found')
