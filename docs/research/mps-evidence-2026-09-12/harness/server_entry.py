"""Record the exact owned serve PID, then exec the unchanged command in place."""
import os
from pathlib import Path
import sys
import time

from nsys_support import process_identity, save

receipt = Path(sys.argv[1])
argv = sys.argv[2:]
if not argv:
    raise RuntimeError('serve argv required')
identity = process_identity(os.getpid())
save(receipt, {**identity, 'argv': argv, 'created_ns': time.time_ns(),
              'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
os.execvpe(argv[0], argv, os.environ)
