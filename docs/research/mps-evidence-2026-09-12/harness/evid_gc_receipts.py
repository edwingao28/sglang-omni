"""Print the per-rank Green Context receipts of a sustained run (login node or container).

  python3 evid_gc_receipts.py <run_dir> [<run_dir> ...]
"""
import json
import sys
from pathlib import Path

for run_dir in sys.argv[1:]:
    d = json.loads((Path(run_dir) / 'run.json').read_text())
    print(f"== {d['run_id']} status={d.get('status')} placement={d.get('placement')} sms={d.get('sms')} expected_actual_sms={d.get('expected_actual_sms')} "
          f"disjoint_proven={d.get('globally_disjoint_replica_sms_proven')} kernel_ownership={d.get('actual_model_kernel_ownership')}")
    for r in d.get('resource_setup', []):
        x = r['receipt']; res = x.get('resource', {})
        cap = x.get('captures', {})
        print(f"  rank {x.get('rank')}: {x.get('status')} group_index={res.get('group_index')} selected={res.get('selected_group_indices')} "
              f"actual_sms={res.get('actual_sms')} group_sms={res.get('group_sms')} remainder={res.get('remainder_sms')} union_next={res.get('union_next')} "
              f"decode_graphs={cap.get('decode_capture_batch_sizes')} predictor_graphs={cap.get('predictor_graph_count')} "
              f"graphs_on_green_stream={cap.get('decode_capture_stream_handle') == res.get('stream_handle') and cap.get('predictor_stream_handle') == res.get('stream_handle')}")
