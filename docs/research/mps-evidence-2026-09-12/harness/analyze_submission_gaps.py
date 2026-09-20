"""Describe recorded kernel gaps relative to host launch completion, without causality claims.

(Verbatim copy of the round-1 analyzer.)
"""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from typing import NamedTuple

from analyze_trace import occupancy, quantiles


class Kernel(NamedTuple):
    start: int
    end: int
    global_pid: int
    correlation: int
    graph_node: int
    name: str


class Launch(NamedTuple):
    start: int
    end: int
    global_pid: int
    correlation: int
    name: str


SUBMITTED = 'recorded_future_kernel_launch_completed'
BEFORE_COMPLETION = 'before_next_recorded_future_launch_completion'
UNRESOLVED = 'future_kernel_launch_unresolved'
TAIL = 'no_future_recorded_kernel_in_window'


def merge_intervals(intervals):
    merged = []
    for left, right in sorted(intervals):
        if right <= left:
            continue
        if merged and left <= merged[-1][1]:
            merged[-1][1] = max(right, merged[-1][1])
        else:
            merged.append([left, right])
    return merged


def overlap_ns(first, second):
    """Intersect two ordered, internally disjoint interval sets."""
    total = a = b = 0
    while a < len(first) and b < len(second):
        total += max(0, min(first[a][1], second[b][1]) - max(first[a][0], second[b][0]))
        if first[a][1] <= second[b][1]:
            a += 1
        else:
            b += 1
    return total


def classify_gaps(gaps, kernels, launches):
    """Partition each gap at the earliest known completion of a future kernel's launch.

    Kernels must be the selected workload window only. A suffix minimum keeps
    this O((kernels + gaps) log kernels), including many nodes per graph launch.
    """
    by_key = defaultdict(list)
    for launch in launches:
        by_key[(launch.global_pid, launch.correlation)].append(launch)
    kernels = sorted(kernels)
    starts = [kernel.start for kernel in kernels]
    best = [None] * (len(kernels) + 1)
    missing = [0] * (len(kernels) + 1)
    for index in range(len(kernels) - 1, -1, -1):
        kernel = kernels[index]
        candidates = by_key[(kernel.global_pid, kernel.correlation)]
        best[index] = best[index + 1]
        missing[index] = missing[index + 1] + (len(candidates) != 1)
        if len(candidates) == 1:
            launch = candidates[0]
            if best[index] is None or launch.end <= best[index][1].end:
                best[index] = (kernel, launch)
    totals, segment_counts, whole_gap_counts = Counter(), Counter(), Counter()
    submitted_graph_witness_ns = 0
    submitted_example = graph_example = None
    examples = []
    for left, right in gaps:
        index = bisect_left(starts, left)
        future = best[index]
        if index == len(kernels):
            segments = [(left, right, TAIL)]
        elif future is None:
            segments = [(left, right, UNRESOLVED)]
        else:
            completion = future[1].end
            split = min(right, max(left, completion))
            prefix = UNRESOLVED if missing[index] else BEFORE_COMPLETION
            segments = ([(left, split, prefix)] if split > left else [])
            if split < right:
                segments.append((split, right, SUBMITTED))
        for a, b, label in segments:
            totals[label] += b-a
            segment_counts[label] += 1
            if label == SUBMITTED and 'GraphLaunch' in future[1].name:
                submitted_graph_witness_ns += b-a
            if label == SUBMITTED:
                is_graph = 'GraphLaunch' in future[1].name
                longest = submitted_example is None or b-a > submitted_example['duration_ns']
                longest_graph = is_graph and (graph_example is None or b-a > graph_example['duration_ns'])
                if longest or longest_graph:
                    item = {'interval_ns':[a,b], 'duration_ns':b-a,
                            'witness_kernel':future[0]._asdict(), 'witness_launch':future[1]._asdict()}
                    if longest:
                        submitted_example = item
                    if longest_graph:
                        graph_example = item
        whole_gap_counts['+'.join(label for _, _, label in segments)] += 1
        # Keep bounded examples; durations/denominators cover every gap.
        if len(examples) < 20 or right-left > examples[-1]['duration_ns']:
            item = {'gap_ns': [left, right], 'duration_ns': right-left,
                    'segments': [{'interval_ns': [a,b], 'classification': label} for a,b,label in segments],
                    'unresolved_future_kernel_records': missing[index]}
            if future:
                item['witness_kernel'] = future[0]._asdict()
                item['witness_launch'] = future[1]._asdict()
            examples.append(item)
            examples.sort(key=lambda item: item['duration_ns'], reverse=True)
            del examples[20:]
    zero_ns = sum(right-left for left, right in gaps)
    assert sum(totals.values()) == zero_ns
    return {'zero_kernel_ns': zero_ns, 'gap_count': len(gaps),
            'duration_ns_by_class': dict(totals),
            'fraction_of_zero_kernel_time_by_class': {key: value/zero_ns for key,value in totals.items()} if zero_ns else {},
            'segment_count_by_class': dict(segment_counts),
            'whole_gap_count_by_class_sequence': dict(whole_gap_counts),
            'submitted_duration_with_graph_launch_witness_ns': submitted_graph_witness_ns,
            'kernels_with_missing_or_ambiguous_launch': missing[0],
            'longest_submitted_segment': submitted_example,
            'longest_submitted_graph_witness_segment': graph_example,
            'longest_gaps': examples}


def api_overlap(rows, gaps, start, end):
    """API wall durations and union overlap; concurrent rows are not additive costs."""
    by_name = defaultdict(list)
    for left, right, name in rows:
        a, b = max(start,left), min(end,right)
        if b > a:
            by_name[name].append((a,b))
    all_intervals = [interval for intervals in by_name.values() for interval in intervals]
    union = merge_intervals(all_intervals)
    return {'api_records_in_window': len(all_intervals),
            'summed_clipped_api_wall_ns': sum(b-a for a,b in all_intervals),
            'api_wall_union_ns': sum(b-a for a,b in union),
            'zero_kernel_gap_overlap_union_ns': overlap_ns(union,gaps),
            'by_name': {name: {'records':len(intervals),
                               'clipped_wall_duration_ns':quantiles([b-a for a,b in intervals]),
                               'summed_clipped_wall_ns':sum(b-a for a,b in intervals),
                               'zero_kernel_gap_overlap_union_ns':overlap_ns(merge_intervals(intervals),gaps)}
                        for name,intervals in sorted(by_name.items())}}


def analyze(database, run, coverage_path):
    coverage = json.loads(coverage_path.read_text())
    run_info = json.loads((run/'run.json').read_text())
    if coverage['run_id'] != run.name or database.parent.name != run.name:
        raise ValueError('Trace, run and coverage identities differ')
    start, end = coverage['cuda_coverage_checks']['client_window_ns']
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro',uri=True) as db:
        kernels = [Kernel(*row) for row in db.execute(
            'SELECT k.start,k.end,k.globalPid,k.correlationId,k.graphNodeId,s.value '
            'FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id '
            'WHERE k.end>? AND k.start<? ORDER BY k.start', (start,end))]
        apis = db.execute('SELECT r.start,r.end,r.globalTid,r.correlationId,s.value '
                          'FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id').fetchall()
        process_pids = dict(db.execute('SELECT globalPid,pid FROM PROCESSES'))
    prefixes = ('cuLaunchKernel','cudaLaunchKernel','cuLaunchCooperativeKernel',
                'cudaLaunchCooperativeKernel','cuGraphLaunch','cudaGraphLaunch')
    launches = [Launch(a,b,tid & ~0xFFFFFF,cid,name) for a,b,tid,cid,name in apis
                if name.startswith(prefixes)]
    union = occupancy([(k.start,k.end,k.global_pid) for k in kernels],start,end)
    if union['zero_kernel_ns'] != coverage['occupancy']['zero_kernel_ns']:
        raise ValueError('Kernel union differs from the existing coverage analysis')
    gaps = union['zero_kernel_intervals_ns']
    sync_names = {'cudaDeviceSynchronize','cudaStreamSynchronize','cudaEventSynchronize',
                  'cuCtxSynchronize','cuStreamSynchronize','cuEventSynchronize'}
    sync_rows = [(a,b,name) for a,b,_,_,name in apis if name.split('_v',1)[0] in sync_names]
    workload_keys = {(k.global_pid,k.correlation) for k in kernels}
    launch_rows = [(a,b,name) for a,b,pid,cid,name in launches if (pid,cid) in workload_keys]
    return {'run_id':run.name,'run_status':run_info.get('status'),
            'controller_stop_exit':coverage.get('controller_stop_exit'),
            'trace_validity':coverage['validity'],'diagnostic_warnings':coverage['diagnostic_warnings'],
            'database':str(database),'coverage_analysis':str(coverage_path),
            'requests':coverage['requests'],'client_window_ns':[start,end],
            'window_ns':end-start,'kernel_union_ns':union['kernel_union_ns'],
            'kernel_records_in_window':len(kernels),'process_pid_by_global_pid':process_pids,
            'submission_gap_analysis':classify_gaps(gaps,kernels,launches),
            'synchronization_api_wall':api_overlap(sync_rows,gaps,start,end),
            'launch_api_wall':api_overlap(launch_rows,gaps,start,end),
            'limitations':[
                'Launch completion is observed host API return, not proof that GPU work is execution-eligible.',
                'Before a recorded launch completion does not mean no work was submitted; the API may still be running.',
                'Missing trace events can hide queued work; absence of a witness is not host-starvation proof.',
                'Future kernels and launch witnesses are restricted to the measured client window; its tail remains unknown.',
                'Synchronization API wall overlap is nonadditive and is not CPU compute time or causal attribution.',
                'These trace-only durations do not replace matched unprofiled performance measurements.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('database','run','coverage','output'):
        parser.add_argument(name,type=Path)
    args = parser.parse_args()
    result = analyze(args.database,args.run,args.coverage)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('run_id','run_status','trace_validity','window_ns','kernel_union_ns')},indent=2))
    print(json.dumps({key:value for key,value in result['submission_gap_analysis'].items()
                      if key != 'longest_gaps'},indent=2))
