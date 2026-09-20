"""Join a campaign's Nsight SQLite trace to its recorded request identities.

The metrics describe kernel presence, not SM utilization. Trace timings never
replace the unprofiled benchmark. Diagnostic warnings remain in the output.
(Verbatim copy of the round-1 analyzer; run.json/metrics layout is unchanged.)
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3


QWEN_TTS_ENDPOINTS = {
    'coordinator': ('request_admission', 'terminal_response'),
    'preprocessing': ('stage_input_received', 'stage_dispatch', 'stage_complete',
                      'stage_hop_sent'),
    'tts_engine': ('stage_input_received', 'stage_aggregate_ready', 'stage_dispatch',
                   'scheduler_request_build_start', 'scheduler_request_build_end',
                   'scheduler_queue_enter', 'model_path_start', 'scheduler_prefill_start',
                   'scheduler_prefill_end', 'model_path_end', 'stage_complete',
                   'stage_hop_sent'),
    'vocoder': ('stage_input_received', 'stage_aggregate_ready', 'stage_dispatch',
                'stage_complete'),
}


def event_coverage(events, nvtx_events, request_ids):
    """Check the expected Qwen3-TTS cohort, including events missing from both sinks."""
    def key(event):
        return (event['request_id'], event['stage'], event['event_name'],
                event.get('event_timestamp_ns', event.get('timestamp_ns')))
    selected = [event for event in events if event['request_id'] in request_ids]
    selected_nvtx = [event for event in nvtx_events if event['request_id'] in request_ids]
    json_keys = Counter(map(key, selected))
    nvtx_keys = Counter(map(key, selected_nvtx))
    endpoints = Counter((e['request_id'], e['stage'], e['event_name']) for e in selected)
    expected = {(rid, stage, name) for rid in request_ids
                for stage, names in QWEN_TTS_ENDPOINTS.items() for name in names}
    checks = {
        'missing_json_request_ids': sorted(request_ids - {e['request_id'] for e in selected}),
        'missing_nvtx_request_ids': sorted(request_ids - {e['request_id'] for e in selected_nvtx}),
        'missing_stage_endpoints': sorted(expected - endpoints.keys()),
        'repeated_stage_endpoints': sorted((key, count) for key, count in endpoints.items()
                                           if count != 1),
        'missing_nvtx_events': sorted((json_keys - nvtx_keys).elements()),
        'extra_nvtx_events': sorted((nvtx_keys - json_keys).elements()),
    }
    return {'complete': bool(request_ids) and not any(checks.values()),
            'expected_requests': len(request_ids), 'json_events': len(selected),
            'nvtx_events': len(selected_nvtx), **checks}


def sentinel_coverage(ranges, api_rows, kernels, replica_by_pid, process_pids,
                      stop_receipts, run_id, start, end, offset, external_stop=None):
    """Validate boundaries using NVTX-thread launch correlation, not temporal overlap alone.

    The hook tests its current stream. Other workload streams are reported as
    untested by these sentinels rather than being called fully covered.
    """
    grouped = defaultdict(list)
    for left, right, tid, payload in ranges:
        if payload.get('run_id') == run_id:
            grouped[(payload.get('pid'), payload.get('phase'))].append((left, right, tid))
    records = []
    for pid, replica in sorted(replica_by_pid.items()):
        phases = {}
        sentinel_streams = set()
        for phase in ('start', 'stop'):
            matches = grouped[(pid, phase)]
            record = {'range_count': len(matches), 'complete': False}
            if len(matches) == 1:
                left, right, tid = matches[0]
                actual_pid = process_pids.get(tid & ~0xFFFFFF)
                valid_range = right is not None and right > left
                keys = {(api_tid & ~0xFFFFFF, cid) for api_tid,cid,name,_,a,b in api_rows
                        if valid_range and api_tid == tid and 'Launch' in name and left <= a <= b <= right}
                linked = [k for k in kernels if valid_range and (k[7],k[8]) in keys
                          and k[2] == pid and left <= k[0] < k[1] <= right]
                streams = {k[4] for k in linked}
                sentinel_streams.update(streams)
                record.update({'range_ns': [left,right], 'nvtx_process_id': actual_pid,
                               'kernel_records': len(linked), 'stream_ids': sorted(streams),
                               'contexts': sorted({(k[9],k[3],k[10]) for k in linked}),
                               'complete': valid_range and actual_pid == pid and bool(linked)})
            phases[phase] = record
        complete_pair = all(record['complete'] for record in phases.values())
        surrounds_client = complete_pair and (
            phases['start']['range_ns'][1] <= start < end <= phases['stop']['range_ns'][0])
        acknowledgements = [receipt for receipt in stop_receipts
                            if receipt.get('pid') == pid and receipt.get('run_id') == run_id]
        ack = acknowledgements[0] if len(acknowledgements) == 1 else {}
        prepared_only = ack.get('cuda_profiler_stop_called') is False
        worker_ack_valid = (len(acknowledgements) == 1 and ack.get('synchronized') is True
                            and isinstance(ack.get('wall_ns'), int) and ack['wall_ns'] >= end+offset)
        external_valid = bool(external_stop and worker_ack_valid and
                              external_stop.get('run_id') == run_id and
                              external_stop.get('stop_exit') == 0 and
                              external_stop.get('nsys_stop_return') == 0 and
                              external_stop.get('report_closed') is True and
                              external_stop.get('workers_prepared') is True and
                              isinstance(external_stop.get('wall_ns'), int) and
                              external_stop['wall_ns'] >= ack['wall_ns'])
        ack_valid = worker_ack_valid and (external_valid if prepared_only else
                                          ack.get('profiler_stop_return') == 0)
        workload_streams = {k[4] for k in kernels if k[2] == pid and k[1] > start and k[0] < end}
        paired_streams = set(phases['start'].get('stream_ids', [])) & set(phases['stop'].get('stream_ids', []))
        records.append({'pid': pid, 'replica': replica, 'phases': phases,
                        'surrounds_client_window': surrounds_client,
                        'stop_ack_count': len(acknowledgements), 'stop_ack_valid': ack_valid,
                        'stop_ack_mode': 'external_controller' if prepared_only else 'worker_cuda_stop',
                        'worker_sync_ack_valid': worker_ack_valid,
                        'external_stop_valid': external_valid if prepared_only else None,
                        'sentinel_stream_ids': sorted(sentinel_streams),
                        'stream_ids_with_both_boundary_sentinels': sorted(paired_streams),
                        'workload_stream_ids_without_boundary_pair': sorted(workload_streams-paired_streams),
                        'complete': complete_pair and surrounds_client and ack_valid})
    unexpected = sorted((pid,phase) for pid,phase in grouped
                        if pid not in replica_by_pid or phase not in ('start', 'stop'))
    observed = bool(ranges or stop_receipts)
    complete = bool(records) and all(record['complete'] for record in records) and not unexpected
    return {'observed': observed, 'complete': complete,
            'status': 'not_recorded' if not observed else 'checks_passed' if complete else 'incomplete',
            'workers': records, 'unexpected_range_owners_or_phases': unexpected,
            'scope': 'worker boundaries and the sentinel streams only; no proof against interior buffer loss'}


def cuda_api_status(api_rows, start, end):
    counts = Counter((tid & ~0xFFFFFF,name,status) for tid,_,name,status,left,right in api_rows
                     if status and right > start and left < end)
    expected, errors = [], []
    for (pid,name,status), count in sorted(counts.items()):
        item = {'global_pid': pid, 'name': name, 'return_value': status, 'count': count}
        # cudaErrorNotReady=600 is documented query status, not failed GPU execution.
        # https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html
        is_query_status = status == 600 and name.split('_v', 1)[0] in ('cudaEventQuery', 'cudaStreamQuery')
        (expected if is_query_status else errors).append(item)
    return {'errors': errors, 'expected_not_ready_queries': expected}


def occupancy(intervals, start, end):
    """Integrate kernel union and overlap between distinct owners, clipped to a window."""
    changes = defaultdict(Counter)
    for left, right, owner in intervals:
        left, right = max(start, left), min(end, right)
        if right <= left:
            continue
        changes[left][owner] += 1
        changes[right][owner] -= 1
    active = Counter()
    previous = start
    busy = overlap = 0
    gaps = []
    for when in sorted(set(changes) | {end}):
        duration = when - previous
        owners = sum(count > 0 for count in active.values())
        if owners:
            busy += duration
        elif duration:
            gaps.append([previous, when])
        if owners > 1:
            overlap += duration
        active.update(changes[when])
        previous = when
    return {'window_ns': end-start, 'kernel_union_ns': busy,
            'zero_kernel_ns': end-start-busy, 'cross_owner_overlap_ns': overlap,
            'zero_kernel_intervals_ns': gaps}


def quantiles(values):
    values = sorted(values)
    if not values:
        return {'count': 0}
    def percentile(p):
        index = (len(values)-1)*p
        lo = int(index)
        return values[lo]+(values[min(lo+1,len(values)-1)]-values[lo])*(index-lo)
    return {'count': len(values), 'min': values[0], 'median': percentile(.5),
            'p95': percentile(.95), 'p99': percentile(.99), 'max': values[-1]}


def launch_correlation(api_rows, kernels, start=None, end=None):
    """Check one scope while resolving correlation against the complete recording.

    A cropped kernel can have been submitted before the window, and a launch near
    its end can execute afterwards. Startup graph-capture API calls remain in the
    full-capture audit rather than being mistaken for measured execution loss.
    """
    prefixes = ('cuLaunchKernel', 'cudaLaunchKernel', 'cuLaunchCooperativeKernel',
                'cudaLaunchCooperativeKernel', 'cuGraphLaunch', 'cudaGraphLaunch')
    launches = [row for row in api_rows if row[2].startswith(prefixes)]
    api_keys = {(tid & ~0xFFFFFF, cid) for tid,cid,*_ in launches}
    kernel_keys = {(k[7],k[8]) for k in kernels}
    selected_kernels = kernels if start is None else [
        k for k in kernels if k[1] > start and k[0] < end]
    selected_launches = launches if start is None else [
        row for row in launches if row[5] > start and row[4] < end]
    result = {
        'scope': 'full_capture' if start is None else 'measured_client_window',
        'kernels_without_api': sorted({(k[7],k[8]) for k in selected_kernels}-api_keys),
        'launches_without_kernel': sorted(
            (tid & ~0xFFFFFF,cid,name) for tid,cid,name,_,_,_ in selected_launches
            if (tid & ~0xFFFFFF,cid) not in kernel_keys),
        'kernel_records': len(selected_kernels),
        'launch_api_records': len(selected_launches),
    }
    result['complete'] = bool(selected_kernels) and not (
        result['kernels_without_api'] or result['launches_without_kernel'])
    return result


def analyze(database, run):
    run_info = json.loads((run/'run.json').read_text())
    rows = json.loads((run/'metrics/speed_results.json').read_text())['per_request']
    request_ids = {row['server_request_id'] for row in rows}
    if not rows or len(request_ids) != len(rows) or not all(row['is_success'] for row in rows):
        raise ValueError('Request IDs must be unique and all outputs accounted for')
    if any(not isinstance(row.get(name), int) for row in rows
           for name in ('request_start_ns', 'request_end_ns')) or any(
            row['request_end_ns'] <= row['request_start_ns'] for row in rows):
        raise ValueError('Every request needs valid start/end timestamps')
    events = [json.loads(line) for path in run.glob('profiles/events/*/*.jsonl')
              for line in path.read_text().splitlines()]
    with sqlite3.connect(database.resolve().as_uri()+'?mode=ro', uri=True) as db:
        request_nvtx = [(ts,tid,json.loads(text.split('|',1)[1])) for ts,tid,text in db.execute(
            "SELECT start,globalTid,text FROM NVTX_EVENTS WHERE text LIKE 'omni_event|%'")]
        nvtx = [(ts,event) for ts,_,event in request_nvtx]
        sentinel_ranges = [(left,right,tid,json.loads(text.split('|',1)[1]))
                           for left,right,tid,text in db.execute(
                               "SELECT start,end,globalTid,text FROM NVTX_EVENTS "
                               "WHERE text LIKE 'campaign_sentinel|%'")]
        process_pids = dict(db.execute('SELECT globalPid,pid FROM PROCESSES'))
        if not nvtx:
            raise ValueError('No request NVTX clock anchors')
        offsets = sorted(event['anchor_wall_ns']-ts for ts,event in nvtx)
        offset = offsets[len(offsets)//2]
        start = min(row['request_start_ns'] for row in rows)-offset
        end = max(row['request_end_ns'] for row in rows)-offset
        kernels = db.execute('SELECT k.start,k.end,p.pid,k.contextId,k.streamId,s.value,k.graphNodeId, '
                             'k.globalPid,k.correlationId,k.deviceId,k.greenContextId '
                             'FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                             'LEFT JOIN PROCESSES p ON k.globalPid=p.globalPid '
                             'LEFT JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start').fetchall()
        if any(k[2] is None for k in kernels):
            raise ValueError('Kernel process identity is unresolved')
        receipts = [(path.parent.name, json.loads(path.read_text())) for path in
                    run.glob('profiles/events/*/cuda-started-*.json')]
        replica_by_pid = {receipt['pid']: replica for replica, receipt in receipts
                          if receipt.get('run_id') == run.name}
        expected_replicas = {f'replica-{index}' for index in range(len(run_info['cpu_sets']))}
        window_kernels = [kernel for kernel in kernels if kernel[1] > start and kernel[0] < end]
        kernel_pids = {kernel[2] for kernel in window_kernels}
        worker_coverage = {
            'expected_replicas': sorted(expected_replicas),
            'started_replicas': sorted(set(replica_by_pid.values())),
            'replica_by_pid': replica_by_pid,
            'missing_start_receipt_replicas': sorted(expected_replicas-set(replica_by_pid.values())),
            'unexpected_start_receipt_replicas': sorted(set(replica_by_pid.values())-expected_replicas),
            'missing_kernel_pids': sorted(set(replica_by_pid)-kernel_pids),
            'unexpected_kernel_pids': sorted(kernel_pids-set(replica_by_pid)),
            'kernel_records_in_client_window': len(window_kernels),
        }
        worker_coverage['complete'] = bool(expected_replicas) and bool(replica_by_pid) and not any(
            worker_coverage[key] for key in ('missing_start_receipt_replicas',
                                           'unexpected_start_receipt_replicas',
                                           'missing_kernel_pids', 'unexpected_kernel_pids'))
        metrics = occupancy([(k[0],k[1],replica_by_pid.get(k[2],f'pid:{k[2]}'))
                             for k in kernels],start,end)
        metrics['owner_kind'] = 'replica' if worker_coverage['complete'] else 'unverified_process'
        metrics['cross_replica_overlap_ns'] = (metrics['cross_owner_overlap_ns']
                                               if worker_coverage['complete'] else None)
        warnings = [dict(zip(('timestamp','timestamp_type','source','severity','text','global_pid'),row))
                    for row in db.execute('SELECT timestamp,timestampType,source,severity,text,globalPid '
                                          'FROM DIAGNOSTIC_EVENT WHERE severity>=2')]
        capture = db.execute("SELECT min(timestamp),max(timestamp) FROM DIAGNOSTIC_EVENT "
                             "WHERE timestampType=1 AND source=3 AND text IN "
                             "('Profiling has started.','Profiling has stopped.')").fetchone()
        capture_covers_window = (all(value is not None for value in capture)
                                 and capture[0] <= start < end <= capture[1])
        api_rows = db.execute('SELECT r.globalTid,r.correlationId,s.value,r.returnValue,r.start,r.end '
                              'FROM CUPTI_ACTIVITY_KIND_RUNTIME r '
                              'JOIN StringIds s ON r.nameId=s.id').fetchall()
        stop_receipts = [json.loads(path.read_text()) for pattern in
                         ('profiles/events/*/cuda-stopped-*.json',
                          'profiles/events/*/cuda-prepared-stop-*.json')
                         for path in run.glob(pattern)]
        external_stop = run_info.get('nsys_external_stop')
        capture_boundaries = sentinel_coverage(sentinel_ranges, api_rows, kernels, replica_by_pid,
                                               process_pids, stop_receipts, run.name, start, end, offset,
                                               external_stop=external_stop)
        full_correlation = launch_correlation(api_rows, kernels)
        correlation = launch_correlation(api_rows, kernels, start, end)
        api_status = cuda_api_status(api_rows, start, end)
        cuda_errors = api_status['errors']
        invalid_kernels = sum(kernel[1] <= kernel[0] for kernel in kernels
                              if start <= kernel[0] < end or kernel[1] > start and kernel[0] < end)
        by_kernel = defaultdict(list)
        for left,right,pid,ctx,stream,name,graph_node,*_ in kernels:
            if right>start and left<end:
                by_kernel[(pid,name)].append(right-left)
        top = sorted(by_kernel,key=lambda key:sum(by_kernel[key]),reverse=True)[:20]
        top_kernels = [{'pid':key[0],'name':key[1],'duration_ns':quantiles(by_kernel[key]),
                        'summed_duration_ns':sum(by_kernel[key])} for key in top]
        contexts = [dict(zip([x[1] for x in db.execute('PRAGMA table_info(TARGET_INFO_CUDA_CONTEXT_INFO)')],row))
                    for row in db.execute('SELECT * FROM TARGET_INFO_CUDA_CONTEXT_INFO')]
        context_keys = {(c['processId'],c['deviceId'],c['contextId']) for c in contexts}
        missing_contexts = sorted({(k[2],k[9],k[3]) for k in window_kernels}-context_keys)
        gpu_rows = db.execute('SELECT id,uuid FROM TARGET_INFO_GPU').fetchall()
        expected_uuid = run_info['gpu_uuid'].removeprefix('GPU-').lower()
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if 'TARGET_INFO_CUDA_DEVICE' in tables:
            # Kernel deviceId is the process's CUDA ordinal (0 under a UUID-valued CUDA_VISIBLE_DEVICES),
            # so resolve it per process through the CUDA-device table rather than the node-wide GPU index.
            device_rows = db.execute('SELECT pid,cudaId,uuid FROM TARGET_INFO_CUDA_DEVICE').fetchall()
            allowed = {(pid, cuda_id) for pid, cuda_id, uuid in device_rows
                       if uuid.removeprefix('GPU-').lower() == expected_uuid}
            expected_devices = {cuda_id for _, cuda_id in allowed}
            device_coverage = bool(window_kernels) and bool(allowed) and all(
                (k[2], k[9]) in allowed for k in window_kernels)
            full_device_coverage = bool(kernels) and bool(allowed) and all((k[2], k[9]) in allowed for k in kernels)
            device_mapping = 'TARGET_INFO_CUDA_DEVICE per-process cudaId->uuid'
        else:
            expected_devices = {device for device,uuid in gpu_rows
                                if uuid.removeprefix('GPU-').lower() == expected_uuid}
            device_coverage = bool(window_kernels) and bool(expected_devices) and (
                {k[9] for k in window_kernels} <= expected_devices)
            full_device_coverage = bool(kernels) and bool(expected_devices) and {k[9] for k in kernels} <= expected_devices
            device_mapping = 'TARGET_INFO_GPU node index (legacy; only valid when CUDA ordinal == node index)'
        full_cuda_coverage = {
            'scope': 'full_capture_including_startup_and_warmup',
            'kernel_records': len(kernels),
            'graph_node_kernel_records': sum(bool(k[6]) for k in kernels),
            'launch_correlation': full_correlation,
            'expected_device_only': full_device_coverage,
            'device_mapping': device_mapping,
            'missing_context_mappings': sorted({(k[2],k[9],k[3]) for k in kernels}-context_keys),
            'invalid_kernel_intervals': sum(k[1] <= k[0] for k in kernels),
            'cuda_api_errors': cuda_api_status(api_rows, float('-inf'), float('inf'))['errors'],
        }
        full_cuda_coverage['anomalies_require_review'] = (
            not full_correlation['complete'] or not full_cuda_coverage['expected_device_only']
            or bool(full_cuda_coverage['missing_context_mappings'])
            or bool(full_cuda_coverage['invalid_kernel_intervals'])
            or bool(full_cuda_coverage['cuda_api_errors']))
        ownership_counts = Counter((k[2],k[7],k[9],k[3],k[10],k[4]) for k in window_kernels)
        graph_counts = Counter((k[2],k[7],k[9],k[3],k[10],k[4]) for k in window_kernels if k[6])
        ownership = [dict(zip(('pid','global_pid','device_id','context_id','green_context_id',
                               'stream_id'), key), replica=replica_by_pid.get(key[0]),
                          kernel_records=count, graph_node_kernel_records=graph_counts[key])
                     for key,count in ownership_counts.items()]
        nvtx_counts = Counter(process_pids.get(tid & ~0xFFFFFF) for _,tid,event in request_nvtx
                              if event['request_id'] in request_ids)
        processes = []
        for pid in sorted(set(replica_by_pid) | {pid for pid in nvtx_counts if pid is not None}):
            owned = [k for k in window_kernels if k[2] == pid]
            processes.append({'pid': pid, 'replica': replica_by_pid.get(pid),
                              'expected_gpu_worker': pid in replica_by_pid,
                              'request_nvtx_marks': nvtx_counts[pid],
                              'kernel_records_in_client_window': len(owned),
                              'graph_node_kernel_records_in_client_window': sum(bool(k[6]) for k in owned),
                              'kernel_stream_ids': sorted({k[4] for k in owned}),
                              'cuda_api_records_in_client_window': sum(
                                  process_pids.get(tid & ~0xFFFFFF) == pid and b > start and a < end
                                  for tid,_,_,_,a,b in api_rows)})
        graph_requested = run_info['config']['stages']['tts_engine']['engine'].get('disable_cuda_graph') is False
        graph_evidence = {'ar_graph_requested': graph_requested,
                          'kernel_records_in_client_window': sum(bool(k[6]) for k in window_kernels),
                          'expected_worker_pids_without_graph_nodes': sorted(
                              set(replica_by_pid)-{k[2] for k in window_kernels if k[6]}),
                          'scope': 'graph node kernels on a worker do not identify the AR/predictor/codec stage; join runtime counters'}
        graph_evidence['requested_graph_nodes_observed'] = (not graph_requested or not
                                                           graph_evidence['expected_worker_pids_without_graph_nodes'])
    request_coverage = event_coverage(events,[event for _,event in nvtx],request_ids)
    cuda_coverage = {
        'scope': 'measured_client_window',
        'complete': (worker_coverage['complete'] and correlation['complete'] and device_coverage
                     and not missing_contexts and not cuda_errors and not invalid_kernels
                     and capture_covers_window),
        'worker_coverage': worker_coverage, 'launch_correlation': correlation,
        'process_coverage': processes, 'boundary_sentinels': capture_boundaries,
        'graph_node_evidence': graph_evidence,
        'kernel_ownership_in_client_window': ownership,
        'expected_device_only': device_coverage, 'device_mapping': device_mapping, 'missing_context_mappings': missing_contexts,
        'cuda_api_errors_in_window': cuda_errors, 'capture_covers_client_window': capture_covers_window,
        'expected_not_ready_queries_in_window': api_status['expected_not_ready_queries'],
        'invalid_kernel_intervals': invalid_kernels,
        'capture_bounds_ns': list(capture), 'client_window_ns': [start,end],
    }
    grouped = defaultdict(dict)
    for event in events:
        grouped[(event['request_id'],event['stage'])][event['event_name']] = event
    boundaries = [('stage_input_received','stage_dispatch'),('stage_dispatch','stage_complete'),
                  ('scheduler_request_build_start','scheduler_request_build_end'),
                  ('scheduler_queue_enter','model_path_start'),('model_path_start','model_path_end'),
                  ('request_admission','terminal_response')]
    stages = defaultdict(list)
    for (request_id,stage),names in grouped.items():
        if request_id not in request_ids:
            continue
        for first,last in boundaries:
            if first not in names or last not in names:
                continue
            a,b = names[first],names[last]
            item = {'request_id':request_id,'wall_ns':b['timestamp_ns']-a['timestamp_ns']}
            if (a['pid'],a['thread_id']) == (b['pid'],b['thread_id']):
                item['same_thread_cpu_ns'] = b['thread_cpu_ns']-a['thread_cpu_ns']
            stages[f'{stage}:{first}->{last}'].append(item)
    return {'run_id':run.name,'run_status':run_info.get('status'),
            'controller_stop_exit': (external_stop.get('stop_exit') if external_stop else
                                     run_info.get('nsys_stop_exit')),
            'external_controller_stop': external_stop,
            'database':str(database),'requests':len(rows),
            'event_requests':len({e['request_id'] for e in events}&request_ids),
            'nvtx_requests':len({e['request_id'] for _,e in nvtx}&request_ids),
            'missing_request_nvtx_events':request_coverage['missing_nvtx_events'],
            'request_coverage':request_coverage,'cuda_coverage_checks':cuda_coverage,
            'full_capture_cuda_checks':full_cuda_coverage,
            'clock_offset_ns':offset,'clock_anchor_range_ns':max(offsets)-min(offsets),
            'kernel_process_ids':sorted({k[2] for k in kernels}),
            'kernel_count':len(kernels),'graph_node_kernel_count':sum(bool(k[6]) for k in kernels),
            'kernel_count_scope':'full_capture',
            'kernel_count_in_client_window':len(window_kernels),
            'graph_node_kernel_count_in_client_window':sum(bool(k[6]) for k in window_kernels),
            'occupancy':metrics,'gap_duration_ns':quantiles([b-a for a,b in metrics['zero_kernel_intervals_ns']]),
            'top_kernels':top_kernels,'cuda_contexts':contexts,'stage_intervals':dict(stages),
            'diagnostic_warnings':warnings,
            'validity': ('request_coverage_incomplete' if not request_coverage['complete'] else
                         'cuda_coverage_incomplete' if not cuda_coverage['complete'] else
                         'capture_boundary_checks_incomplete' if capture_boundaries['observed'] and not capture_boundaries['complete'] else
                         'graph_node_evidence_missing' if not graph_evidence['requested_graph_nodes_observed'] else
                         'capture_review_required' if warnings or not capture_boundaries['complete']
                         or full_cuda_coverage['anomalies_require_review'] else 'coverage_checks_passed'),
            'limitations':['Kernel union is not SM utilization.',
                           'Same-thread CPU time includes other work interleaved on that thread.',
                           'Stage intervals overlap; their sums are not a latency decomposition.',
                           'Profiler diagnostics require review before causal interpretation.',
                           'Paired API/activity records do not exclude loss of both from one buffer.',
                           'No SM activity counters or profiler perturbation bound are established.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('database',type=Path)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    result = analyze(args.database,args.run)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('run_id','requests','nvtx_requests','missing_request_nvtx_events','clock_anchor_range_ns','kernel_count','validity')},indent=2))
    print({k:v for k,v in result['occupancy'].items() if not isinstance(v,list)})
