#!/usr/bin/env python3
"""Snapshot existing aviation logs and prepare the 2026-11-06 meeting handout.

Standard library only. Never starts or changes a solve. Each invocation keeps a
new dated directory. Stop for editorial review if solutions or terminal results
appear, rather than reuse the present unsolved-run interpretation.
"""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
RUN = HERE / 'runs/aviation-k4-seed0'
ZONE = dt.timezone(dt.timedelta(hours=8))
LABELS = {'sssp': 'SSSP＋原生 SCIP', 'mamba': 'Mamba CPU', 'tgppo': '原 TGPPO B0 CPU'}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def duration(seconds):
    minutes = int(seconds // 60)
    return f'{minutes//60}小时{minutes%60:02d}分'


def beijing(value):
    return dt.datetime.fromisoformat(value).astimezone(ZONE).strftime('%m-%d %H:%M:%S')


def main():
    started = dt.datetime.now(dt.timezone.utc)
    out = HERE / 'reports' / ('meeting_' + started.astimezone(ZONE).strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    evidence, attempts = [], []

    def capture(path, relative):
        # Concurrent writers replace JSON atomically; also reject a log that
        # changes while it is being read so saved bytes and metadata agree.
        for _ in range(4):
            before = path.stat()
            data = path.read_bytes()
            after = path.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino):
                break
        else:
            raise RuntimeError(f'File kept changing; retry snapshot: {path}')
        target = out / 'evidence' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        entry = dict(source=str(path.relative_to(ROOT)), snapshot=str(target.relative_to(out)),
                     bytes=len(data), sha256=checksum(data),
                     modified_utc=dt.datetime.fromtimestamp(after.st_mtime, dt.timezone.utc).isoformat())
        evidence.append(entry)
        return data, entry

    for method in LABELS:
        method_attempts = sorted((RUN / f'{method}-s0').glob('attempt-*'))
        for attempt in method_attempts:
            item = dict(method=method, attempt=str(attempt.relative_to(ROOT)),
                        latest_attempt=attempt == method_attempts[-1], files={})
            for name in ('launch.json', 'result.json', 'progress.json', 'progress.jsonl', 'scip.log', 'settings.set', 'incumbent.sol'):
                path = attempt / name
                if not path.exists():
                    continue
                data, entry = capture(path, Path(method) / attempt.name / name)
                item['files'][name] = entry
                if name.endswith('.json'):
                    item[name[:-5]] = json.loads(data)
                elif name == 'scip.log':
                    lines = data.decode(errors='replace').splitlines()
                    rows, restarts, converted = [], [], []
                    for number, line in enumerate(lines, 1):
                        match = re.match(r'\s*([\d.]+)([smhd])\s*\|', line)
                        if match:
                            cells = [s.strip() for s in line.split('|')]
                            rows.append(dict(line=number, text=line, time_label=match[0].strip(' |'),
                                             display_seconds=float(match[1])*dict(s=1, m=60, h=3600, d=86400)[match[2]],
                                             nodes=int(cells[1]), nodes_left=int(cells[2]),
                                             lp_iterations=cells[3], solver_memory=cells[5], max_depth=int(cells[6]),
                                             dual_bound=cells[14], primal_bound=cells[15], gap=cells[16]))
                        match = re.search(r'\(run (\d+), node (\d+)\) performing user restart', line)
                        if match:
                            restarts.append(dict(line=number, run=int(match[1]), first_run_nodes=int(match[2])))
                        match = re.search(r'\(restart\) converted (\d+) cuts', line)
                        if match:
                            converted.append(int(match[1]))
                    item['log_rows'] = rows
                    item['internal_restarts'] = restarts
                    item['restart_converted_cuts'] = converted
                    item['initial_root_departure'] = next((r for r in rows if r['nodes'] == 1 and r['nodes_left'] > 0), None)
            attempts.append(item)

    manifest_path = next((HERE / 'artifacts').glob('model-*/manifest.json'))
    manifest = json.loads(capture(manifest_path, Path('model-manifest.json'))[0])
    capture(HERE / 'checks/mamba-cpu/validation.json', Path('mamba-cpu-validation.json'))
    for name in ('compare.py', 'mamba_cpu.py', 'README.md'):
        capture(HERE / name, Path('driver') / name)
    capture(Path(__file__), Path('prepare_meeting_report.py'))

    current = [a for a in attempts if a['latest_attempt']]
    identities = [{key: a['result'][key] for key in
                   ('model_sha256', 'binding_sha256', 'libscip_sha256', 'parameters', 'scip_version')}
                  for a in current]
    assert all(value == identities[0] for value in identities), 'Model/solver identities differ'
    assert identities[0]['model_sha256'] == manifest['files']['model.cip']
    assert all(a['result']['status'] == 'running' and a['progress']['solutions'] == 0 for a in current), \
        'New solution or terminal status: update the meeting interpretation before publishing'
    assert all(not (out / a['files'].get('incumbent.sol', {}).get('snapshot', 'missing')).is_file()
               for a in current), 'Incumbent appeared: review snapshot'
    assert all(a['log_rows'][-1]['primal_bound'] == '--' and a['log_rows'][-1]['gap'] == 'Inf' for a in current)
    assert all(a['internal_restarts'] and a['log_rows'][-1]['nodes'] == 1 for a in current)
    assert max(a['progress']['dual_bound_real'] for a in current) - min(a['progress']['dual_bound_real'] for a in current) < 1e-8
    # Verify the actual model and frozen weights without copying large inputs.
    external_hashes = {}
    paths = [(manifest_path.parent / 'model.cip', manifest['files']['model.cip'])]
    paths += [(Path(a['result']['policy']['checkpoint']), a['result']['policy']['checkpoint_sha256'])
              for a in current if a['method'] != 'sssp']
    for path, expected in paths:
        actual = checksum(path.read_bytes())
        assert actual == expected, path
        external_hashes[str(path.relative_to(ROOT))] = actual

    ended = dt.datetime.now(dt.timezone.utc)
    rows = []
    for a in current:
        last = a['log_rows'][-1]
        rows.append(dict(method=a['method'], label=LABELS[a['method']],
                         device=a['result'].get('policy', {}).get('device', 'cpu'),
                         cpu_model=a['result']['hardware']['cpu_model'],
                         latest_log_time_label=last['time_label'],
                         latest_log_approx_solve_seconds=last['display_seconds'],
                         latest_log_modified_beijing=beijing(a['files']['scip.log']['modified_utc']),
                         latest_log_age_minutes=round((ended-dt.datetime.fromisoformat(a['files']['scip.log']['modified_utc'])).total_seconds()/60, 1),
                         first_run_nodes=a['internal_restarts'][0]['first_run_nodes'],
                         restart_converted_cuts=a['restart_converted_cuts'][0],
                         latest_log_run_nodes=last['nodes'], latest_log_run_nodes_left=last['nodes_left'],
                         last_progress_nodes=a['progress']['nodes'], last_progress_nodes_left=a['progress']['nodes_left'],
                         last_progress_solve_seconds=a['progress']['scip_seconds'],
                         last_progress_observed_beijing=beijing(a['progress']['observed_utc']),
                         solutions=0, upper_bound_real=None, lower_bound_real=a['progress']['dual_bound_real'],
                         gap='infinite', optimality_proven=False,
                         persisted_status=a['result']['status'], platform_liveness_verified=False,
                         initial_root_departure_seconds=a['initial_root_departure']['display_seconds'],
                         evidence_dir=str(Path(a['files']['scip.log']['snapshot']).parent)))
    snapshot = dict(capture_started_utc=started.isoformat(), capture_finished_utc=ended.isoformat(),
                    capture_beijing=ended.astimezone(ZONE).isoformat(),
                    model_manifest=manifest, common_model_and_solver_verified=True,
                    external_input_hashes_verified=external_hashes, summary_rows=rows,
                    attempts=attempts, evidence=evidence, original_runs_modified=False,
                    new_solves_or_training_started=False)
    write_json(out / 'snapshot.json', snapshot)
    with (out / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    at = ended.astimezone(ZONE).strftime('%Y年%m月%d日 %H:%M:%S')
    table = '\n'.join(f"| {r['label']} | 约{duration(r['latest_log_approx_solve_seconds'])} | {r['first_run_nodes']:,} | 0 | 30.762624 | 无穷 | {r['latest_log_modified_beijing']} |" for r in rows)
    result_table = ('| 方法 | 最后日志累计求解时长 | 首轮重启节点 | 可行解数 | 下界（Real） | gap | 日志更新时间（北京） |\n'
                    '|---|---:|---:|---:|---:|---|---|\n' + table)
    restarts = '、'.join(f"{LABELS[a['method']]}：{a['internal_restarts'][0]['first_run_nodes']}节点／保留{a['restart_converted_cuts'][0]}条割" for a in current)
    root_times = '、'.join(f"{LABELS[a['method']]}约{a['initial_root_departure']['display_seconds']/60:.1f}分钟" for a in current)
    hardware_table = '\n'.join(f"| {r['label']} | {r['cpu_model']} | CPU |" for r in rows)
    source_table = '\n'.join(f"| {r['label']} | [{r['evidence_dir']}/scip.log]({r['evidence_dir']}/scip.log) | [{r['evidence_dir']}/progress.json]({r['evidence_dir']}/progress.json) |" for r in rows)
    mamba_age = next(r['latest_log_age_minutes'] for r in rows if r['method'] == 'mamba')

    report = f'''**航空布线最大实例：SSSP、Mamba、TGPPO 求解阶段汇报**

数据截点：北京时间 {at}。这是进行中实验的落盘快照，原作业和日志保持原样；没有为本次整理启动求解或训练。快照保存在本目录，后续日志变化不会改写这里的证据。

本轮把航空布线源码的同一个 MILP 分别交给原生 SCIP、冻结 Mamba 分支器和冻结原 TGPPO B0 分支器求解。截至快照，三组均未找到可行整数解，也未给出最优性证明；三组全局下界都停留在约 30.762624，尚无可确认的学习分支优势。这说明目前配置仍未解决该实例的首个可行解与搜索推进问题，不能据此认定实例无解，也不能据节点数或运行时长给方法排名。

实例为 `edges.csv`＋`pairs-4-246.csv`，k=4：原图546节点、641边、246点对；导出258,164变量（5,736二进制、252,428连续）和162,800约束。初次预处理后为257,710变量、5,617二进制变量、162,505约束。下文目标沿用原程序的 Real 口径（模型目标除以1e6），不另赋物理单位。

三个方法共享原航空建模与字节一致的 CIP。SSSP 组指 `sssp_0923.cpp` 的原图化简／建模加 SCIP 原生分支；Mamba 组在同一模型上使用原已选 CL44／IL50 权重、24步历史；TGPPO 组使用原 B0、训练种子11、开发集选定 episode100 Actor。两组模型均冻结，没有用航空结果重新选模或训练；这里不包含最新 MLP＋BC 或持续专家监督改进模型。

已核对三组同一 SCIP 8.0.4、定制 PySCIPOpt 4.3.0、绑定和 libscip SHA、模型 SHA 与已记录参数一致：单线程、solver seed=0、容差1e-6、无已知最优 cutoff、启用原生预处理／割／启发式；求解时间设 SCIP 无限值1e20，gap／绝对gap阈值为0，节点等限制为-1。平台作业时限仍独立存在。完整模型与两份冻结权重本次重新核对哈希通过。

{result_table}

表中时长来自各自最后一条 SCIP 文本日志的累计时间，分钟显示有舍入，应读作“已观察到约该时长”，不是完成用时，也不是把启动到现在的日历时长当成有效计算时长。三组日志都已进入内部重启后的根节点，当前根节点行的 node=1、left=0 不能解释为已完成或总共只搜索1节点。

本轮新观察是三组均触发内部重启：{restarts}。日志给出的触发原因为连续50次估计剩余搜索树会很大。重启后保留全局割并再次处理根节点；它属于同一 SCIP 进程的搜索操作，不是平台回收后的作业恢复。重启尚未带来可观测的全局下界改善。

SSSP 和 TGPPO 的文本日志在截点前数分钟内仍有新记录。Mamba 最新文本日志距截点约{mamba_age:.0f}分钟，最后在重启后的根节点进行割分离；快照文件的最后完成节点更早。尚无平台进程／CPU使用率证据，不能由 `running` 字段确认其存活，也不能仅凭日志停更宣布它已被回收。三组均没有 `incumbent.sol`。没有可行解意味着当前上界缺失、gap无穷；30.762624只是下界，不能当成已经获得的布线方案成本。

初次根节点进入分支阶段的日志时间为：{root_times}。这包含该阶段内的LP、割、启发式和分支初始化等开销；不同CPU影响计时，不能把这些时间差解释成神经推理带来的加速。

| 当前组 | 实际CPU型号 | 推理设备 |
|---|---|---|
{hardware_table}

SSSP 与 TGPPO 当前为同CPU型号，但不是同一节点、同一时长、同步完成的受控比较；Mamba 的CPU型号不同。加上单实例／单solver种子、两模型训练任务与航空任务不同、TGPPO由SCIP6／Torch1.13迁到SCIP8／Torch2.2、Mamba由CUDA迁到CPU，这些都是性能解释的边界。节点处理多不等于更接近最优；各自实际时长上的PDI也不能直接作为共同预算比较，本次不报告加速比。

旧GPU尝试单列保留，不和新CPU搜索拼接：Mamba旧GPU作业被回收是用户提供的平台信息，最后落盘进度仅能确认1小时16分、60节点、0解，不能把它当成整个作业的最终耗时；TGPPO旧GPU尝试实际记录SIGTERM15中断，累计3小时25分43秒、280节点、0解，平台终止原因需另查。两组后来均从头在CPU重投，没有恢复旧搜索树。

CPU部署解决了GPU长时间空闲这一资源不匹配问题。Mamba CPU适配保留真实Mamba-1结构与冻结权重，通过官方CPU参考核对及29次实际小模型分支，测试未初始化CUDA；CPU／CUDA近并列候选动作不保证相同。这是部署验证成果，不是航空求解性能成果。旧TGPPO GPU尝试中Actor累计仅3.05秒，占其12,343秒求解时长约0.025%，也说明该次观测中长期占用GPU收益有限；此比例不外推到Mamba CPU作业。

今晚可以直接这样汇报：

> 我们已将最大航空布线实例接入同一个SCIP求解框架，比较原生分支、复现的Mamba分支和原TGPPO B0分支。模型有25.8万变量、16.3万约束，三组均按无限时配置运行。到今天这次快照，原生已观察到约19.7小时、Mamba约11.3小时、TGPPO约15.6小时，但都还没有找到可行整数解，下界都停在30.762624。三组在约一千节点时均触发内部搜索重启。因此目前的成果是完成同模型迁移和CPU部署，求解质量上尚未观察到学习分支优势，也没有证据认为即将完成。下一步优先核实Mamba作业存活，针对首个可行解和搜索树增长做诊断，再决定是否调整建模、构造热启动或分支策略。

后续建议先核实平台存活、截止时限与CPU使用率，组会前再截一次最新结果。随后将“构造并验证首个可行解／热启动”和“定位LP／启发式／分支搜索耗时及特征迁移”作为候选诊断方向，另行设计同机同预算对照。本次没有修改现行求解参数、停止／重提作业或启动这些后续实验。

本次可复核证据如下。文本日志与节点快照分开保存，避免重启期间误用旧快照。

| 方法 | 文本日志 | 节点快照 |
|---|---|---|
{source_table}

机器表见 [comparison.csv](comparison.csv)，完整证据索引／SHA／全部5次尝试见 [snapshot.json](snapshot.json)，一页汇报见 [ONE_SLIDE.md](ONE_SLIDE.md)。原始模型SHA：`{manifest['files']['model.cip']}`。
'''
    (out / 'MEETING_REPORT.md').write_text(report)
    slide = f'''**最大航空布线实例：三种分支策略仍未找到可行解**

快照：北京时间 {at}；阶段结果，非最终完成结果。

同一MILP：k=4，641边／246点对，258,164变量、162,800约束；统一SCIP8.0.4、单线程、seed0、无求解时限、启发式开启。

{result_table}

- 三组上界均缺失，下界从根节点起未改善；没有可行布线方案或最优性证明。
- 三组均在约千节点处内部重启，现为新根节点阶段；Mamba日志停更约{mamba_age:.0f}分钟，平台存活待核实。
- 已完成同模型迁移与CPU部署；尚无学习分支性能优势证据。CPU异构、预算不齐，不按节点数排名或宣称加速。

下一步建议：核实作业状态；诊断首个可行解／LP与搜索耗时，评估构造热启动及模型改进。后续实验尚未启动。

时长为各自最后日志累计值，非完成用时；证据和口播稿见 [MEETING_REPORT.md](MEETING_REPORT.md)。
'''
    (out / 'ONE_SLIDE.md').write_text(slide)
    # Audit only the saved snapshot; do not re-read changing source logs.
    for entry in evidence:
        assert checksum((out / entry['snapshot']).read_bytes()) == entry['sha256']
    assert len(rows) == 3 and all(r['solutions'] == 0 and not r['optimality_proven'] for r in rows)
    print(json.dumps(dict(output=str(out), capture_beijing=snapshot['capture_beijing'],
                          evidence_files=len(evidence), attempts=len(attempts), rows=rows),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
