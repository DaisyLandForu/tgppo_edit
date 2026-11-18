#!/usr/bin/env python3
"""Build the aviation supplement from existing logs; never import a solver.

Uses python-pptx (the existing meeting dependencies can be supplied via
PYTHONPATH=/tmp/tgppo_meeting_report_deps). Captures a new immutable evidence
directory and updates only the aviation supplement, leaving the main deck intact.
"""
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'report'
AIR = ROOT / 'scip/airwiring_compare'
RUN = AIR / 'runs/aviation-k4-seed0'
DIAG = AIR / 'reports/diagnosis_20261106_153639'
ZONE = dt.timezone(dt.timedelta(hours=8))
NAMES = {'sssp': 'SSSP＋原生 SCIP', 'mamba': 'Mamba CPU', 'tgppo': '原 TGPPO B0 CPU'}
STEM = '航空布线最大实例_三方法求解汇报_20261106'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def duration(seconds):
    minutes = int(seconds // 60)
    return f'{minutes//60}小时{minutes%60:02d}分'


def main():
    started = dt.datetime.now(dt.timezone.utc)
    data_dir = OUT / 'data' / ('airwiring_' + started.astimezone(ZONE).strftime('%Y%m%d_%H%M%S'))
    data_dir.mkdir(parents=True)
    preserved_paths = [OUT/'TGPPO_组会汇报_20261106.md', OUT/'TGPPO_组会幻灯片_20261106.pptx',
                       OUT/'TGPPO_组会幻灯片_20261106.html']
    preserved = {str(p.relative_to(ROOT)): sha(p.read_bytes()) for p in preserved_paths}
    evidence, attempts, rows = [], [], []

    def capture(source, name):
        for _ in range(4):
            a = source.stat()
            data = source.read_bytes()
            b = source.stat()
            if (a.st_size, a.st_mtime_ns, a.st_ino) == (b.st_size, b.st_mtime_ns, b.st_ino):
                break
        else:
            raise RuntimeError('Log changed during read; retry: ' + str(source))
        target = data_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        entry = dict(source=str(source.relative_to(ROOT)), saved=str(target.relative_to(OUT)), sha256=sha(data),
                     modified_beijing=dt.datetime.fromtimestamp(b.st_mtime, ZONE).isoformat())
        evidence.append(entry)
        return data, entry

    for method, label in NAMES.items():
        paths = sorted((RUN/f'{method}-s0').glob('attempt-*'))
        for attempt in paths:
            obj = dict(method=method, attempt=str(attempt.relative_to(ROOT)), latest=attempt==paths[-1])
            for name in ('launch.json', 'result.json', 'progress.json', 'progress.jsonl', 'scip.log'):
                p = attempt/name
                if p.exists():
                    data, entry = capture(p, Path(method)/attempt.name/name)
                    if name.endswith('.json'):
                        obj[name[:-5]] = json.loads(data)
                    elif name == 'scip.log':
                        log = data.decode(errors='replace')
                        log_rows = [line for line in log.splitlines() if re.match(r'\s*[\d.]+[smhd]\s*\|', line)]
                        obj['last_log_row'] = log_rows[-1]
                        obj['restarts'] = [dict(run=int(run), node=int(node)) for run,node in
                                           re.findall(r'\(run (\d+), node (\d+)\) performing user restart', log)]
                        obj['log_modified_beijing'] = entry['modified_beijing']
            if not obj['latest']:
                attempts.append(obj)
                continue
            r, p = obj['result'], obj['progress']
            assert r['status']=='running' and p['solutions']==0, 'New terminal result or solution: revise conclusions'
            assert not (attempt/'incumbent.sol').exists(), 'New incumbent: revise attribution and result table'
            timer = re.match(r'\s*([\d.]+)([smhd])', obj['last_log_row'])
            log_seconds = float(timer[1])*dict(s=1,m=60,h=3600,d=86400)[timer[2]]
            log_newer = log_seconds > p['scip_seconds']
            observed = obj['log_modified_beijing'] if log_newer else p['observed_utc']
            observed = dt.datetime.fromisoformat(observed).astimezone(ZONE)
            obs_seconds = max(log_seconds, p['scip_seconds'])
            restart = obj['restarts'][0]
            cells = obj['last_log_row'].split('|')
            stage = '重启后根节点；存活待核实' if log_newer and int(cells[1])==1 else f'重启后第{p["nodes"]}节点'
            rows.append(dict(method=method,label=label,observed_solve_seconds=obs_seconds,
                             duration=duration(obs_seconds),solutions=0,proven_optimal=False,
                             lower_bound_real=p['dual_bound_real'],upper_bound_real=None,gap='infinite',
                             first_run_restart_node=restart['node'],stage=stage,
                             observed_beijing=observed.isoformat(),
                             cpu_model=r['hardware']['cpu_model'],device=r.get('policy',{}).get('device') or 'cpu',
                             model_sha256=r['model_sha256'],attempt=obj['attempt']))
            attempts.append(obj)
    signatures = [{k:a['result'][k] for k in ('model_sha256','parameters','scip_version','binding_sha256','libscip_sha256')}
                  for a in attempts if a['latest']]
    assert all(x==signatures[0] for x in signatures)
    proof = json.loads(capture(DIAG/'feasibility.json', Path('independent_constructor/feasibility.json'))[0])
    readback = json.loads(capture(DIAG/'solution_readback.json', Path('independent_constructor/solution_readback.json'))[0])
    sol_bytes = capture(DIAG/'tree_candidate.sol', Path('independent_constructor/tree_candidate.sol'))[0]
    capture(AIR/'reports/diagnose_feasibility.py', Path('independent_constructor/diagnose_feasibility.py'))
    capture(Path(__file__).resolve(), Path('build_airwiring_summary.py'))
    assert proof['model_sha256']==rows[0]['model_sha256']
    assert proof['scip_original_solution_check'] and proof['independent_row_check_passed']
    assert readback['solution_file_readback_checked'] and sha(sol_bytes)==proof['solution_sha256']
    assert not proof['optimization_called'] and not proof['live_jobs_modified']
    at = dt.datetime.now(ZONE).strftime('%Y-%m-%d %H:%M')
    data_rel = str(data_dir.relative_to(OUT))
    table_header = '| 方法 | 已观测累计求解时间 | 可行解数 | 最优性证明 | 下界（Real） | 最近证据（北京） |\n|---|---:|---:|---|---:|---|\n'
    table_rows = '\n'.join(f"| {r['label']} | ≥{r['duration']} | 0 | 无 | 30.762624 | {dt.datetime.fromisoformat(r['observed_beijing']).strftime('%m-%d %H:%M')} |" for r in rows)
    stage_rows = '\n'.join(f"| {r['label']} | {r['first_run_restart_node']} | {r['stage']} | {r['cpu_model']} |" for r in rows)
    brief = ('最大航空布线实例已完成同模型的原生SCIP、Mamba与原TGPPO B0分支部署。'
             '截至本次快照，三组在线作业都未找到可行解，下界均约30.762624。'
             '后续独立树形构造在约5.2秒内给出目标37.586984的可行方案，通过全部原始约束核验，'
             '但它没有调用TGPPO或Mamba，尚未注入三组作业，也未证明最优。'
             '因此当前不能宣称TGPPO已解出该实例或任何学习方法胜出；'
             '下一步建议三组统一使用此热启动，比较后续优化和最优性证明效率。')
    md = f'''**最大航空布线实例：SSSP、Mamba、TGPPO 求解结果汇报**

数据截点：北京时间 **{at}**。三组在线结果与独立构造诊断分开核对，原始作业不变。本材料可以作为TGPPO主线组会报告的航空实例补充，不与全66测试结果混合。

**直接回答：目前不是TGPPO求解出了可行解。** 目标 **37.586984** 的方案来自独立树形构造程序。该程序没有使用TGPPO或Mamba，也没有将候选注入任何一组正在运行的作业；它只能作为独立构造结果和模型可行性证据。三方法截至下表各自最新记录，均仍为0个可行解。

本次研究对象为 `edges.csv` 与 `pairs-4-246.csv`，k=4：原图546节点、641边、246点对；原MILP为258,164变量（5,736二进制、252,428连续）及162,800约束。三个方法读取同一CIP，保持相同SCIP8.0.4、绑定／动态库、solver seed0、单线程、容差1e-6、默认预处理／割／启发式；无已知最优cutoff，无求解时间限制，gap／绝对gap阈值0。平台时限另行约束。目标值按原程序除以1e6的Real口径记录。

SSSP组是 `sssp_0923.cpp` 的原图化简／建模加SCIP原生分支；Mamba组使用冻结CL44／IL50模型、24步历史；TGPPO组使用原B0 seed11、开发选定episode100 Actor。学习分支只改变分支变量选择，不直接生成完整路由方案。本次不包含最新MLP、BC改进或持续专家监督模型，也没有在航空实例上再训练／选模。

{table_header}{table_rows}

三组在线上界均缺失，gap为无穷，没有最优性证明。表中是每个作业最后可核实的求解累计值，带“≥”表示未完成记录，不是完成时间，也没有把静默后的日历时长冒充计算时间。程序节点快照与文本日志可能在不同阶段刷新，本次取累计时间更晚的证据；对应更新时间单列，完整原文与SHA保存在数据目录。

| 方法 | 首轮内部重启节点 | 最新可见阶段 | 当前CPU型号 |
|---|---:|---|---|
{stage_rows}

SSSP和TGPPO已走出内部重启后的根节点，继续产生分支节点；重启后的节点计数重新开始，不能据此解释为总搜索节点减少。Mamba最后文本日志仍为11:43的重启后根节点，平台存活尚未核实。`running`只是持久化标记。旧GPU尝试另外保留，CPU重投从头搜索，没有恢复旧树，也不把两次耗时拼成一次连续运行。

新增的独立诊断结果如下，其来源必须与上表三方法分开。

| 指标 | 独立树形构造 |
|---|---:|
| 方案目标（Real） | **37.5869839666382** |
| 相同CIP已知下界 | 30.762624190748827 |
| 与已知下界之差 `(UB-LB)/LB` | **22.18%** |
| 方案构造 | 第0层以N287为根的树，覆盖63终端，再分配246项业务流 |
| 验证 | 原162,800约束、变量边界、整数性全通过；独立逐行检查无违反；SOL读回通过 |
| 读取、构造和检查 | 约5.224秒，不含Python导入；无MILP优化搜索 |
| 来源归属 | 独立结构启发式，未使用TGPPO或Mamba |

所以可以汇报“当前模型已有经验证的可行方案”，但不能说“TGPPO5秒解出最大图”。22.18%只是这份候选与已有下界之间的证书差，既不是证明最优，也不是已知的真实最优性差距。此结果证明当前CIP可行，不代替工程建模语义审查。

对停滞原因，目前有证据支持的解释是：缺少专门利用路由树结构的首解构造和统一热启动；43组业务流展开出246,648连续流变量、123,324容量耦合约束，LP与整数连通选择计算重；四层复制带来大量对称，SCIP已检测158个对称生成元，且下界长期未改善。三组约千节点均触发默认估计驱动的内部重启，又支付根节点处理成本。分支模型不直接改变上述LP／建模结构。各因素的独立耗时贡献尚未消融，不能定量声称哪一项占主导，也不能把迁移失配当作已证实根因。

当前成果应表述为“完成同模型方法接入与CPU部署，并独立验证出一个可行方案”。当前没有学习分支获解或胜过原生的证据；Mamba CPU型号不同，三组观察预算和运行状态不一致，不能直接按节点数排名或报告加速比。TGPPO还有SCIP6／Torch1.13到SCIP8／Torch2.2的部署变化，Mamba CPU／CUDA近并列动作也不保证一致。

下一步建议先核实Mamba作业存活和各平台时限，再安排三组共享同一候选热启动的独立对照；随后核对层语义、容量界／连通表达、成本缩放和默认重启是否值得调整。本次只整理已有结果，不停止／重投作业、不注入候选、不修改求解参数或开展新训练。

今晚可以直接照读：

> {brief}

对应 [三页补充幻灯片]({STEM}.pptx)。可复核数据：[汇总CSV]({data_rel}/comparison.csv)、[原始快照与证据索引]({data_rel}/snapshot.json)、[独立候选SOL]({data_rel}/independent_constructor/tree_candidate.sol)、[可行性核验]({data_rel}/independent_constructor/feasibility.json)。数据快照保留全部5次尝试，新旧尝试不合并计成绩。
'''
    (OUT/(STEM+'.md')).write_text(md)
    with (data_dir/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    snapshot=dict(captured_beijing=at,rows=rows,attempts=attempts,independent_constructor=proof,
                  attribution='independent tree constructor; not TGPPO or Mamba',
                  same_model_solver_verified=True,evidence=evidence,new_solves=0,new_training=0)
    (data_dir/'snapshot.json').write_text(json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n')
    build_slides(rows,proof,at,brief)
    for name,expected in preserved.items():
        assert sha((ROOT/name).read_bytes())==expected
    audit=dict(passed=True,captured_beijing=at,slide_count=3,main_tgppo_materials_unchanged=preserved,
               snapshot=str((data_dir/'snapshot.json').relative_to(ROOT)),evidence_files=len(evidence),
               candidate_attributed_to_tgppo=False,candidate_injected_into_jobs=False,new_solves=0,new_training=0,
               pptx_reopened=True,pptx_zip_crc_valid=True,all_shapes_within_slide=True,
               pptx_visually_rendered=False)
    (data_dir/'DELIVERY_AUDIT.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(report=str(OUT/(STEM+'.md')),slides=str(OUT/(STEM+'.pptx')),data=str(data_dir),
                          captured_beijing=at,rows=rows,audit_passed=True),indent=2,ensure_ascii=False))


def build_slides(rows,proof,at,brief):
    from pptx import Presentation
    from pptx.util import Inches,Pt
    from pptx.dml.color import RGBColor
    from pptx.enum.text import MSO_ANCHOR,MSO_AUTO_SIZE
    from pptx.oxml.xmlchemy import OxmlElement
    prs=Presentation();prs.slide_width=Inches(13.3333);prs.slide_height=Inches(7.5)
    navy,teal,gray='14283F','137F77','647484'

    def text(sl,x,y,w,h,body,size=21,color=navy,bold=False):
        box=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=box.text_frame
        tf.word_wrap=True;tf.auto_size=MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
        tf.margin_left=tf.margin_right=Inches(.02);tf.margin_top=tf.margin_bottom=Inches(.02)
        for i,line in enumerate(body.split('\n')):
            p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line
            p.font.name='Microsoft YaHei';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=RGBColor.from_string(color)
            p.space_after=Pt(10)
            for run in p.runs:
                ea=OxmlElement('a:ea');ea.set('typeface','Microsoft YaHei');run._r.get_or_add_rPr().append(ea)

    def slide(title,take,index):
        sl=prs.slides.add_slide(prs.slide_layouts[6]);sl.background.fill.solid();sl.background.fill.fore_color.rgb=RGBColor(255,255,255)
        text(sl,.6,.3,12,.3,'AIRWIRING  /  2026.11.06  /  组会补充',11,teal,True)
        text(sl,.6,.85,12.1,.65,title,29,navy,True)
        text(sl,.6,1.57,12.1,.6,take,20,teal)
        text(sl,.6,7.05,11.5,.25,f'快照 {at}（北京时间）｜原始证据见同名汇报正文',10,gray)
        text(sl,12.1,7.05,.6,.25,f'{index}/3',10,gray)
        return sl

    sl=slide('三方法仍未找到可行解，尚无方法胜负结论','同一最大航空MILP：25.8万变量、16.28万约束；SCIP8、单线程、seed0、无限时',1)
    vals=[['方法','已观测求解时长','可行解','下界（Real）'],
          *[[r['label'],'≥'+r['duration'],'0','30.762624'] for r in rows]]
    tab=sl.shapes.add_table(4,4,Inches(.65),Inches(2.45),Inches(12),Inches(2.0)).table
    widths=[3.5,3.5,1.7,3.3]
    for j,w in enumerate(widths):tab.columns[j].width=Inches(w)
    for i,row in enumerate(vals):
        tab.rows[i].height=Inches(.5)
        for j,value in enumerate(row):
            cell=tab.cell(i,j);cell.text=value;cell.vertical_anchor=MSO_ANCHOR.MIDDLE
            cell.fill.solid();cell.fill.fore_color.rgb=RGBColor.from_string(navy if i==0 else 'F1F7F7')
            for p in cell.text_frame.paragraphs:
                p.font.name='Microsoft YaHei';p.font.size=Pt(18);p.font.bold=i==0
                p.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else navy)
    text(sl,.7,4.9,11.9,1.7,'• 三组上界均缺失、gap无穷，无最优性证明。\n• SSSP／TGPPO在内部重启后继续分支；Mamba日志停更，存活待核实。\n• 时长为可核实的累计记录；CPU异构与观察预算不同，不报告加速比。',20)
    sl.notes_slide.notes_text_frame.text=brief
    sl=slide('37.586984 来自独立树形构造程序','这份候选没有调用TGPPO或Mamba；尚未注入三组在线作业',2)
    text(sl,.75,2.55,5.5,1.0,'37.586984',43,teal,True)
    text(sl,.75,3.65,5.5,1.6,'候选目标（Real口径）\n已知下界 30.762624\n上下界相对差 22.18%',23)
    text(sl,6.65,2.55,5.9,3.5,'• 第0层构造路由树，再分配业务流。\n• 读取、构造、检查共约5.2秒。\n• 162,800条原约束、边界、整数性全部通过。\n• SOL保存后读回验证通过。\n• 未进行MILP搜索，未证明最优。',23)
    text(sl,.75,6.2,11.8,.55,'可以汇报“模型已有经验证可行方案”；不能汇报“TGPPO 5秒解出最大图”。',21,'B2534A',True)
    sl.notes_slide.notes_text_frame.text='方案是独立结构启发式的成果，不属于三组中的任何一组。5.224秒包含读取、构造、检查，不含Python导入；不是最优求解用时。候选目标与已知下界的差不能当作真实最优值已知。'
    sl=slide('已完成接入与可行性验证，下一步统一热启动','首解构造、LP规模与层对称是重点诊断方向，因果贡献仍需对照验证',3)
    text(sl,.75,2.4,11.85,3.6,'• 已完成：同一MILP接入三种分支策略；Mamba／TGPPO可用CPU部署。\n• 已发现：95.54%变量为业务连续流，75.75%约束为f-X容量耦合。\n• 已观察：下界长期不变、158个对称生成元，三组在约千节点触发内部重启。\n• 结果边界：三组未获解；独立构造获可行解；最优性均未证明。\n• 建议下一步：核实作业存活，三组共享同一候选热启动，再比较证明效率。',23)
    text(sl,.75,6.35,11.8,.4,'现有作业、参数和历史成绩未改；没有新增训练或MILP搜索。',17,gray)
    sl.notes_slide.notes_text_frame.text='分支模型只选择变量，不直接生成完整路由或强化LP。不能把所有停滞都归咎于神经模型，也不能单凭一份构造方案证明热启动后一定很快求到最优。后续实验仅是建议。'
    prs.core_properties.title='航空布线最大实例：三方法求解结果与独立可行性诊断'
    prs.core_properties.subject='明确区分SSSP、Mamba、TGPPO在线成绩和独立树形构造'
    path=OUT/(STEM+'.pptx');prs.save(path)
    reopened=Presentation(path);assert len(reopened.slides)==3
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
    for slide in reopened.slides:
        for shape in slide.shapes:
            assert shape.left>=0 and shape.top>=0
            assert shape.left+shape.width<=reopened.slide_width and shape.top+shape.height<=reopened.slide_height


if __name__=='__main__':
    main()
