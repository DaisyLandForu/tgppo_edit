#!/usr/bin/env python3
"""Build meeting artifacts from completed records; never run training or SCIP."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import base64
import csv
import hashlib
import html
import json
import math
import re
import shutil
import sys
import textwrap
import unicodedata
import zipfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'report'
R = ROOT / 'research'
DATA = OUT / 'data'
FIG = OUT / 'figures'
DATA.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)
S66 = R / 'docs/methods/tgppo_light_actor_test66_20261105'
LIGHT = R / 'docs/methods/tgppo_light_actor_20261105'
SUP = R / 'docs/methods/tgppo_supervised_ppo_20261105/results'
RUN = R / 'tgppo/runs/light-actor-20261105'
SOURCES = {}


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def track(p):
    p = Path(p)
    SOURCES[str(p.relative_to(ROOT))] = digest(p)
    return p


def read(p):
    return json.loads(track(p).read_text())


def csvwrite(name, rows):
    with (DATA / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


preservation = {}
for label, p in [('test66', S66 / 'FINAL_AUDIT.json'), ('supervised', SUP / 'FINAL_AUDIT.json')]:
    audit = read(p)
    assert audit['passed']
    for name, sha in audit['source_output_sha256'].items():
        q = R / name
        assert digest(q) == sha, f'Original result changed: {q}'
        track(q)
    preservation[label] = len(audit['source_output_sha256'])

test = read(S66 / 'REVIEW_DATA.json')
supervised = read(SUP / 'REVIEW_DATA.json')
split = read(R / 'docs/stage1/protocol_draft.json')['data']
expert = read(RUN / 'offline/data_audit.json')
assert test['valid_tasks'] == 1188 and supervised['valid_tasks'] == 110
assert not supervised['gate_passed']
assert not set(split['train']) & set(split['development'])
assert not set(test['panels']['all']) & set(split['train'] + split['development'])
assert set(expert['train_instances'] + expert['validation_instances']) <= set(split['train'])

copies = {
    'test66_summary.csv': S66 / 'summary.csv',
    'test66_effects.csv': S66 / 'effects.csv',
    'test66_per_instance.csv': S66 / 'instances.csv',
    'test66_cohorts.csv': S66 / 'cohorts.csv',
    'development_8actors.csv': LIGHT / 'continuation/development_panels.csv',
    'development_effects.csv': LIGHT / 'continuation/factor_effects.csv',
    'cross8_effects.csv': LIGHT / 'continuation/cross_factor_effects.csv',
    'bc_initialization.csv': LIGHT / 'results_review/bc.csv',
    'training_inventory.csv': LIGHT / 'results_review/training.csv',
    'supervised_dev.csv': SUP / 'verified_summary.csv',
    'supervised_seed_effects.csv': SUP / 'fixed_budget_effects.csv',
}
for name, source in copies.items():
    shutil.copyfile(track(source), DATA / name)

training_rows=list(csv.DictReader((DATA/'training_inventory.csv').open()))
assert len(training_rows)==16
for row in training_rows:
    summary=read(RUN/row['variant']/'training'/('B0-s'+row['training_seed'])/'summary.json')
    assert summary['status']=='training_complete' and summary['trained_episodes']==100
    assert summary['complete_ppo_updates']==int(row['ppo_updates'])
    assert summary['optimizer_steps']==int(row['optimizer_steps'])

documents = [
    R / 'docs/diagnostics/tgppo_20261030/MEETING_BRIEF_20261030.md',
    R / 'docs/diagnostics/tgppo_20261030/01_paper_limitations_evidence_matrix.md',
    R / 'docs/diagnostics/tgppo_20261102_signal_localization/REPORT.md',
    R / 'docs/diagnostics/tgppo_20261102_ppo_replay/REPORT.md',
    R / 'docs/diagnostics/tgppo_20261102_head_reset/REPORT.md',
    R / 'docs/diagnostics/tgppo_20261102_tail8_probe/REPORT.md',
    R / 'docs/diagnostics/tgppo_20261102_end2end/REPORT.md',
    R / 'docs/methods/tgppo_candidate_residual_20261104/cross_instance/REPORT.md',
    LIGHT / 'continuation/METHOD.md', LIGHT / 'continuation/DEVELOPMENT_REVIEW.md',
    LIGHT / 'continuation/RESULTS_REVIEW.md',
    LIGHT / 'bc_only_ablation_20261105/REPORT.md',
    LIGHT / 'dev_curve_20261105/RESULTS_REVIEW.md',
    S66 / 'RESULTS_REVIEW.md', S66 / 'BASELINE_PAPER_ALIGNMENT_20261106.md',
    SUP / 'RESULTS_REVIEW.md',
    R / 'tgppo/experiments/light-actor-20261105/code/project/policy/light_actor.py',
    R / 'tgppo/experiments/light-actor-20261105/code/project/policy/actor.py',
    R / 'tgppo/experiments/light-actor-20261105/code/pretrain.py',
    R / 'tgppo/experiments/light-actor-20261105/configs/mlp_bc.json',
]
for p in documents:
    track(p)

group = test['groups']['all']
b0 = group['b0']['pooled']
mlp = group['mlp']['pooled']
bc = group['mlp_bc']['pooled']
common = test['groups']['common_actor']
changes = {
    'actor_ms_reduction_percent': 100 * (1 - mlp['actor_ms'] / b0['actor_ms']),
    'parameter_reduction_percent': 100 * (1 - 24493 / 4738895),
    'proof_rate_increase_percentage_points': 100 * (mlp['proof_rate'] - b0['proof_rate']),
    'all66_pdi_midpoint_change_percent': 100 * (mlp['pdi'] / b0['pdi'] - 1),
    'common57_pdi_change_percent': 100 * (common['mlp']['pooled']['pdi'] / common['b0']['pooled']['pdi'] - 1),
    'optimize_wall_change_percent': 100 * (mlp['wall'] / b0['wall'] - 1),
    'par2_change_percent': 100 * (mlp['par2'] / b0['par2'] - 1),
    'bc_all66_pdi_change_percent': 100 * (bc['pdi'] / mlp['pdi'] - 1),
    'all66_pdi_measurement_delta_bounds': [mlp['lower'] - b0['upper'], mlp['upper'] - b0['lower']],
    'common57_pdi_measurement_delta_bounds': [common['mlp']['pooled']['lower'] - common['b0']['pooled']['upper'], common['mlp']['pooled']['upper'] - common['b0']['pooled']['lower']],
    'measurement_bounds_are_statistical_confidence_intervals': False,
}
assert round(changes['actor_ms_reduction_percent'], 2) == 92.83
assert round(changes['proof_rate_increase_percentage_points'], 2) == 3.79
assert round(changes['common57_pdi_change_percent'], 2) == -3.10
assert changes['all66_pdi_measurement_delta_bounds'][0] < 0 < changes['all66_pdi_measurement_delta_bounds'][1]
(DATA / 'key_claims.json').write_text(json.dumps(changes, ensure_ascii=False, indent=2) + '\n')

rows = []
sets = {'ppo_train': split['train'], 'development': split['development'],
        'bc_fit': expert['train_instances'], 'bc_internal_selection': expert['validation_instances'],
        'test66': test['panels']['all'], 'test66_common_actor': test['panels']['common_actor'],
        'test66_no_actor': test['panels']['no_actor'], 'test66_mixed_actor': test['panels']['mixed_actor'],
        'cross8_exploration': ['bell5', 'bell3a', 'neos-1215259', 'neos-612125', 'sp98ir', 'neos-512201', 'n3seq24', 'bc1']}
for purpose, names in sets.items():
    for name in names:
        rows.append(dict(purpose=purpose, instance=name))
csvwrite('datasets.csv', rows)

appendix = ['# 数据划分与训练／评测协议附录', '', '截至2026-11-06。名单从冻结划分、专家审计及全66复核JSON读取；不是按本次结果重选。', '']
for key, title in [('ppo_train', 'PPO训练20实例'), ('development', '开发5实例'), ('bc_fit', 'BC拟合12实例（1468状态）'),
                   ('bc_internal_selection', 'BC内部选模3实例（288状态）'), ('cross8_exploration', '8实例探索'), ('test66', '完整66测试目录实例'),
                   ('test66_no_actor', '全66中全模型均无Actor的5实例'), ('test66_mixed_actor', '全66中混合参与的4实例')]:
    appendix += ['## ' + title, '', '、'.join(sets[key]) + '。', '']
appendix += ['## 关键语义', '',
             '- BC内部选模实例属于PPO训练集，不能把PPO后的专家指标称作未见实例泛化。开发实例不参与BC梯度或PPO训练。',
             '- 既有60份relpscost专家轨迹来自SCIP8／无cutoff，PPO使用SCIP6／已知最优cutoff；候选特征13／14计数分母也有差异。仅作离线迁移，不当作当前PPO轨迹。',
             '- 原66目录已在历史研究中反复查看，不能说完全未见或新盲测。8例是66子集，不与66相加成74个独立测试实例。',
             '- 全66每学习条件264任务＝66实例×2 solver seeds×2 training seeds；原生132，共1188。共同Actor57按全部八模型、两个solver均实际调用划分。',
             '- 旧16组均100episode、原H3／Critic／GAE、4episode收集批、3优化epoch、逻辑batch256／微批16。BC先完成1000步再按内部固定里程碑选初始化；全66固定PPO100。',
             '- 新持续监督独立4组100episode，开发在0／20／100中按全5实例PDI选模；不与旧16组权重或不同批次绝对PDI混算。',
             '- SCIP6.0.1、SoPlex4.0.1、定制PySCIPOpt1.4.9；关闭启发式、已知最优cutoff，原生同外围设置。不是默认生产SCIP。',
             '- 120秒PDI按相同窗口比较，过冲保留积分上下界；证明还须SCIP时间≤120秒。测量界不是统计CI。', '',
             '## 训练清单', '', '逐组完成episode、有效PPO轮数、优化步、transition、耗时、硬件和权重SHA见[data/training_inventory.csv](data/training_inventory.csv)。', '',
             '## 评测覆盖进度', '',
             '| 阶段 | 模型任务 | 原生任务 | 合计 | 状态 |', '|---|---:|---:|---:|---|',
             '| 三结构完整开发 | 160 | 10 | 170 | 完成 |', '| 冻结MLP结构2×2的8实例探索 | 128 | 16 | 144 | 完成 |',
             '| BC-only局部消融 | 16 | 4 | 20 | 完成 |', '| BC／PPO20／PPO100开发曲线 | 60 | 10 | 70 | 完成 |',
             '| 旧B0／MLP×BC的全66覆盖 | 1056 | 132 | 1188 | 完成 |', '| 新持续监督三时间点开发 | 100 | 10 | 110 | 完成；筛选未通过 |', '',
             '各阶段可能重复实例，表中任务数不能当成独立样本量；旧中断开发记录保留，不重复计作新有效覆盖。SET／ISAB只完成开发，持续监督只完成开发，没有它们的全66结果。', '',
             '论文对齐与统计限制见[正文第10节](TGPPO_组会汇报_20261106.md)和[S11](SOURCES.md#s11)。', '']
(OUT / '数据划分与协议附录.md').write_text('\n'.join(appendix))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'svg.fonttype': 'none', 'axes.titleweight': 'bold'})
NAVY = '#14283F'
TEAL = '#137F77'
BLUE = '#4575A5'
AMBER = '#BC7E26'
RED = '#B2534A'
GRAY = '#778594'
def savefig(fig, name):
    for ext in ['svg', 'png', 'pdf']:
        fig.savefig(FIG / (name + '.' + ext), dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# A same-run online cost comparison, separate from end-to-end time.
fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.6))
for ax, title, vals, labels, xmax in [
    (axes[0], 'Actor latency (ms/call)', [b0['actor_ms'], mlp['actor_ms']], ['B0', 'MLP'], 70),
    (axes[1], 'SCIP optimize (seconds)', [b0['wall'], mlp['wall']], ['B0', 'MLP'], 125),
    (axes[2], 'Proof completion (%)', [100*b0['proof_rate'], 100*mlp['proof_rate']], ['B0', 'MLP'], 30)]:
    ax.barh(labels, vals, color=[GRAY, TEAL], height=.52)
    ax.invert_yaxis(); ax.set_xlim(0, xmax); ax.set_title(title, fontsize=13, loc='left')
    ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    for i, v in enumerate(vals): ax.text(v+xmax*.025, i, f'{v:.2f}', va='center', color=NAVY, fontsize=13)
fig.tight_layout(w_pad=2.5); savefig(fig, '01_cpu_cost_and_proof')

fig, axes = plt.subplots(1, 2, figsize=(12, 3.7))
for ax, panel, title in [(axes[0], 'all', 'All 66: midpoint / measurement bounds'), (axes[1], 'common_actor', 'Common Actor 57: paired difference')]:
    ref = test['groups'][panel]['b0']['pooled']; val = test['groups'][panel]['mlp']['pooled']
    delta = val['pdi']-ref['pdi']; low=val['lower']-ref['upper']; high=val['upper']-ref['lower']
    ax.errorbar([delta], [0], xerr=[[delta-low], [high-delta]], fmt='o', capsize=10, markersize=9, color=TEAL, linewidth=3)
    ax.axvline(0, color=GRAY, linestyle='--'); ax.set_yticks([]); ax.set_ylim(-.7,.7)
    ax.set_title(title, fontsize=13, loc='left'); ax.set_xlabel('MLP minus B0, PDI (lower is better)')
    ax.text(.03,.87,f'{delta:.2f}  [{low:.2f}, {high:.2f}]', transform=ax.transAxes, fontsize=12, color=NAVY)
    ax.grid(axis='x', alpha=.15)
fig.tight_layout(w_pad=2); savefig(fig, '02_pdi_measurement_bounds')

decomp=list(csv.DictReader(track(S66/'mean_decomposition.csv').open()))
# Read the audited individual tail records instead of estimating contributions.
tail = test['tail_details']
def instance_mean(prefix):
    vals=[r['pdi_horizon_midpoint'] for r in tail if r['instance']=='ns1208400' and r['label'].startswith(prefix+'-s')]
    assert len(vals)==4
    return sum(vals)/4
tail_contrib=(instance_mean('mlp_bc')-instance_mean('mlp'))/66
net=bc['pdi']-mlp['pdi']; other=net-tail_contrib
fig, ax = plt.subplots(figsize=(10.5,3.6))
vals=[tail_contrib, other, net]
ax.barh(['ns1208400', 'Other 65 combined', 'Net, all 66'], vals, color=[RED, TEAL, AMBER], height=.58)
ax.invert_yaxis();ax.axvline(0,color=GRAY);ax.set_xlim(-40,135)
for i,v in enumerate(vals):ax.text(v+(3 if v>=0 else -3),i,f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right',fontsize=13)
ax.set_xlabel('Contribution to mean PDI: MLP+BC minus MLP (positive is worse)')
ax.set_title('BC tail risk: all 66 instances retained',loc='left',fontsize=14)
ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();savefig(fig,'03_bc_tail_risk')

fig, ax=plt.subplots(figsize=(12.2,3.3)); ax.set_xlim(0,12);ax.set_ylim(0,3);ax.axis('off')
def box(x,y,w,h,text,color=TEAL):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',facecolor=color,edgecolor='none'))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',color='white',fontsize=12)
def arrow(x,y,xx,yy): ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':NAVY,'lw':2})
box(.1,1.8,2.1,.75,'Candidate x_i\n25 dimensions',BLUE)
box(.1,.25,2.1,.75,'Node 8 + MIP 53\n61 dimensions',BLUE)
box(2.9,1.8,2.15,.75,'Shared encoder\n25 → 64 → 64')
box(2.9,.25,2.15,.75,'Tree encoder\n61 → 64 → 64')
box(6,1.05,2.4,.9,'Concatenate [h_i, t]\n128 dimensions')
box(9.15,1.05,2.55,.9,'Shared score head\n128 → 64 → 32 → 1')
arrow(2.25,2.18,2.8,2.18);arrow(2.25,.62,2.8,.62)
arrow(5.15,2.18,5.92,1.7);arrow(5.15,.62,5.92,1.25);arrow(8.5,1.5,9.05,1.5)
ax.text(8.8,.3,'Masked softmax across valid candidates → sampled branch',ha='center',color=NAVY,fontsize=12)
fig.tight_layout();savefig(fig,'04_mlp_architecture')

slides=[]
def slide(title, take, bullets=None, table=None, image=None, source='', notes=''):
    slides.append(dict(title=title,take=take,bullets=bullets or [],table=table,image=image,source=source,notes=notes))

slide('把分支策略做轻：TGPPO的方法与实验', '截至2026-11-06｜从诊断到全66评估的结果闭环',
      ['当前最可靠成果：共享树条件化MLP＋原PPO。', 'Actor成本−92.83%；证明率+3.79个百分点；共同Actor57 PDI−3.10%。', 'BC与持续监督的收益边界单独报告，保留全部反例。'],
      source='S8／S10；共同120秒预算，非论文完整性能复现',
      notes='先给最终结论。诊断是选方法的依据，汇报重点是网络改进、真实训练和评测。轻量结构的当前有效证据主要来自无BC条件。')
slide('研究如何从瓶颈走到方法', '先验证机制与任务收益，再决定计算投入',
      table=[['阶段','做了什么','结果去向'],['全面诊断','旧日志／权重／轨迹复用','评分弱响应＋CPU成本切入口'],['局部修补','头重置、8维读出、短在线pilot','机制部分恢复；求解不稳定，停止'],['候选残差CR','6组训练＋局部／跨实例复测','弱局部收益，已终止'],['轻量Actor','3结构×BC×两种子，共16组','完成开发、8例、全66'],['持续专家监督','4组匹配训练＋110开发评估','未过门槛，停止追加']],
      source='S1—S10',notes='说明不是一开始选MLP然后只报赢家。诊断工作不单独计为方法创新。')
slide('诊断确认了什么，又没有确认什么', '本地已训练模型的现象，不外推为论文模型必然失败',
      table=[['证据类别','已确认','结论边界'],['已测事实','近均匀评分；真实PPO参数仍会更新','不是“没有训练”'],['源码机制','多阶段候选压缩；原Actor计算重','未唯一定位LN2或Transformer'],['已测成本','历史Actor约21%、回调约33%墙时','不是全部性能差距的因果份额'],['待验证','奖励／PDI对齐、覆盖与长程信用','没有证明遗忘或奖励是唯一根因']],
      source='S1／S2；原论文局限见TGPPO §6',notes='局部重放动作分布TV仅约1e−8，参数不是没更新。作者明示局限与我们的本地诊断分开。')
slide('未形成稳定收益的路线也保留', '局部响应或单例获胜不足以扩大训练',
      table=[['路线','关键观察','当前决定'],['Critic增强','离线价值改善未传到稳定求解收益','停止当前增强'],['8→1评分尾部','3/4局部响应；在线两来源方向相反','不扩尾部扫描'],['候选残差CR','8例相对S仅−0.07%，种子方向不同','已终止'],['BC初始化','开发／8例正向；全66有尾部退步','不作为稳定正增量'],['持续CE监督','110项闭环；跨种子不一致','不扩episode／λ／全66']],
      source='S2／S3／S8／S10；CL、树信用、历史模块未实现',notes='这些是方法尝试与停止依据，不把未实现的调研写成负实验。CR保留原重Actor，与现在整体替换Actor不同。')
slide('主要方法：共享树条件化MLP', '沿用廉价特征，整体替换重交互与深评分链',
      bullets=['候选25维，节点8＋MIP53维；树编码一次，候选共享参数。', '原五层Transformer／BiMatching／TreeGate → 候选自身表示直达共享非线性读出。', '动态候选、padding与排列等变；Actor仅24,493参数。'],
      image='04_mlp_architecture.png',source='S4；light_actor.py；保留原Critic、H3、GAE和求解设置',
      notes='按图解释两条编码分支，拼接后128→64→32→1，再对有效候选softmax。没有完整二部图，也不是在旧logits外加CR残差。')
slide('三个轻量Actor都进入真实PPO', '比较是否需要集合上下文与内容相关交互',
      table=[['Actor','交互设计','参数','开发Actor ms，无／有BC'],['B0','5层256维、8头Transformer','4,738,895','17.27／17.07'],['MLP','共享候选＋树编码','24,493','1.74／1.71'],['SET','mean／max／log(1+K)集合汇聚','32,749','2.06／2.03'],['ISAB','8诱导点，4头压缩／读回','92,205','3.42／3.57']],
      bullets=['MLP综合成本与开发结果最好；SET／ISAB仅完成开发，没有全66结果。', '结构整体替换含容量、交互、读出与成本变化，不是严格参数匹配因果实验。'],
      source='S4／S5；SET参考Deep Sets，ISAB参考Set Transformer',notes='模型原理不主张是我们新发明。适配和受控实验是当前工作内容，不能把省参数本身当成求解效果。')
slide('训练、专家与评测数据怎样分开', '合法专家迁移；明确内部选模和历史测试暴露',
      table=[['用途','规模','选择与边界'],['PPO训练','20实例','原25中固定划分；solver0／1／2'],['开发','5实例','vpm2／dcmulti／eil33-2／neos-476283／ROI'],['BC拟合／内部选模','12例1468状态／3例288状态','旧relpscost轨迹；不含dev/test'],['8实例探索','8例×2 solver','历史成本分层固定盐哈希；不是新盲测'],['完整测试目录','全部66例×2 solver','无删例；已历史查看，不再称盲测']],
      source='S5／S6／S9；完整实例名单见数据划分附录',notes='BC内部三例仍属于PPO训练集，PPO后专家指标不是未见泛化。专家是SCIP8原生轨迹，有cutoff和两项特征分母迁移边界，不是强分支标签。')
slide('怎样分离结构、BC与PPO增量', '原／轻量Actor × 无／有BC，训练种子11／22',
      table=[['条件','Actor','训练方式'],['B0','原结构','随机初始化→PPO'],['B0＋BC','原结构','BC选中初始化→PPO'],['MLP','轻量结构','随机初始化→PPO'],['MLP＋BC','轻量结构','BC选中初始化→PPO']],
      bullets=['4结构×2初始化×2种子＝16组，均100完整episode、24—25轮PPO更新。', 'BC统一1000步，以内部NLL选初始化；MLP两seed选100步。PPO新优化器。', '120秒／episode；4episode收集批；3优化epoch；H3／GAE不变。'],
      source='S4／S6；同episode不等于相同transition、优化步或墙时',notes='1000是BC更新预算，100是BC选中步，100episode是之后PPO求解次数，三者不要混淆。全66中的+BC都经过PPO。')
slide('开发筛选为何曾选MLP＋BC', '小样本正向支持扩大评估，后来按更大覆盖更新结论',
      table=[['Actor','无BC开发PDI','＋BC开发PDI','证明率'],['B0','3192.67','3186.27','40%'],['MLP','3107.75','3016.98','60%'],['SET','3095.27','3089.19','60%'],['ISAB','3112.20','3036.90','60%']],
      bullets=['冻结8例：MLP＋BC相对B0 PDI−6.25%，BC相对MLP−2.18%，两种子一致。', '8例证明率未提高、仍落后原生；不能把小样本收益推广为全66稳定收益。'],
      source='S5；170开发＋144探索项完成；阶段绝对均值不跨批次混算',notes='当时选MLP+BC有依据，现在无BC结构证据更强是结果更新，不回写历史选模。8例是66子集，不能说74个独立测试实例。')
table66=[['方法','PDI ↓','证明率 ↑','PAR-2秒 ↓','Actor ms ↓']]
for v,label in [('native','relpscost'),('b0','B0'),('b0_bc','B0＋BC'),('mlp','MLP'),('mlp_bc','MLP＋BC')]:
    q=group[v]['pooled'];table66.append([label,f"{q['pdi']:.2f}",f"{100*q['proof_rate']:.2f}%",f"{q['par2']:.2f}",'—' if q['actor_ms'] is None else f"{q['actor_ms']:.2f}"])
slide('全66评估已完成：1188／1188有效', '最终主表保留全部实例，包括失败和无Actor任务', table=table66,
      bullets=['固定旧PPO100权重；8模型×66×2 solver＋132本轮新原生。', 'MLP的效率与部分求解收益保留；MLP＋BC不能写成全量稳定改进。'],
      source='S8／S9；每学习条件264任务，实例才是比较单位',notes='PDI越低越好，证明是已知最优cutoff下的120秒任务证明。原生没有训练种子，分母132。四CPU分片同型号不同主机。')
slide('效率收益明确，端到端收益需分开', '网络成本−92.83%，SCIP optimize墙时−2.83%',
      image='01_cpu_cost_and_proof.png',
      bullets=['Actor占optimize由26.94%降至3.85%；参数减少99.48%。', '特征提取约13.43ms，已超过MLP约4.17ms；下一成本入口在features／callback。', '节点均值增加，不能宣称树缩小；更多节点可能反映吞吐与路径共同变化。'],
      source='S8；在线状态分布不同，Actor／features／callback嵌套计时',notes='证明率17.80→21.59是+3.79个百分点，不是+3.79%。网络快约14倍不等于求解快14倍。')
slide('PDI收益：全量与共同Actor一起报告', '共同Actor57：−3.10%；全66中点−1.08%，测量界跨0',
      image='02_pdi_measurement_bounds.png',
      bullets=['共同57：两训练种子−2.36%／−3.87%，差额测量界均支持改善。', '全66受mspp16时限过冲影响；5条SCIP实际停止约241—261秒。', '5全无Actor、4混合仍留在主表；积分界不是统计置信区间。'],
      source='S8；辅助面板按统一Actor参与规则划分，不删除不利实例',notes='不是选择成功子集。全66主表不变，共同Actor是预定参与分解。全量中点排序与确定积分收益必须区分。')
slide('BC为何没有成为稳定正增量', '少数尾部退步可抵消大量局部胜例',
      image='03_bc_tail_risk.png',
      bullets=['MLP＋BC相对MLP：全66 PDI+3.33%，共同57 +5.36%。', 'ns1208400：MLP证明4／4，BC后2／4；均值贡献+108.49，超过净损失+89.23。', '保留反例；不以删除该实例后转正的敏感性结果宣称成功。'],
      source='S8；BC也有局部赢家，不能推广成专家学习普遍无效',notes='BC拟合可学不等于全局求解可靠。不是只看一例：两个训练种子同结构BC全量中点都退步。')
slide('持续专家监督：最新一轮已闭环', '固定0.01×专家CE/log(K)，没有跨种子稳定收益',
      table=[['固定100episode','全5 PDI','共同Actor4 PDI','证明率'],['匹配普通PPO','3035.89','906.86','60%'],['持续监督PPO','3044.07','908.36','60%'],['监督对照变化，s11','−0.37%','−1.04%','不变'],['监督对照变化，s22','+0.91%','+1.39%','不变']],
      bullets=['4组训练完成；110项开发有效；实际专家batch537／597。', '预定0／20／100选模：control20／20；expert100／0；门槛未过。', '停止当前CE方案的扩训、λ扫描和全66评估。'],
      source='S10；全体差额85.3%来自无Actor ROI，不能归因；共同Actor仍方向不一',notes='seed22选0不是没有训练，而是训练后没胜BC起点。训练硬件seed11有混杂，seed22同类别仍退步。不能写成遗忘已被证明。')
slide('现在可以主张的贡献与边界', '轻量结构成立到哪一步，就汇报到哪一步',
      bullets=['支持：当前预算下显著降低Actor成本，提高证明率，改善共同Actor面板PDI。', '不支持：完整MLP＋BC稳定获益、全66精确PDI必胜、全面超过relpscost。', 'MLP对原生仍PDI+0.97%、证明率−10.23pp、optimize墙时+5.79%。', '当前B0保留原架构，但100episode／120秒／Adam尚未对齐论文完整协议。', '旧结构间CUDA流未完全对齐；只有2训练种子，66集合有历史暴露。'],
      source='S8／S11；不宣称统计显著或生产默认SCIP收益',notes='当前适合讲一项有证据的效率方法与收益边界。论文级新算法或超过论文完整TGPPO还需要强基线与新的确认性证据。')
slide('下一步：先建立强参照，再扩大确认', '停止当前低收益分支，集中验证轻量主体',
      table=[['顺序','工作','晋级／停止依据'],['1','冻结并对齐基线协议、修正新副本随机流','少量共同开发预算先核验；不直接全量开跑'],['2','无BC MLP与强B0匹配训练／选模','跨种子PDI、证明率与成本共同成立'],['3','本地剖析features／callback剩余开销','先复用状态；改路径须核对语义等值'],['4','冻结候选与新外层保留集合','先估时，再按全体指标和尾部确认']],
      bullets=['BC／持续CE不扩当前预算；CL、树信用与历史模型后置。', '本次只整理报告，不启动新训练或求解。'],
      source='当前证据支持的下一阶段建议；待用户验收',notes='不能只给新方法更多训练、保留弱预算B0。信用分配需先解决全局PDI和未完成子树偏差。')
slide('附录：当前B0与论文的差异', '原网络结构保留，协议与性能复现没有完成等同',
      table=[['项目','论文报告','当前主对照'],['Actor','5层256维、8头','保留原结构'],['最终训练','500episode，25实例×5solver','100episode，20训练＋5开发'],['单次求解','3600秒','120秒'],['优化与更新','AdamW，每episode后3epoch','Adam，4episode收集批'],['评测','66×5，easy节点／hard PDI等','66×2，全体PDI／证明／成本']],
      bullets=['历史确有500episode B0，但也是120秒等协议，非本轮权重来源。'],
      source='S11；论文§4.3／§5.3／§5.4',notes='500是总episode数，不是每实例500次。不能按episode和时限比例直接换算性能或计算量。')
slide('附录：PPO在BC之上的贡献尚不稳定', '真实训练已完成；没有收敛或普遍无效的证明',
      table=[['验证','已完成结果','可支持结论'],['20项局部消融','2实例×2训练种子：1改善、3退步','局部PPO附加收益混合'],['70项0／20／100开发','PPO100对BC：全体+0.16%、共同−0.45%','种子方向相反，无稳定增量'],['专家信号保留','MLP NLL从0.833／0.843到0.807／0.820','没有整体遗忘证据'],['110项持续监督','专家loss实际启用，筛选未过','当前固定监督方案停止']],
      source='S6／S7／S10；不要因效果弱而说训练未执行',notes='专家内部NLL是选模数据指标，不是独立泛化。BC-only局部两个实例是已查看的正反例，不能当新测试集合。')

assert len(slides)==18
(OUT / 'slides_content.json').write_text(json.dumps(slides, ensure_ascii=False, indent=2)+'\n')

# Offline HTML, all images embedded. No app, external script, or network required.
def h(s):return html.escape(str(s))
sections=[]
for i,s in enumerate(slides,1):
    content=''
    if s['table']:
        content+='<table>'+''.join('<tr>'+''.join(f'<{"th" if j==0 else "td"}>{h(v)}</{"th" if j==0 else "td"}>' for v in row)+'</tr>' for j,row in enumerate(s['table']))+'</table>'
    if s['image']:
        encoded=base64.b64encode((FIG/s['image']).read_bytes()).decode()
        content+=f'<img class="chart" alt="{h(s["take"])}" src="data:image/png;base64,{encoded}">'
    if s['bullets']:content+='<ul>'+''.join('<li>'+h(v)+'</li>' for v in s['bullets'])+'</ul>'
    sections.append(f'<section class="slide" id="s{i}"><div class="kicker">TGPPO RESEARCH · 2026.11.06</div><h1>{h(s["title"])}</h1><p class="take">{h(s["take"])}</p><div class="content">{content}</div><footer><span>{h(s["source"])}</span><b>{i:02d} / 18</b></footer></section>')
css='''*{box-sizing:border-box}body{margin:0;background:#e8edf1;color:#14283f;font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif}.slide{position:relative;display:none;width:min(100vw,177.78vh);height:min(56.25vw,100vh);margin:auto;background:#fff;padding:3.2% 5%;overflow:hidden}.slide.active{display:block}.kicker{font-size:clamp(8px,1vw,17px);letter-spacing:.16em;color:#137f77;font-weight:700}h1{font-size:clamp(16px,2.8vw,44px);margin:.3em 0;font-weight:750;line-height:1.2}.take{font-size:clamp(10px,1.6vw,26px);color:#137f77;margin:.5em 0 1em}.content{font-size:clamp(10px,1.4vw,22px)}table{width:100%;border-collapse:collapse;font-size:clamp(9px,1.3vw,21px);margin:.3em 0 1em}th{text-align:left;background:#14283f;color:white;padding:.65em}td{padding:.55em .65em;border-bottom:1px solid #dce5e8}tr:nth-child(2n){background:#f2f7f7}ul{padding-left:1.2em;margin:.6em 0}li{margin:.5em 0;line-height:1.4}.chart{display:block;margin:0 auto;width:100%;max-height:29vh;object-fit:contain}footer{position:absolute;bottom:3%;left:5%;right:5%;display:flex;justify-content:space-between;align-items:center;gap:2em;font-size:clamp(7px,.86vw,14px);color:#61717f}button{position:fixed;bottom:8px;z-index:3;border:0;padding:8px 16px;background:#14283f;color:white;border-radius:4px;cursor:pointer}#prev{left:8px}#next{right:8px}.slide:first-child{background:#14283f;color:white}.slide:first-child .take{color:#63d3bf}.slide:first-child .content{margin-top:6%;font-size:clamp(12px,1.9vw,30px)}.slide:first-child footer{color:#bdcbd7}@media print{@page{size:landscape;margin:0}body{background:white}.slide,.slide.active{display:block;width:297mm;height:167mm;break-after:page;margin:0;padding:10mm 15mm;print-color-adjust:exact}h1{font-size:25pt}.take{font-size:16pt}.content{font-size:14pt}table{font-size:12pt}.chart{max-height:68mm}footer{font-size:8pt}button{display:none}}'''
js='''let n=0;const s=[...document.querySelectorAll('.slide')];function show(i){n=Math.max(0,Math.min(s.length-1,i));s.forEach((x,j)=>x.classList.toggle('active',j===n));history.replaceState(null,'','#s'+(n+1))}document.querySelector('#prev').onclick=()=>show(n-1);document.querySelector('#next').onclick=()=>show(n+1);document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(n+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(n-1)}if(e.key==='Home')show(0);if(e.key==='End')show(s.length-1)});show(Number(location.hash.slice(2)||1)-1);'''
(OUT / 'TGPPO_组会幻灯片_20261106.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TGPPO组会｜2026-11-06</title><style>'+css+'</style><body>'+''.join(sections)+'<button id="prev" aria-label="上一页">←</button><button id="next" aria-label="下一页">→</button><script>'+js+'</script></body></html>')

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.oxml.xmlchemy import OxmlElement
from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image
prs=Presentation();prs.slide_width=Inches(13.3333);prs.slide_height=Inches(7.5)
def rgb(c):return RGBColor.from_string(c.lstrip('#'))
def text(sl,x,y,w,h,txt,size=20,color=NAVY,bold=False):
    box=sl.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h));tf=box.text_frame
    tf.word_wrap=True;tf.auto_size=MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
    tf.margin_left=Inches(.02);tf.margin_right=Inches(.02)
    tf.margin_top=Inches(.02);tf.margin_bottom=Inches(.02)
    for i,line in enumerate(txt.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph();p.text=line
        p.font.name='Microsoft YaHei';p.font.size=Pt(size);p.font.bold=bold;p.font.color.rgb=rgb(color)
        p.space_after=Pt(8)
        for run in p.runs:
            ea=OxmlElement('a:ea');ea.set('typeface','Microsoft YaHei');run._r.get_or_add_rPr().append(ea)
    return box
for i,s in enumerate(slides,1):
    sl=prs.slides.add_slide(prs.slide_layouts[6]);dark=i==1
    bg=sl.background.fill;bg.solid();bg.fore_color.rgb=rgb(NAVY if dark else '#FFFFFF')
    text(sl,.6,.3,12,.25,'TGPPO RESEARCH   /   2026.11.06',11,TEAL if not dark else '#63D3BF',True)
    text(sl,.6,.76,12.15,.65,s['title'],30,'#FFFFFF' if dark else NAVY,True)
    text(sl,.6,1.47,12.1,.56,s['take'],20,'#63D3BF' if dark else TEAL)
    y=2.17
    if s['table']:
        rows=s['table']; nr,nc=len(rows),len(rows[0]);font_size=15 if nc<5 else 16
        widths=[2.15,4.25,5.7] if nc==3 else [12.1/nc]*nc
        row_heights=[]
        for row in rows:
            lines=max(math.ceil(sum(1 if unicodedata.east_asian_width(ch) in 'WF' else .57 for ch in str(value))*font_size/((width-.2)*72)) for value,width in zip(row,widths))
            row_heights.append(max(.43,(lines*font_size*1.15+10)/72))
        height=sum(row_heights)
        table=sl.shapes.add_table(nr,nc,Inches(.6),Inches(y),Inches(12.1),Inches(height)).table
        # More space for explanations in matrices.
        if nc==3:
            table.columns[0].width=Inches(2.15);table.columns[1].width=Inches(4.25);table.columns[2].width=Inches(5.7)
        for ri,row in enumerate(rows):
            table.rows[ri].height=Inches(row_heights[ri])
            for ci,val in enumerate(row):
                cell=table.cell(ri,ci);cell.text=str(val);cell.margin_left=Inches(.1);cell.margin_right=Inches(.1)
                cell.text_frame.auto_size=MSO_AUTO_SIZE.TEXT_TO_FIT_SHAPE
                cell.margin_top=Inches(.07);cell.margin_bottom=Inches(.05);cell.vertical_anchor=MSO_ANCHOR.MIDDLE
                cell.fill.solid();cell.fill.fore_color.rgb=rgb(NAVY if ri==0 else ('#F1F7F7' if ri%2 else '#FFFFFF'))
                for p in cell.text_frame.paragraphs:
                    p.font.name='Microsoft YaHei';p.font.size=Pt(font_size);p.font.bold=ri==0
                    p.font.color.rgb=rgb('#FFFFFF' if ri==0 else NAVY)
                    p.space_after=Pt(0)
        y+=height+.15
    if s['image']:
        p=FIG/s['image'];iw,ih=Image.open(p).size
        w=11.9;hh=w*ih/iw
        if hh>2.85:hh=2.85;w=hh*iw/ih
        sl.shapes.add_picture(str(p),Inches((13.3333-w)/2),Inches(y),width=Inches(w),height=Inches(hh));y+=hh+.12
    if s['bullets']:
        txt='\n'.join('• '+line for line in s['bullets'])
        remaining=6.88-y
        size=23 if dark else (17 if s['image'] or s['table'] else 22)
        text(sl,.65,y,12,remaining,txt,size,'#FFFFFF' if dark else NAVY)
    text(sl,.6,7.08,11.55,.22,s['source'],9,'#BCCCD7' if dark else GRAY)
    text(sl,12.25,7.05,.55,.25,f'{i:02d}/18',10,'#BCCCD7' if dark else GRAY)
    sl.notes_slide.notes_text_frame.text=s['notes']+'\n\n来源：'+s['source']
prs.core_properties.title='面向异构MILP的轻量树条件化分支策略'
prs.core_properties.subject='TGPPO系统诊断、轻量Actor、全66结果与监督消融'
prs.core_properties.author='TGPPO研究项目'
ppt=OUT/'TGPPO_组会幻灯片_20261106.pptx';prs.save(ppt)

# Structural delivery checks, with no claim of rendering the PPT on this host.
reopened=Presentation(ppt);assert len(reopened.slides)==18
outside=[]
for i,sl in enumerate(reopened.slides,1):
    for sh in sl.shapes:
        if min(sh.left,sh.top)<0 or sh.left+sh.width>prs.slide_width+20 or sh.top+sh.height>prs.slide_height+20:
            outside.append((i,sh.name))
assert not outside,outside
assert 'torch' not in sys.modules and 'pyscipopt' not in sys.modules
for name,sha in SOURCES.items():assert digest(ROOT/name)==sha, f'Source changed during build: {name}'
with zipfile.ZipFile(ppt) as z:assert z.testzip() is None
local_links=0
for p in OUT.glob('*.md'):
    for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)',p.read_text()):
        if target.startswith(('http:','https:','#')):continue
        assert (p.parent/target.split('#')[0]).exists(),(str(p),target)
        local_links+=1
audit=dict(passed=True,verified_beijing=datetime.now(timezone(timedelta(hours=8))).isoformat(),
           existing_tasks_verified={'test66':1188,'supervised_development':110},
           new_solver_tasks=0,new_training_episodes=0,torch_imported=False,
           unchanged_prior_audit_outputs=preservation,source_files_unchanged=len(SOURCES),
           slide_count=18,pptx_reopened=True,pptx_zip_crc_valid=True,all_shapes_within_slide=True,
           actual_training_summaries_verified=16,local_links_checked=local_links,broken_local_links=0,
           pptx_visual_rendered_on_host=False,figures_exported_as=['PNG','SVG','PDF'],
           statistical_significance_claimed=False,full_paper_performance_reproduction_claimed=False,
           source_sha256=SOURCES,
           output_sha256={str(p.relative_to(OUT)):digest(p) for p in OUT.rglob('*') if p.is_file() and p.name!='DELIVERY_AUDIT.json'})
(OUT/'DELIVERY_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k not in ['source_sha256','output_sha256']},ensure_ascii=False,indent=2))
