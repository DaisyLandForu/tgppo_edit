#!/usr/bin/env python3
"""Read existing evidence; render matching native PPTX, vector PDF/SVG, and PNG.

No Torch, training, solver, or checkpoint mutations. Requires a separate reporting
environment: matplotlib, pillow, python-pptx, PyMuPDF, fonttools (Python >=3.8).
"""
import base64
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import zipfile
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image, ImageDraw
import fitz
from fontTools.ttLib import TTFont
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
R = ROOT/'research'
SOURCE = R/'docs/methods/tgppo_paper100_test66_20261110'
FIG = OUT/'figures'
DATA = OUT/'data'
for p in (FIG, DATA): p.mkdir(parents=True, exist_ok=True)
REG = font_manager.FontProperties(fname=str(OUT/'assets/NotoSansSC-Regular.ttf'))
BOLD = font_manager.FontProperties(fname=str(OUT/'assets/NotoSansSC-SemiBold.ttf'))
plt.rcParams.update({'svg.fonttype':'path', 'pdf.fonttype':42, 'axes.unicode_minus':False})
C = dict(ink='#1F3046', muted='#66778A', line='#CDD7E1', panel='#F5F8FB',
         blue='#3B6DA4', blue_bg='#EAF1F9', teal='#177D79', teal_bg='#E9F5F2',
         orange='#B77D36', orange_bg='#FCF2E5', slate='#75849B', red='#B46659', white='#FFFFFF')
SOURCE_HASHES = {}
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def track(path):
    path = Path(path); SOURCE_HASHES[str(path.relative_to(ROOT))] = sha(path); return path
def read(path): return json.loads(track(path).read_text())
def csvread(path):
    with track(path).open() as f: return list(csv.DictReader(f))

summary = csvread(SOURCE/'summary.csv')
MET = {r['label']:{k:(float(v) if v else None) for k,v in r.items() if k!='label'} for r in summary}
results = read(SOURCE/'results.json')
raw = csvread(SOURCE/'raw_rows.csv')
pairs = csvread(SOURCE/'paired.csv')
effects = csvread(SOURCE/'independent_effects.csv')
assert results['complete'] and results['tasks']==330 and len(raw)==330
assert all(sum(r['label']==label for r in raw)==66 for label in MET)
assert len({r['cohort_id'] for r in raw})==1
bc = {str(s):read(R/f'tgppo/runs/paper-aligned-20261106/training/mlp_bc-s{s}/bc/result.json') for s in (11,22)}
expert = read(R/'tgppo/runs/light-actor-20261105/offline/data_audit.json')
protocol = read(R/'tgppo/experiments/paper-aligned-20261106/configs/protocol.json')
plan = read(R/'tgppo/experiments/paper100-test66-20261110/plan.json')
for f in ['light_actor.py','actor.py','modules.py']:
    track(R/'tgppo/experiments/paper-aligned-20261106/code/project/policy'/f)
for f in ['pretrain.py','expert_data.py','paper.py','project/agent/agent.py']:
    track(R/'tgppo/experiments/paper-aligned-20261106/code'/f)
for name in ['summary.csv','independent_effects.csv','paired.csv','raw_rows.csv']:
    shutil.copyfile(SOURCE/name, DATA/name)
PARAMS = dict(b0=4738895, mlp=24493)
assert 50+(25*64+64)+(64*64+64)+122+(61*64+64)+(64*64+64)+(128*64+64)+(64*32+32)+(32+1)==PARAMS['mlp']
change = lambda a,b,k='mean_pdi':100*(MET[a][k]/MET[b][k]-1)
STATS = dict(metrics=MET, parameters=PARAMS,
    mlp_pdi_vs_new_b0=change('mlp','b0_new'), combined_pdi_vs_new_b0=change('mlp_bc','b0_new'),
    bc_pdi_vs_mlp=change('mlp_bc','mlp'), actor_path_speedup=MET['b0_new']['actor_mean_ms']/MET['mlp']['actor_mean_ms'],
    mlp_optimize_vs_new_b0=change('mlp','b0_new','mean_optimize_seconds'),
    combined_optimize_vs_new_b0=change('mlp_bc','b0_new','mean_optimize_seconds'),
    bc_selected_steps={s:d['selected_step'] for s,d in bc.items()},
    bc_heldout_accuracy={s:d['final']['validation']['macro']['accuracy'] for s,d in bc.items()},
    evaluation='fixed100; training seed11; solver seed0; 66 instances; 3600s; known-optimum cutoff',
    latency_scope='input conversion/checks + network + categorical action/log-prob; excludes feature extraction, Critic and SCIP branch',
    solver_calls=0, training_steps=0)
assert round(STATS['mlp_pdi_vs_new_b0'],2)==-19.35
assert round(STATS['combined_pdi_vs_new_b0'],2)==-17.12
assert round(STATS['bc_pdi_vs_mlp'],2)==2.76
assert round(STATS['actor_path_speedup'],2)==20.43
(DATA/'method_evidence.json').write_text(json.dumps(STATS,ensure_ascii=False,indent=2)+'\n')

class Scene:
    W,H=16.,9.
    def __init__(self, key, title, subtitle='', tag='METHOD', footer=''):
        self.key,self.title,self.subtitle=key,title,subtitle
        self.ops=[]; self.notes=''; self.text_bounds=[]
        self.rect(0,0,16,9,C['white'],None,0)
        self.text(.65,.42,tag,10,C['teal'],True)
        self.text(.65,.95,title,25,C['ink'],True)
        if subtitle:self.text(.65,1.53,subtitle,12,C['muted'])
        self.line([(.65,8.43),(15.35,8.43)],C['line'],.7)
        self.text(.65,8.67,footer or '固定100 episode · 训练seed11 / solver seed0 · 66实例 · 3600秒 · 受控cutoff',8.5,C['muted'])
    def rect(self,x,y,w,h,fill=None,stroke=None,r=.12,lw=.8,dash=False):
        self.ops.append(dict(kind='rect',x=x,y=y,w=w,h=h,fill=fill,stroke=stroke,r=r,lw=lw,dash=dash))
    def circle(self,x,y,r,fill=None,stroke=None,lw=1):
        self.ops.append(dict(kind='circle',x=x,y=y,r=r,fill=fill,stroke=stroke,lw=lw))
    def text(self,x,y,t,size=14,color=None,bold=False,ha='left'):
        # Portable mathematical labels, avoiding absent superscript glyphs in CJK fonts.
        for old,new in [('ᵢ','_i'),('⁶⁴','^64'),('ᴮᶜ','_BC'),('ᴱ','_E'),('10⁻³','1e-3')]:
            t=str(t).replace(old,new)
        self.ops.append(dict(kind='text',x=x,y=y,t=str(t),size=size,color=color or C['ink'],bold=bold,ha=ha))
    def line(self,points,color=None,lw=1.1,arrow=False,dash=False):
        self.ops.append(dict(kind='line',points=points,color=color or C['muted'],lw=lw,arrow=arrow,dash=dash))
    def equation(self,x,y,tex,ppt,size=14,color=None,bold=False,ha='left'):
        self.text(x,y,tex,size,color,bold,ha)
        self.ops[-1].update(ppt_text=ppt,ppt_font='Cambria Math')
    def arrow(self,a,b,color=None,via=None,dash=False):self.line([a]+(via or [])+[b],color,1.35,True,dash)
    def label(self,x,y,w,top,sub='',color='teal',h=.85):
        self.rect(x,y,w,h,C[color+'_bg'],None)
        self.text(x+.16,y+.25,top,13,C[color],True)
        if sub:self.text(x+.16,y+.58,sub,10,C['muted'])
    def band(self,y,title,body,color='teal'):
        self.rect(.65,y,14.7,.8,C[color+'_bg'],None)
        self.text(.9,y+.26,title,13,C[color],True)
        self.text(.9,y+.59,body,10.5,C['ink'])
    def tensor(self,x,y,rows=4,cols=6,w=1.55,h=1.35,color='blue'):
        for i in range(rows):
            for j in range(cols):
                self.rect(x+j*w/cols,y+i*h/rows,w/cols-.035,h/rows-.05,
                          C[color] if j==0 else C[color+'_bg'],None,.025)

def module(s,x,y,w,h,title,body,color='teal'):
    s.rect(x,y,w,h,C['white'],C[color],.12,1.1)
    s.rect(x,y,w,.43,C[color+'_bg'],None,.1)
    s.text(x+w/2,y+.23,title,13,C[color],True,'center')
    for i,line in enumerate(body):s.text(x+w/2,y+.8+i*.35,line,12,C['ink'],False,'center')

def hero():
    s=Scene('01_mlp_actor','轻量 Actor：树上下文条件化的共享候选评分',
            'MLP 与 MLP＋BC 共用同一部署网络；BC改变初始化，不增加推理模块。',footer='源码：LightActor(kind="mlp") · 维度与连接逐项核对 · 参数总数24,493')
    for x,t in [(.8,'01  特征输入'),(3.8,'02  双路编码'),(7.85,'03  条件融合'),(10.1,'04  共享评分'),(13.3,'05  分支分布')]:s.text(x,2.12,t,11,C['muted'],True)
    s.tensor(.85,2.95,4,6,1.8,1.45,'blue')
    s.text(.85,2.63,'候选集合 X',15,C['blue'],True);s.text(1.75,4.68,'L × 25',14,C['blue'],False,'center')
    s.text(.85,5.35,'当前节点与MIP状态',13,C['orange'],True)
    s.label(.85,5.65,.98,'node','8维','orange',.85)
    s.label(1.98,5.65,1.0,'MIP','53维','orange',.85)
    s.text(1.9,6.77,'拼接为61维',11,C['orange'],False,'center')
    module(s,3.65,2.75,2.9,1.85,'候选编码 φ', ['LN(25)','25 → 64 → 64','两层 Linear＋GELU'],'blue')
    module(s,3.65,5.3,2.9,1.85,'树上下文编码 ψ',['LN(61)','61 → 64 → 64','两层 Linear＋GELU'],'orange')
    s.arrow((2.75,3.68),(3.55,3.68),C['blue'])
    s.arrow((3.04,6.08),(3.55,6.08),C['orange'])
    s.arrow((6.65,3.68),(7.9,3.68),C['blue'])
    s.equation(7.2,3.24,r'$h_i\in\mathbb{R}^{64}$','hᵢ ∈ ℝ⁶⁴',12,C['blue'],ha='center')
    s.arrow((6.65,6.08),(8.15,4.55),C['orange'],via=[(7.25,6.08),(7.25,4.55)])
    s.equation(7.25,6.48,r'$t\in\mathbb{R}^{64}$','t ∈ ℝ⁶⁴',12,C['orange'],ha='center')
    s.rect(8.05,3.2,.85,1.9,C['teal_bg'],C['teal'])
    s.text(8.475,3.77,'拼接',13,C['teal'],True,'center');s.equation(8.475,4.26,r'$[h_i;t]$','[hᵢ;t]',12,C['ink'],ha='center')
    s.text(8.5,5.45,'每个候选128维',10.5,C['muted'],False,'center')
    s.arrow((9.0,4.1),(9.65,4.1),C['teal'])
    module(s,9.75,2.9,2.55,2.65,'共享评分头 ρ', ['128 → 64','GELU','64 → 32 → 1','GELU后直接读出标量'],'teal')
    s.text(11.02,5.95,'同一组权重用于所有候选',10,C['teal'],False,'center')
    s.arrow((12.4,4.1),(13.05,4.1),C['teal'])
    s.rect(13.12,2.95,2.12,2.6,C['panel'],None)
    s.text(14.18,3.32,'Masked softmax',12,C['ink'],True,'center')
    for i,v in enumerate([.18,.44,.91,.3]):
        s.rect(13.47,3.8+i*.34,v*1.45,.16,C['teal'] if i==2 else '#BFD9D6',None,.03)
    s.text(14.18,5.2,'π(a=i | X,c)',13,C['teal'],False,'center')
    s.text(13.1,6.4,'采样合法候选 → SCIP分支',11,C['ink'])
    s.band(7.45,'保留候选身份与树状态条件化', '候选轴共享权重、输出随候选排列等变；固定宽度下计算随 L 线性增长。')
    return s

def comparison():
    s=Scene('02_actor_comparison','结构增量：从重交互链到树条件化共享评分',
            '保留原有25/8/53维廉价特征、H3奖励与Critic；改动集中于Actor。')
    s.text(.8,2.05,'TGPPO Actor',18,C['slate'],True)
    s.text(.8,2.48,'自复现架构 · 4,738,895参数',11,C['muted'])
    xs=[.85,3.75,7.05,10.0,13.1];ws=[2.25,2.65,2.3,2.5,1.65]
    labels=[('线性嵌入与融合',['候选25 / 树61','融合到256维']),('Transformer × 5',['宽256 · 8 heads','候选间自注意力']),('BiMatching',['双向树—候选匹配','与树状态再融合']),('TreeGate 评分头',['逐级压缩至1维','含低维LayerNorm']),('Softmax',['候选概率'])]
    for x,w,(t,b) in zip(xs,ws,labels):module(s,x,2.95,w,1.65,t,b,'blue')
    for x,w,n in zip(xs[:-1],ws[:-1],xs[1:]):s.arrow((x+w+.05,3.78),(n-.12,3.78))
    s.text(11.25,4.9,'末端包含 8 → 4 → 2 → 1',10.5,C['muted'],False,'center')
    s.text(.8,5.35,'我们的 MLP Actor',18,C['teal'],True)
    s.text(.8,5.78,'24,493参数 · 减少99.48%',11,C['teal'])
    specs=[(.85,3.5,'候选 / 树双路非线性编码',['25 → 64 → 64；61 → 64 → 64']), (5.05,3.2,'按候选拼接上下文',['候选64维 + 树64维 = 128维']), (9.0,3.25,'共享MLP直接评分',['128 → 64 → 32 → 1']), (13.1,1.65,'Softmax',['候选概率'])]
    for x,w,t,b in specs:module(s,x,6.15,w,1.4,t,b,'teal')
    for (x,w,_,_),(nx,_,_,_) in zip(specs[:-1],specs[1:]):s.arrow((x+w+.05,6.85),(nx-.12,6.85),C['teal'])
    s.text(.85,8.04,'删除候选自注意力与BiMatching；以共享非线性评分替代树门控窄压缩链。',11,C['muted'])
    return s

def training():
    s=Scene('03_bc_ppo_pipeline','训练增量：专家初始化，然后由PPO继续优化',
            'MLP＋BC = 同一轻量Actor ＋ 一次离线BC初始化 ＋ 原有PPO流程。')
    s.rect(.65,2.05,5.3,5.65,C['orange_bg'],None)
    s.text(.95,2.4,'A  离线专家预训练',16,C['orange'],True)
    s.label(1.0,3.02,4.55,'复用 relpscost 专家决策','只取合法训练实例，标签为候选相对索引','orange')
    s.arrow((3.27,3.95),(3.27,4.37),C['orange'])
    module(s,1.0,4.46,4.55,1.17,'共享MLP Actor', ['交叉熵：−log πθ(aᴱ | s)'],'orange')
    s.arrow((3.27,5.72),(3.27,6.05),C['orange'])
    s.rect(1.0,6.08,4.55,.56,C['white'],C['orange'],.1,.8)
    s.text(3.27,6.35,'BC内部留出选初始化 θᴮᶜ',14,C['orange'],True,'center')
    s.text(3.27,6.85,'Adam；1000步预算，实际选中第100步',10.5,C['ink'],False,'center')
    s.rect(6.35,2.05,6.05,5.65,C['teal_bg'],None)
    s.text(6.65,2.4,'B  完整episode的PPO',16,C['teal'],True)
    module(s,6.9,3.02,4.95,1.05,'轻量Actor  πθ', ['MLP：随机初始化；MLP＋BC：θᴮᶜ'],'teal')
    s.arrow((5.65,6.35),(6.8,3.55),C['orange'],via=[(6.1,6.35),(6.1,3.55)])
    s.arrow((9.38,4.18),(9.38,4.58),C['teal'])
    module(s,6.9,4.67,4.95,1.04,'SCIP采样完整轨迹',['H3奖励 · 原Critic估值 · 标准GAE'],'teal')
    s.arrow((9.38,5.82),(9.38,6.19),C['teal'])
    module(s,6.9,6.27,4.95,.98,'PPO更新',['AdamW · 每episode后3轮 · 原clip/entropy'],'teal')
    s.arrow((11.96,6.75),(11.96,3.9),C['teal'],via=[(12.16,6.75),(12.16,3.9)])
    s.arrow((11.96,3.38),(12.95,3.38),C['blue'])
    s.rect(13.05,3.02,2.2,2.72,C['blue_bg'],None)
    s.text(14.15,3.46,'C  CPU部署',14,C['blue'],True,'center')
    s.text(14.15,4.14,'只加载Actor',13,C['ink'],True,'center')
    s.text(14.15,4.62,'提特征 → 选变量',11,C['ink'],False,'center')
    s.text(14.15,5.11,'无在线专家调用',11,C['blue'],False,'center')
    s.text(.9,8.02,'BC只预训练Actor；进入PPO前重建优化器状态。当前方案未加入对比学习或持续专家损失。',11,C['muted'])
    return s

def results_scene():
    s=Scene('04_full66_results','完整66实例：结构收益清楚，BC呈现指标取舍',
            '330/330项有效 · 每方法66项 · 同机配对 · 本轮只覆盖固定100 episode。',tag='RESULTS')
    labels=[('native','relpscost'),('b0','旧B0'),('b0_new','新B0'),('mlp','MLP'),('mlp_bc','MLP＋BC')]
    colors={'native':C['slate'],'b0':'#A8B4C3','b0_new':C['blue'],'mlp':C['teal'],'mlp_bc':C['orange']}
    for x,t,sub in [(2.3,'平均PDI ↓','0–30,000'),(7.25,'证明率 ↑','受控cutoff，时限内证书'),(11.85,'Actor决策路径 ↓','毫秒 / 调用，独立量尺')]:
        s.text(x,2.2,t,16,C['ink'],True);s.text(x,2.59,sub,10,C['muted'])
    for j,(label,name) in enumerate(labels):
        y=3.12+j*.72;col=colors[label]
        s.text(.8,y+.1,name,14,col,True)
        width=MET[label]['mean_pdi']/30000*3.4
        s.rect(2.3,y,width,.22,col,None,.03);s.text(6.15,y+.12,f"{MET[label]['mean_pdi']:,.0f}",12,C['ink'],False,'right')
        width=MET[label]['proof_rate']*2.45
        s.rect(7.25,y,width,.22,col,None,.03);s.text(10.72,y+.12,f"{MET[label]['proof_rate']*100:.2f}%",12,C['ink'],False,'right')
        ms=MET[label]['actor_mean_ms']
        if ms is None:s.text(14.75,y+.12,'—',12,C['muted'],False,'right')
        else:
            s.rect(11.85,y,ms/60*2.2,.22,col,None,.03)
            s.text(15.1,y+.12,f'{ms:.2f}',12,C['ink'],False,'right')
    s.label(.8,7.1,4.45,'MLP vs 新B0：PDI −19.35%','Actor决策路径约快20.43倍；证明率＋7.58pp','teal')
    s.label(5.5,7.1,4.45,'BC vs MLP：证明率＋10.61pp','平均PDI＋2.76%，并非所有指标都改善','orange')
    s.label(10.2,7.1,5.0,'整体仍未超过 relpscost','单训练 / 求解种子，不宣称稳定泛化或论文复现成功','blue')
    return s

def tradeoff():
    s=Scene('05_bc_tradeoff','为什么多证明了7个实例，平均PDI却上升？',
            '证明率看截止时刻是否完成证书；PDI衡量整个求解过程中的gap面积。',tag='INTERPRETATION')
    s.rect(.8,2.2,6.35,4.73,C['orange_bg'],None)
    s.text(1.1,2.65,'BC相对MLP的证明完成变化',16,C['orange'],True)
    s.text(1.1,3.65,'新增 8',32,C['teal'],True);s.text(4.45,3.65,'损失 1',32,C['red'],True)
    s.text(1.1,4.58,'36 → 43 / 66',25,C['ink'],True)
    s.text(1.1,5.25,'净增加7个，证明率＋10.61个百分点',14,C['ink'])
    s.text(1.1,6.19,'仍有逐实例退步，不能把BC描述为全面占优。',11,C['muted'])
    s.rect(7.65,2.2,7.55,4.73,C['panel'],None)
    s.text(7.95,2.65,'一个真实的指标分歧：ns1830653',16,C['ink'],True)
    s.text(7.95,3.47,'方法',12,C['muted']);s.text(10.15,3.47,'PDI ↓',12,C['muted']);s.text(13.5,3.47,'证明完成',12,C['muted'])
    s.text(7.95,4.18,'MLP',18,C['teal'],True);s.text(10.15,4.18,'1,282.45',18,C['ink']);s.text(13.65,4.18,'否',18,C['ink'])
    s.text(7.95,4.93,'MLP＋BC',18,C['orange'],True);s.text(10.15,4.93,'80,679.28',18,C['ink']);s.text(13.65,4.93,'是',18,C['ink'])
    s.text(7.95,5.8,'能证明完成，不代表整个过程的gap面积更小。',12,C['ink'])
    s.text(7.95,6.28,'仅为已观测结果；未构造或伪称测得gap时间曲线。',10,C['muted'])
    s.band(7.35,'BC的准确结论：改善证明覆盖，同时保留PDI尾部风险',
           '平均PDI由22,098.60升至22,709.29（＋2.76%）；不能因证明率提升就忽略主指标退步。','orange')
    return s

def cover():
    s=Scene('slide_01_cover','轻量 Actor 与专家初始化',
            '面向异构MILP分支的 TGPPO 增量方法｜2026.11.13',tag='METHOD & EVIDENCE',footer='本次汇报依据：冻结100 episode模型的完整66实例评测；500 episode正式训练仍在进行。')
    s.text(.9,2.7,'把分支决策做得更轻，',33,C['ink'],True)
    s.text(.9,3.47,'再给策略一个有信息的起点。',33,C['teal'],True)
    s.text(.95,4.55,'结构：树上下文条件化共享MLP',18,C['ink'])
    s.text(.95,5.2,'训练：relpscost行为克隆初始化 → PPO',18,C['ink'])
    s.label(.95,6.35,6.5,'两项可区分的增量','B0 → MLP 识别结构整体；MLP → MLP＋BC 识别初始化','teal',1.05)
    s.rect(9.65,2.4,5.45,4.9,C['panel'],None)
    s.text(10.05,3.07,'−19.35%',36,C['teal'],True)
    s.text(10.05,3.77,'MLP相对新B0的平均PDI',14,C['ink'])
    s.line([(10.05,4.28),(14.7,4.28)],C['line'])
    s.text(10.05,5.08,'31 → 43',36,C['orange'],True)
    s.text(10.05,5.79,'MLP＋BC相对新B0的证明完成数',13,C['ink'])
    s.text(10.05,6.55,'原生relpscost为60/66，仍有明显差距。',11,C['muted'])
    return s

def intuition():
    s=Scene('slide_04_intuition','为什么这种轻量化有合理性？',
            '这些是设计依据与机制解释；本轮没有分别消融每个内部模块。')
    items=[(.8,'01','保留同一候选的身份','逐候选共享φ与ρ，而非先把集合\n汇聚成一个向量；softmax产生竞争。','blue'),
           (5.8,'02','保留搜索状态条件化','节点8维与MIP53维共同编码，\n同一候选在不同树状态可有不同分数。','orange'),
           (10.8,'03','简化可训练的评分路径','64/32维非线性表示后直接读出1维，\n移除重交互与末端低维归一化链。','teal')]
    for x,n,title,body,col in items:
        s.rect(x,2.3,4.4,3.0,C[col+'_bg'],None)
        s.text(x+.25,2.82,n,28,C[col],True)
        s.text(x+.25,3.55,title,16,C['ink'],True)
        for i,line in enumerate(body.split('\n')):s.text(x+.25,4.22+i*.38,line,12,C['ink'])
    s.equation(.95,5.95,r'$z_i=\rho([\phi(x_i);\psi(c)])\qquad \pi_i=\operatorname{softmax}(z)_i$',
               'zᵢ = ρ([φ(xᵢ); ψ(c)])       πᵢ = softmax(z)ᵢ',22,C['ink'])
    s.text(.95,6.65,'排列等变：交换候选行，概率随之交换；模型没有显式候选两两交互。',13,C['muted'])
    s.band(7.35,'主要取舍：降低CPU成本与模型容量，同时放弃显式候选交互',
           '本轮结构组合获得收益；尚不能把19.35%改善单独归因于去注意力、改读出头或去低维LN。')
    return s

def data_scene():
    s=Scene('slide_06_data','数据与训练：把专家初始化和在线PPO分开说明',
            '专家标签来自已有relpscost轨迹；当前没有额外强分支采样、对比学习或持续监督。')
    cols=[(.8,'A  BC数据与拟合','orange',[
        '12个实例 / 1,468个专家状态',
        '另3个BC内部留出 / 288个状态',
        '候选数2–256；保留完整状态',
        '实例均衡采样；交叉熵监督',
        'Adam，lr=10⁻³，batch=16',
        '1000步预算；两种子均选第100步']),
      (5.8,'B  PPO训练协议','teal',[
        '25个训练实例 × solver seeds 0–4',
        '计划500个完整episode；本轮取100',
        '每个有效episode后更新3轮',
        'Actor / Critic独立参数与优化器',
        'H3 + 标准GAE + AdamW',
        '原clip/entropy；不混入专家轨迹']),
      (10.8,'C  本轮完整评测','blue',[
        '四模型 + relpscost，共330项',
        '66个测试实例，solver seed0',
        '训练seed11，固定100 episode',
        '单次3600秒，同机CPU配对',
        'Actor-only部署；动态候选集合',
        '已知最优cutoff的受控求解设置'])]
    for x,title,col,lines in cols:
        s.rect(x,2.25,4.4,4.85,C[col+'_bg'],None)
        s.text(x+.25,2.75,title,16,C[col],True)
        for i,t in enumerate(lines):s.text(x+.25,3.48+i*.48,t,12,C['ink'])
    s.text(.95,7.63,'BC内部留出只用于选择初始化；它们之后仍可进入PPO训练。66测试集不用于挑权重。',11,C['muted'])
    s.text(.95,8.02,'专家SCIP8 → 当前SCIP6存在域差异；当前只有一个训练 / 求解种子的完整66结果。',11,C['muted'])
    return s

def mechanism():
    s=Scene('slide_09_mechanism','如何解释当前收益，哪些还只是待验证假设？',
            '把可测事实与可能机制并排讲，避免把关联直接说成因果。',tag='EVIDENCE')
    rows=[('参数与部署成本','4.74M → 24.49K；Actor决策路径56.64 → 2.77ms',
           '移除重注意力链可直接减少CPU计算。','blue'),
          ('轻量结构的求解结果','相对新B0：PDI −19.35%，证明率＋7.58pp',
           '较简洁的候选评分可能更易优化；子模块因果尚未分离。','teal'),
          ('BC确实学到专家信号','BC内部留出准确率39.93%，均匀动作参考8.35%',
           '提供非随机的排序起点；不能等同最终PPO收益。','orange'),
          ('BC的求解增量有取舍','相对MLP：证明数＋7，但平均PDI＋2.76%',
           '初始化可能改变搜索偏好；当前仍有异构实例尾部退步。','orange')]
    for i,(title,fact,interpret,col) in enumerate(rows):
        y=2.1+i*1.25;s.rect(.8,y,14.4,1.07,C[col+'_bg'],None)
        s.text(1.02,y+.3,title,14,C[col],True)
        s.text(4.15,y+.3,fact,13,C['ink'],True)
        s.text(4.15,y+.76,interpret,11,C['muted'])
    s.text(.95,7.58,'成本更低，不代表求解同比例加速：MLP平均optimize墙时仅改善约2.99%。',13,C['ink'],True)
    s.text(.95,8.02,'MLP的节点几何均值还高于B0；当前收益不能表述为“搜索树普遍更小”。',11,C['muted'])
    return s

def conclusion():
    s=Scene('slide_10_conclusion','今天可以讲清楚的方法结论与下一步',
            '将模块组合贡献、受控实验证据和后续因果问题放在各自的位置。',tag='TAKEAWAYS')
    s.label(.85,2.25,14.3,'方法贡献：轻量结构 + 专家初始化，两项独立可检验的设计',
            '共享MLP、树条件化和BC已有文献基础；当前贡献是对TGPPO的替换、组合与同预算验证。','teal',1.02)
    s.text(1.0,3.95,'已支持',19,C['teal'],True)
    for i,t in enumerate(['轻量Actor显著降低CPU决策路径成本。','固定100 episode下，MLP相对自复现B0的PDI更低。','BC增加本轮证明完成覆盖，同时存在PDI取舍。']):s.text(1.0,4.58+i*.5,t,15,C['ink'])
    s.text(9.0,3.95,'接下来验证',19,C['blue'],True)
    for i,t in enumerate(['完成原定500 episode与两训练种子。','冻结后按原协议评测，不用测试集选模。','若要归因PPO，补BC-only的同条件对照。']):s.text(9.0,4.58+i*.5,t,14,C['ink'])
    s.band(7.2,'当前边界', '整体仍落后relpscost；单种子、测试集有历史暴露，尚无最终500episode或细节因果证据。','blue')
    return s

def appendix():
    s=Scene('slide_11_exact_spec','附录｜可复述的网络与损失定义',
            'MLP与MLP＋BC的部署网络完全一致。',tag='APPENDIX',footer='LN只在25维与61维输入上；评分头没有LayerNorm或门控。当前MLP无Dropout / BatchNorm。')
    lines=[('候选编码 φ','LN(25) → Linear(25,64) → GELU → Linear(64,64) → GELU'),
           ('树编码 ψ','LN(61) → Linear(61,64) → GELU → Linear(64,64) → GELU'),
           ('条件化评分 ρ','拼接为128维 → Linear(128,64) → GELU → Linear(64,32) → GELU → Linear(32,1)'),
           ('动作分布','将padding位置置为−∞，在当前合法候选上softmax，并采样分支动作'),
           ('BC目标','实例均衡的交叉熵 −E[log πθ(aᴱ | s)]；仅初始化Actor'),
           ('PPO目标','标准clipped surrogate＋entropy；独立Critic拟合GAE回报目标')]
    for i,(a,b) in enumerate(lines):
        y=2.18+i*.81;s.text(.9,y,a,14,C['teal'],True);s.text(3.4,y,b,12,C['ink'])
        s.line([(.9,y+.43),(15.1,y+.43)],C['line'],.5)
    s.text(.9,7.47,'参数分解：候选编码5,874 + 树编码8,250 + 评分头10,369 = 24,493',15,C['ink'],True)
    return s

def references():
    s=Scene('slide_12_references','附录｜方法依据与图形参考',
            '借鉴分区、张量维度、模块展开与闭环表示；图形根据本项目源码重新绘制。',tag='REFERENCES',footer='完整链接、源码与数据索引见 SOURCES_AND_EVIDENCE.md；图形不是对原论文模型的截图或复用。')
    refs=[('TGPPO · AAAI 2026 / arXiv 2511.12986','原Actor结构、PPO/H3与25/8/53输入；参考其solver—policy训练闭环。'),
          ('TreeGate · AAAI 2021 / Zarpellon et al.','搜索树条件化的依据；这些廉价特征沿用已有工作，不计作新增贡献。'),
          ('Deep Sets · NeurIPS 2017 / Zaheer et al.','共享参数与排列等变的依据；当前MLP没有集合汇聚，不称完整DeepSets。'),
          ('Set Transformer · ICML 2019 / Lee et al.','参考Fig.1的模块化分图和张量级表达；本方法不使用ISAB。'),
          ('Gasse et al. · NeurIPS 2019','BC与交叉熵用于分支策略的依据，参考Fig.2布局；本项目教师为relpscost。')]
    for i,(title,body) in enumerate(refs):
        y=2.15+i*1.04;s.text(.95,y,title,16,C['ink'],True);s.text(.95,y+.43,body,12,C['muted'])
    return s

SCENES=[cover(),comparison(),hero(),intuition(),training(),data_scene(),results_scene(),tradeoff(),mechanism(),conclusion(),appendix(),references()]

def draw_scene(scene):
    fig=plt.figure(figsize=(16,9),dpi=150,facecolor='white')
    ax=fig.add_axes([0,0,1,1]);ax.set_xlim(0,16);ax.set_ylim(9,0);ax.axis('off')
    texts=[]
    for op in scene.ops:
        k=op['kind']
        if k=='rect':
            ax.add_patch(FancyBboxPatch((op['x'],op['y']),op['w'],op['h'],boxstyle=f"round,pad=0,rounding_size={op['r']}",
                linewidth=op['lw'],edgecolor=op['stroke'] or 'none',facecolor=op['fill'] or 'none',linestyle='--' if op['dash'] else '-'))
        elif k=='circle':ax.add_patch(Circle((op['x'],op['y']),op['r'],facecolor=op['fill'] or 'none',edgecolor=op['stroke'] or 'none',lw=op['lw']))
        elif k=='text':
            texts.append(ax.text(op['x'],op['y'],op['t'],fontproperties=BOLD if op['bold'] else REG,fontsize=op['size'],color=op['color'],ha=op['ha'],va='center',linespacing=1.35))
        elif k=='line':
            pts=op['points'];ax.plot([p[0] for p in pts],[p[1] for p in pts],color=op['color'],lw=op['lw'],linestyle='--' if op['dash'] else '-',solid_capstyle='round')
            if op['arrow']:
                a,b=pts[-2:];dx,dy=b[0]-a[0],b[1]-a[1];norm=math.hypot(dx,dy);ux,uy=dx/norm,dy/norm
                ax.add_patch(Polygon([b,(b[0]-.105*ux+.045*uy,b[1]-.105*uy-.045*ux),(b[0]-.105*ux-.045*uy,b[1]-.105*uy+.045*ux)],closed=True,color=op['color']))
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    bounds=[]
    for t in texts:
        box=t.get_window_extent(renderer);inv=ax.transData.inverted();x0,y0=inv.transform((box.x0,box.y0));x1,y1=inv.transform((box.x1,box.y1))
        if x0<-.02 or x1>16.02 or min(y0,y1)<-.02 or max(y0,y1)>9.02:bounds.append(t.get_text())
    assert not bounds,(scene.key,'text outside canvas',bounds)
    return fig

def ppt_scene(prs,scene,index):
    slide=prs.slides.add_slide(prs.slide_layouts[6]);scale=13.333333/16
    inch=lambda x:Inches(x*scale)
    rgb=lambda c:RGBColor.from_string(c.lstrip('#'))
    for op in scene.ops:
        k=op['kind']
        if k in ('rect','circle'):
            if k=='circle':x,y,w,h=op['x']-op['r'],op['y']-op['r'],2*op['r'],2*op['r'];kind=MSO_SHAPE.OVAL
            else:x,y,w,h=op['x'],op['y'],op['w'],op['h'];kind=MSO_SHAPE.ROUNDED_RECTANGLE if op['r'] else MSO_SHAPE.RECTANGLE
            sh=slide.shapes.add_shape(kind,inch(x),inch(y),inch(w),inch(h))
            if kind==MSO_SHAPE.ROUNDED_RECTANGLE:sh.adjustments[0]=.06
            if op['fill']:sh.fill.solid();sh.fill.fore_color.rgb=rgb(op['fill'])
            else:sh.fill.background()
            if op['stroke']:sh.line.color.rgb=rgb(op['stroke']);sh.line.width=Pt(op['lw']*scale)
            else:sh.line.fill.background()
        elif k=='text':
            width=15.3-op['x'] if op['ha']=='left' else op['x']-.4 if op['ha']=='right' else min(op['x'],16-op['x'])*2-.05
            x=op['x'] if op['ha']=='left' else op['x']-width if op['ha']=='right' else op['x']-width/2
            height=op['size']/72*1.5
            sh=slide.shapes.add_textbox(inch(x),inch(op['y']-height/2),inch(max(width,.2)),inch(height))
            tf=sh.text_frame;tf.margin_left=tf.margin_right=tf.margin_top=tf.margin_bottom=0;tf.word_wrap=False;tf.vertical_anchor=MSO_ANCHOR.MIDDLE
            p=tf.paragraphs[0];p.text=op.get('ppt_text',op['t']);p.alignment={'left':PP_ALIGN.LEFT,'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT}[op['ha']]
            p.font.name=op.get('ppt_font','Microsoft YaHei');p.font.size=Pt(op['size']*scale);p.font.bold=op['bold'];p.font.color.rgb=rgb(op['color']);p.space_before=p.space_after=Pt(0)
        elif k=='line':
            pts=op['points']
            for n,(a,b) in enumerate(zip(pts[:-1],pts[1:])):
                sh=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,inch(a[0]),inch(a[1]),inch(b[0]),inch(b[1]))
                sh.line.color.rgb=rgb(op['color']);sh.line.width=Pt(op['lw']*scale)
                if op['dash']:
                    from pptx.enum.dml import MSO_LINE_DASH_STYLE
                    sh.line.dash_style=MSO_LINE_DASH_STYLE.DASH
                if op['arrow'] and n==len(pts)-2:
                    end=OxmlElement('a:tailEnd');end.set('type','none');sh.line._get_or_add_ln().append(end)
                    end=OxmlElement('a:headEnd');end.set('type','triangle');sh.line._get_or_add_ln().append(end)
    slide.notes_slide.notes_text_frame.text=scene.notes or scene.title+'。详细讲稿见方法详解与讲稿.md。'
    return slide

def build():
    # A changed caption must never silently produce missing-character squares.
    warnings.filterwarnings('error',message=r'Glyph .* missing from current font.*')
    chars={ord(ch) for scene in SCENES for op in scene.ops if op['kind']=='text' for ch in op['t'] if not ch.isspace()}
    for face in ('Regular','SemiBold'):
        cmap=TTFont(OUT/f'assets/NotoSansSC-{face}.ttf').getBestCmap()
        assert not chars-set(cmap),(face,'font subset needs new characters',chars-set(cmap))
    prs=Presentation();prs.slide_width=Inches(13.333333);prs.slide_height=Inches(7.5)
    notes_path=OUT/'slide_notes.json'
    notes=json.loads(notes_path.read_text()) if notes_path.exists() else {}
    images=[]
    pdf=OUT/'MLP_MLPBC_方法讲解_20261113.pdf'
    with PdfPages(pdf) as pages:
        for i,s in enumerate(SCENES,1):
            s.notes=notes.get(s.key,'')
            s.text(15.3,8.67,f'{i:02d} / {len(SCENES):02d}',8.5,C['muted'],False,'right')
            fig=draw_scene(s)
            for ext in ['svg','pdf','png']:fig.savefig(FIG/(s.key+'.'+ext),dpi=180,facecolor='white',bbox_inches=None)
            pages.savefig(fig);plt.close(fig)
            ppt_scene(prs,s,i)
            images.append(FIG/(s.key+'.png'))
    ppt=OUT/'MLP_MLPBC_方法讲解_20261113.pptx';prs.save(ppt)
    reopened=Presentation(ppt);assert len(reopened.slides)==12
    for slide in reopened.slides:
        for shape in slide.shapes:
            assert shape.left>=-2 and shape.top>=-2 and shape.left+shape.width<=prs.slide_width+5 and shape.top+shape.height<=prs.slide_height+5,(slide.slide_id,shape.name)
    with zipfile.ZipFile(ppt) as z:assert z.testzip() is None
    # Actual exported PDFs are rasterized for visual inspection; no fake PPT render claim.
    doc=fitz.open(pdf);assert len(doc)==12
    thumbs=[]
    for i,p in enumerate(doc):
        pix=p.get_pixmap(matrix=fitz.Matrix(1,1));im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples);im.thumbnail((640,360));thumbs.append(im)
    sheet=Image.new('RGB',(1280,6*382),'#E8EDF2');pen=ImageDraw.Draw(sheet)
    for i,im in enumerate(thumbs):x=(i%2)*640;y=(i//2)*382;sheet.paste(im,(x,y));pen.text((x+8,y+362),f'{i+1:02d} {SCENES[i].key}',fill='#1F3046')
    sheet.save(OUT/'PREVIEW_ALL.png')
    sections=[]
    for i,s in enumerate(SCENES):
        encoded=base64.b64encode((FIG/(s.key+'.svg')).read_bytes()).decode()
        sections.append(f'<section class="slide{ " active" if i==0 else ""}"><img alt="{html.escape(s.title)}" src="data:image/svg+xml;base64,{encoded}"></section>')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>轻量Actor与专家初始化</title><style>html,body{margin:0;background:#dfe6ed;height:100%;font-family:sans-serif}.slide{display:none;align-items:center;justify-content:center;height:100vh}.slide.active{display:flex}.slide img{width:min(100vw,177.777vh);height:auto;max-height:100vh;box-shadow:0 10px 40px #9aabb9}nav{position:fixed;right:14px;bottom:10px;background:#1f3046;color:white;padding:8px 16px;border-radius:5px;font-size:12px}@media print{.slide,.slide.active{display:flex;height:auto;break-after:page}.slide img{width:100%;box-shadow:none}nav{display:none}@page{size:16in 9in;margin:0}}</style>'''+''.join(sections)+'''<nav>← → 翻页 · F 全屏</nav><script>let n=0;const s=[...document.querySelectorAll('.slide')];function show(i){s[n].classList.remove('active');n=(i+s.length)%s.length;s[n].classList.add('active')}document.addEventListener('keydown',e=>{if(['ArrowRight',' ','PageDown'].includes(e.key))show(n+1);if(['ArrowLeft','PageUp'].includes(e.key))show(n-1);if(e.key.toLowerCase()==='f')document.documentElement.requestFullscreen()});</script></html>'''
    (OUT/'MLP_MLPBC_方法讲解_20261113.html').write_text(page)
    audit=dict(passed=True,slides=12,source_hashes=SOURCE_HASHES,source_hashes_unchanged=all(sha(ROOT/p)==d for p,d in SOURCE_HASHES.items()),
        figures='SVG / PDF vector, PNG 2880x1620',native_editable_ppt_shapes=True,ppt_reopened=True,ppt_zip_valid=True,
        all_shapes_within_slide=True,all_rendered_text_within_canvas=True,font_glyph_coverage_verified=True,pdf_pages_rasterized_for_visual_review=True,
        ppt_rendered_by_office=False,training_steps=0,solver_calls=0,torch_imported=False,figure_script_sha256=sha(__file__))
    assert audit['source_hashes_unchanged']
    (OUT/'DELIVERY_AUDIT.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(slides=12,pptx=str(ppt),pdf=str(pdf),preview=str(OUT/'PREVIEW_ALL.png')),ensure_ascii=False),flush=True)

if __name__=='__main__':build()
