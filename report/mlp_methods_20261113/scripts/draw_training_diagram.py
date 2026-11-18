#!/usr/bin/env python3
"""Draw the implemented MLP / MLP+BC training procedure as a paper figure.

Only draws graphics. No training environment, model, or solver is imported.
Dependencies belong to the separate presentation Python environment.
"""
import hashlib
import json
from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from fontTools import subset
from fontTools.ttLib import TTFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.util import Inches, Pt

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]
FIG = OUT / 'figures'
FIG.mkdir(exist_ok=True)
STEM = '06_mlp_mlpbc_training_mamba_style'
W, H = 22., 12.4
C = dict(ink='#202020', muted='#62696F', teal='#27AAA6', violet='#8A59B3',
         red='#D83737', blue='#DCE9F6', orange='#FBE4D4', purple='#E9E2F4',
         green='#E3EFD9', yellow='#FFF0C9', gray='#ECECEC', pale='#F8F8F8',
         white='#FFFFFF', line='#BFC7CE')
OPS = []


def text(x, y, value, size=16, color='ink', bold=False, ha='center',
         ppt=None, rotation=0, background=False):
    OPS.append(dict(kind='text', x=x, y=y, text=value, size=size,
                    color=C.get(color, color), bold=bold, ha=ha,
                    ppt=ppt, rotation=rotation, background=background))


def box(x, y, w, h, fill=None, edge=None, dash=False, radius=.09, lw=1.1):
    OPS.append(dict(kind='box', x=x, y=y, w=w, h=h,
                    fill=C.get(fill, fill), edge=C.get(edge, edge),
                    dash=dash, radius=radius, lw=lw))


def arrow(points, gradient=False, head=True, lw=1.4):
    OPS.append(dict(kind='arrow', points=points, gradient=gradient, head=head, lw=lw))


def dot(x, y, r=.035):
    OPS.append(dict(kind='dot', x=x, y=y, r=r))


def build():
    # Two stages, following the supplied reference's pastel blocks and dashed frames.
    box(.32, .30, 6.55, 8.55, edge='teal', dash=True, radius=.35)
    box(7.12, .30, 14.55, 8.55, edge='violet', dash=True, radius=.35)
    text(.55, .77, 'Step 1. 专家行为克隆', 21, ha='left')
    text(7.38, .77, 'Step 2. PPO 策略优化', 21, ha='left')
    text(14.50, 1.16, '完整 episode 后更新 3 轮 · AdamW', 12, color='muted')
    text(5.95, 1.16, '仅 MLP+BC', 12.5, color='teal', bold=True)
    text(20.72, 1.16, '两种方法共用', 12.5, color='violet')

    # BC: expert actions and predicted probabilities meet only at cross entropy.
    box(3.04, 1.55, 1.34, .70, fill='white', edge='violet', lw=1.8)
    text(3.71, 1.90, r'$\mathcal{L}_{\mathrm{BC}}$', 22, ppt='ℒ_BC')
    text(3.71, 2.55, 'Cross entropy', 15)
    box(.98, 3.03, 1.36, .70, fill='green')
    text(1.66, 3.38, r'$a^{E}$', 20, ppt='aᴱ')
    box(4.50, 3.03, 1.78, .70, fill='purple')
    text(5.39, 3.38, r'$\pi_{\theta}(\cdot\mid s^{E})$', 17,
         ppt='π_θ(· | sᴱ)')
    arrow([(1.66, 3.03), (1.66, 1.90), (3.04, 1.90)])
    arrow([(5.39, 3.03), (5.39, 1.90), (4.38, 1.90)])
    text(1.66, 4.00, '专家动作', 12.5)
    text(5.39, 4.00, 'Masked softmax', 12.5)
    box(1.30, 4.49, 5.00, 1.12, fill='gray')
    text(3.80, 4.86, r'共享 MLP Actor $\pi_{\theta}$', 18,
         ppt='共享 MLP Actor π_θ')
    text(3.80, 5.30, '双路编码 + 共享候选评分', 14)
    arrow([(5.39, 4.49), (5.39, 4.13)])
    arrow([(5.39, 3.90), (5.39, 3.73)])
    box(1.30, 6.65, 5.00, .80, fill='green')
    text(3.80, 7.05, r'专家数据 $\mathcal{D}_{E}=\{(s^{E},a^{E})\}$', 17,
         ppt='专家数据 D_E = {(sᴱ, aᴱ)}')
    text(3.80, 7.84, '既有 SCIP relpscost 轨迹', 15)
    text(3.80, 8.30, '只预训练 Actor，不预训练 Critic', 13, color='muted')
    arrow([(3.80, 6.65), (3.80, 5.61)])
    text(4.62, 6.14, r'$s^{E}=(X,c)$', 14, ppt='sᴱ = (X, c)')
    arrow([(1.30, 7.05), (.79, 7.05), (.79, 3.38), (.98, 3.38)])
    arrow([(3.04, 1.67), (.54, 1.67), (.54, 5.05), (1.30, 5.05)],
          gradient=True)

    # Alternative initialization: both routes terminate at exactly the same Actor.
    box(7.55, 3.52, 2.12, .66, fill='blue')
    text(8.61, 3.85, '随机初始化', 15)
    text(8.61, 3.15, 'MLP', 15, bold=True)
    arrow([(8.61, 4.18), (8.61, 5.05)], head=False)
    arrow([(6.30, 5.05), (10.20, 5.05)])
    dot(8.61, 5.05)
    text(7.50, 4.68, r'BC 权重 $\theta_{\mathrm{BC}}$', 12.5,
         ppt='BC 权重 θ_BC', background=True)

    # Online learner: disjoint policy and value networks, distinct gradients.
    box(10.20, 4.49, 3.90, 1.12, fill='blue')
    text(12.15, 4.86, r'MLP Actor $\pi_{\theta}$', 18, ppt='MLP Actor π_θ')
    text(12.15, 5.30, '同一轻量网络', 14)
    box(17.10, 4.49, 3.90, 1.12, fill='orange')
    text(19.05, 4.86, r'原 Critic $V_{\phi}$', 18, ppt='原 Critic V_φ')
    text(19.05, 5.30, '独立参数，保留原结构', 13)

    box(10.55, 3.03, 3.20, .65, fill='purple')
    text(12.15, 3.35, r'$\pi_{\theta}(\cdot\mid s_t)$', 18,
         ppt='π_θ(· | s_t)')
    box(17.50, 3.03, 3.10, .65, fill='yellow')
    text(19.05, 3.35, r'$V_{\phi}(s_t)$', 18, ppt='V_φ(s_t)')
    arrow([(12.15, 4.49), (12.15, 3.68)])
    arrow([(19.05, 4.49), (19.05, 3.68)])

    box(10.10, 1.55, 4.20, .70, fill='white', edge='violet', lw=1.8)
    text(12.20, 1.90, r'$\mathcal{L}_{\pi}$  PPO-Clip + 熵正则', 16,
         ppt='ℒ_π   PPO-Clip + 熵正则')
    box(17.00, 1.55, 4.10, .70, fill='white', edge='#D98C5C', lw=1.8)
    text(19.05, 1.90, r'$\mathcal{L}_{V}$  价值回归', 17, ppt='ℒ_V   价值回归')
    arrow([(12.15, 3.03), (12.15, 2.25)])
    arrow([(19.05, 3.03), (19.05, 2.25)])
    arrow([(10.10, 1.90), (9.95, 1.90), (9.95, 4.73), (10.20, 4.73)],
          gradient=True)
    arrow([(21.10, 1.90), (21.40, 1.90), (21.40, 5.05), (21.00, 5.05)],
          gradient=True)

    # Targets are computed from rollout values; no gradient through GAE.
    box(14.98, 3.03, 1.72, .76, fill='green')
    text(15.84, 3.27, 'GAE', 16)
    text(15.84, 3.57, '停止梯度', 11, color='muted')
    arrow([(15.40, 3.03), (15.40, 2.47), (14.00, 2.47), (14.00, 2.25)])
    arrow([(16.25, 3.03), (16.25, 2.47), (17.60, 2.47), (17.60, 2.25)])
    text(14.72, 2.16, r'$\widehat A_t$', 16, ppt='Â_t')
    text(16.81, 2.16, r'$\widehat R_t$', 16, ppt='R̂_t')

    box(10.20, 6.73, 10.80, .76, fill='green')
    text(15.60, 7.11,
         r'完整轨迹：$(s_t,\ a_t,\ r_t,\ \log\pi_{\mathrm{old}},\ V_{\mathrm{old}},\ d_t)$',
         16, ppt='完整轨迹：(s_t, a_t, r_t, log π_old, V_old, d_t)')
    arrow([(12.15, 6.73), (12.15, 5.61)])
    arrow([(19.05, 6.73), (19.05, 5.61)])
    text(12.57, 6.20, r'$s_t$', 15, ppt='s_t')
    text(19.48, 6.20, r'$s_t$', 15, ppt='s_t')
    arrow([(15.84, 6.73), (15.84, 3.79)])
    text(15.84, 5.90, r'$r_t,V_{\mathrm{old}},d_t$', 12.5,
         ppt='r_t, V_old, d_t', background=True)
    # Stored actions and old log probabilities are loss inputs, not Actor inputs.
    arrow([(14.55, 6.73), (14.55, 2.73), (13.30, 2.73), (13.30, 2.25)])
    text(14.55, 4.51, r'$a_t,\log\pi_{\mathrm{old}}$', 12,
         ppt='a_t, log π_old', rotation=90, background=True)

    box(8.00, 7.93, 13.00, .62, fill='gray')
    text(14.50, 8.24, 'SCIP B&B：当前 Actor / Critic 采样 + H3 奖励', 16)
    arrow([(15.84, 7.93), (15.84, 7.49)])
    arrow([(10.20, 5.44), (7.55, 5.44), (7.55, 8.24), (8.00, 8.24)])
    text(8.72, 6.36, '更新后采集\n下一 episode', 13, background=True)

    # Full shared Actor definition, separated from the optimization diagram.
    box(.32, 9.18, 21.35, 2.40, edge='line', dash=True, radius=.22)
    text(.62, 9.55, '共享 Actor 结构', 16, bold=True, ha='left')
    text(21.30, 9.55, 'MLP 与 MLP+BC 的网络完全相同 · 24,493 参数',
         13, ha='right', color='muted')
    box(.77, 10.06, 2.25, .50, fill='blue')
    box(.77, 10.80, 2.25, .50, fill='orange')
    text(1.895, 10.31, r'$X\in\mathbb{R}^{L\times25}$', 16, ppt='X ∈ ℝ^(L×25)')
    text(1.895, 11.05, r'$c\in\mathbb{R}^{61}$', 16, ppt='c ∈ ℝ^61')
    box(3.68, 10.06, 3.58, .50, fill='blue')
    box(3.68, 10.80, 3.58, .50, fill='orange')
    text(5.47, 10.31, r'$\phi$: LN + 25→64→64', 14.5, ppt='φ：LN + 25→64→64')
    text(5.47, 11.05, r'$\psi$: LN + 61→64→64', 14.5, ppt='ψ：LN + 61→64→64')
    arrow([(3.02, 10.31), (3.68, 10.31)])
    arrow([(3.02, 11.05), (3.68, 11.05)])
    box(8.65, 10.43, 2.30, .56, fill='green')
    text(9.80, 10.71, r'拼接 $[h_i;t]$', 16, ppt='拼接 [h_i; t]')
    arrow([(7.26, 10.31), (8.05, 10.31), (8.05, 10.58), (8.65, 10.58)])
    arrow([(7.26, 11.05), (8.05, 11.05), (8.05, 10.86), (8.65, 10.86)])
    text(7.93, 9.98, r'$h_i$', 14, ppt='h_i')
    text(7.93, 11.35, r'$t$ 广播', 12, ppt='t 广播')
    box(11.62, 10.43, 4.17, .56, fill='purple')
    text(13.705, 10.71, '共享评分 128→64→32→1', 15)
    arrow([(10.95, 10.71), (11.62, 10.71)])
    box(16.47, 10.43, 2.72, .56, fill='gray')
    text(17.83, 10.71, 'Masked softmax', 14)
    arrow([(15.79, 10.71), (16.47, 10.71)])
    text(20.43, 10.71, r'$\pi_{\theta}(a=i\mid s)$', 17,
         ppt='π_θ(a=i | s)')
    arrow([(19.19, 10.71), (19.74, 10.71)])
    text(21.30, 11.33, 'c = 节点8 + MIP53；编码器两层 GELU；评分头前两层 GELU。',
         11, color='muted', ha='right')

    arrow([(.65, 12.02), (1.58, 12.02)])
    text(1.83, 12.02, '数据 / 权重流', 13, ha='left')
    arrow([(5.0, 12.02), (5.93, 12.02)], gradient=True)
    text(6.18, 12.02, '梯度更新', 13, ha='left', color='red')
    text(21.35, 12.02, 'BC 只用于初始化；PPO 不再加 BC 损失；部署仅使用 Actor。',
         12.5, color='muted', ha='right')


def make_font():
    dest = OUT / 'assets/NotoSansSC-Training.ttf'
    chars = ''.join(op['text'] for op in OPS if op['kind'] == 'text')
    source = Path('/tmp/tgppo_figure_references_20261113/NotoSansSC-Regular.ttf')
    if source.exists():
        font = TTFont(source)
        options = subset.Options()
        options.layout_features = ['*']
        sub = subset.Subsetter(options=options)
        sub.populate(text=chars + 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')
        sub.subset(font)
        font.save(dest)
    if not dest.exists():
        raise FileNotFoundError('Need packaged training font or original Noto Sans SC source')
    cmap = TTFont(dest).getBestCmap()
    required = {c for c in chars if '\u4e00' <= c <= '\u9fff'}
    assert not required - set(map(chr, cmap)), 'Missing Chinese glyphs'
    return FontProperties(fname=str(dest))


def render(font):
    plt.rcParams.update({'svg.fonttype': 'path', 'pdf.fonttype': 42,
                         'mathtext.fontset': 'stix', 'axes.unicode_minus': False})
    fig = plt.figure(figsize=(W, H), facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    texts = []
    for op in OPS:
        kind = op['kind']
        if kind == 'box':
            ax.add_patch(FancyBboxPatch((op['x'], op['y']), op['w'], op['h'],
                boxstyle='round,pad=0,rounding_size={}'.format(op['radius']),
                facecolor=op['fill'] or 'none', edgecolor=op['edge'] or 'none',
                linewidth=op['lw'], linestyle=(0, (4, 3)) if op['dash'] else '-'))
        elif kind == 'text':
            prop = font.copy(); prop.set_weight('bold' if op['bold'] else 'normal')
            t = ax.text(op['x'], op['y'], op['text'], ha=op['ha'], va='center',
                fontproperties=prop, fontsize=op['size'], color=op['color'],
                rotation=op['rotation'], linespacing=1.5,
                bbox=dict(facecolor='white', edgecolor='none', pad=2) if op['background'] else None,
                zorder=8)
            texts.append(t)
        elif kind == 'dot':
            ax.add_patch(Circle((op['x'], op['y']), op['r'], facecolor=C['ink'], zorder=5))
        else:
            pts = op['points']; color = C['red'] if op['gradient'] else C['ink']
            ls = (0, (4, 2)) if op['gradient'] else '-'
            for i, (a, b) in enumerate(zip(pts, pts[1:])):
                ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>' if op['head'] and i == len(pts)-2 else '-',
                    mutation_scale=15, lw=op['lw'], color=color, linestyle=ls,
                    shrinkA=0, shrinkB=0, capstyle='butt', joinstyle='miter', zorder=4))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for t in texts:
        b = t.get_window_extent(renderer)
        assert bounds.contains(b.x0, b.y0) and bounds.contains(b.x1, b.y1), t.get_text()
    for ext in ('svg', 'pdf', 'png'):
        fig.savefig(FIG/(STEM+'.'+ext), dpi=200, facecolor='white')
    plt.close(fig)


def rgb(value):
    return RGBColor.from_string(value.lstrip('#'))


def pptx():
    p = Presentation(); p.slide_width=Inches(W); p.slide_height=Inches(H)
    slide = p.slides.add_slide(p.slide_layouts[6])
    for op in OPS:
        k = op['kind']
        if k == 'text':
            raw = op['ppt'] or op['text']
            assert '$' not in raw, 'Supply native editable PPT text for math: '+raw
            width = max(1., len(raw) * op['size']/72 * .70)
            if op['ha']=='center': left=op['x']-width/2
            elif op['ha']=='right':left=op['x']-width
            else:left=op['x']
            h = .78 if '\n' in raw else .60
            tb=slide.shapes.add_textbox(Inches(left), Inches(op['y']-h/2), Inches(width), Inches(h))
            if op['rotation']:tb.rotation=(360-op['rotation'])%360
            frame=tb.text_frame; frame.clear(); frame.word_wrap=False
            frame.margin_left=frame.margin_right=frame.margin_top=frame.margin_bottom=0
            frame.vertical_anchor=MSO_ANCHOR.MIDDLE
            if op['background']:
                tb.fill.solid(); tb.fill.fore_color.rgb=rgb(C['white']); tb.line.fill.background()
            for i,line in enumerate(raw.split('\n')):
                para=frame.paragraphs[0] if i==0 else frame.add_paragraph()
                para.alignment={'center':PP_ALIGN.CENTER,'right':PP_ALIGN.RIGHT,'left':PP_ALIGN.LEFT}[op['ha']]
                para.space_before=para.space_after=Pt(0)
                run=para.add_run();run.text=line;run.font.name='Noto Sans SC';run.font.size=Pt(op['size'])
                run.font.bold=op['bold'];run.font.color.rgb=rgb(op['color'])
        elif k in ('box','dot'):
            if k=='box': x,y,w,h=op['x'],op['y'],op['w'],op['h']; shape_type=MSO_SHAPE.ROUNDED_RECTANGLE
            else:x,y,w,h=op['x']-op['r'],op['y']-op['r'],2*op['r'],2*op['r'];shape_type=MSO_SHAPE.OVAL
            sh=slide.shapes.add_shape(shape_type, Inches(x), Inches(y), Inches(w), Inches(h))
            fill=C['ink'] if k=='dot' else op['fill'];edge=None if k=='dot' else op['edge']
            if fill:sh.fill.solid();sh.fill.fore_color.rgb=rgb(fill)
            else:sh.fill.background()
            if edge:
                sh.line.color.rgb=rgb(edge);sh.line.width=Pt(op['lw'])
                if op['dash']:
                    node=OxmlElement('a:prstDash');node.set('val','dash');sh.line._get_or_add_ln().append(node)
            else:sh.line.fill.background()
            if k=='box' and len(sh.adjustments):sh.adjustments[0]=min(.12,op['radius']/max(.01,min(w,h)))
        else:
            pts=op['points']
            for i,(a,b) in enumerate(zip(pts,pts[1:])):
                sh=slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,Inches(a[0]),Inches(a[1]),Inches(b[0]),Inches(b[1]))
                sh.line.color.rgb=rgb(C['red'] if op['gradient'] else C['ink']);sh.line.width=Pt(op['lw'])
                ln=sh.line._get_or_add_ln()
                if op['gradient']:
                    node=OxmlElement('a:prstDash');node.set('val','dash');ln.append(node)
                if op['head'] and i==len(pts)-2:
                    node=OxmlElement('a:tailEnd');node.set('type','triangle');ln.append(node)
    slide.notes_slide.notes_text_frame.text=(
        'Step 1只用于MLP+BC：既有relpscost标签交叉熵监督Actor。'
        'MLP从随机初始化直接进入Step 2；MLP+BC只迁移Actor权重，PPO使用新优化器。'
        'Actor、Critic不共享参数；GAE使用采样轨迹中的旧价值，目标停止梯度。'
        '每完整episode后3轮PPO更新；PPO不含BC损失；部署只用Actor。'
        '下方双路MLP结构严格依据light_actor.py绘制；无Mamba、CL或历史模块。')
    dest=FIG/(STEM+'.pptx');p.save(dest)
    check=Presentation(dest);assert len(check.slides)==1


def audit():
    paths=[ROOT/'research/tgppo/experiments/paper-aligned-20261106/code/project/policy/light_actor.py',
           ROOT/'research/tgppo/experiments/paper-aligned-20261106/code/pretrain.py',
           ROOT/'research/tgppo/experiments/paper-aligned-20261106/code/project/agent/agent.py']
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    record=dict(reference='User-supplied Mamba-Branching figure; visual grammar only; original re-drawing.',
        dimensions_inches=[W,H],source_hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
        actor_parameters=24493,branches=['MLP: random initialization -> PPO','MLP+BC: expert BC -> Actor initialization -> PPO'],
        no_contrastive_learning=True,no_history_or_mamba=True,no_bc_loss_during_ppo=True,
        separate_actor_critic=True,gae_targets_from_rollout_values=True,gae_targets_stop_gradient=True,
        native_editable_ppt_shapes=True,ppt_reopened=True,ppt_rendered_by_office=False,
        rendered_text_within_canvas=True,training_steps=0,solver_calls=0,
        artifacts={p.name:sha(p) for p in FIG.glob(STEM+'.*')})
    (FIG/(STEM+'_AUDIT.json')).write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')


if __name__ == '__main__':
    build()
    font=make_font()
    with warnings.catch_warnings():
        warnings.filterwarnings('error',message='Glyph.*missing from current font')
        render(font)
    pptx(); audit()
    print('Created original paper-style training diagram: PNG / SVG / PDF / native PPTX.')
