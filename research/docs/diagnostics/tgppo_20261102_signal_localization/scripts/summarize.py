"""Describe measurements by checkpoint and instance; no model import."""
import csv
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

OUT = Path(__file__).resolve().parents[1]


def read(name):
    return list(csv.DictReader((OUT / name).open()))


def write(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def group(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def quant(values):
    values = [float(x) for x in values if x not in [None, '']]
    return dict(zip(['min', 'p50', 'p95', 'max'], map(float, np.quantile(values, [0, .5, .95, 1])))) if values else {}


def imedian(rows, field):
    return [float(np.median([float(r[field]) for r in rs])) for rs in group(rows, ['instance']).values()]


def main():
    protocol = json.loads((OUT / 'frozen_protocol.json').read_text())
    policies = read('policy_metrics.csv')
    layers = read('layer_metrics.csv')
    ranks = read('effective_ranks.csv')
    paired = read('paired_intervention.csv')
    grads = read('score_gradients.csv')
    precision = read('precision_checks.csv')
    beta = read('bimatching_probe.csv')
    invariants = read('invariance_checks.csv')
    perturbations = read('input_perturbation.csv')
    assert len(policies) == 1312 and len(paired) == 656 and len(precision) == 68
    assert len(invariants) == 136
    policy_summary = []
    for (model, ep, variant), rs in group(policies, ['model', 'episode', 'variant']).items():
        policy_summary.append(dict(model=model, episode=ep, variant=variant,
                                   states=len(rs), instances=len({r['instance'] for r in rs}),
                                   tv_uniform=quant(r['tv_uniform'] for r in rs),
                                   instance_median_tv=quant(imedian(rs, 'tv_uniform')),
                                   logit_spread=quant(r['logit_spread'] for r in rs),
                                   states_tv_ge_001=sum(float(r['tv_uniform']) >= .01 for r in rs),
                                   all_zero_relu_states=sum(r['ln2_relu_all_zero'] == 'True' for r in rs)))
    layer_summary = []
    for (model, ep, variant, layer), rs in group(layers, ['model', 'episode', 'variant', 'layer']).items():
        layer_summary.append(dict(model=model, episode=ep, variant=variant, layer=layer,
                                  centered_rms_p50=quant(r['centered_rms'] for r in rs)['p50'],
                                  relative_centered_rms_p50=quant(r['relative_centered_rms'] for r in rs)['p50'],
                                  instance_median_relative_rms_p50=float(np.median(imedian(rs, 'relative_centered_rms'))),
                                  exactly_constant_states=sum(r['exactly_constant'] == 'True' for r in rs)))
    write('layer_summary.csv', layer_summary)
    index = {(r['model'], r['episode'], r['variant'], r['observation'], r['layer']): r for r in layers}
    contractions = []
    for r in policies:
        key = (r['model'], r['episode'], r['variant'], r['observation'])
        for a, b in protocol['predeclared_screens']['contraction_comparisons']:
            first, second = index[key + (a,)], index[key + (b,)]
            den = float(first['relative_centered_rms'])
            num = float(second['relative_centered_rms'])
            if den:
                contractions.append(dict(model=r['model'], episode=r['episode'], variant=r['variant'],
                                         observation=r['observation'], instance=r['instance'], L=r['L'],
                                         from_layer=a, to_layer=b, relative_rms_ratio=num / den))
    write('contraction_by_state.csv', contractions)
    contraction_summary, contraction_instance = [], []
    for (model, ep, variant, a, b), rs in group(contractions, ['model', 'episode', 'variant', 'from_layer', 'to_layer']).items():
        medians = imedian(rs, 'relative_rms_ratio')
        count = sum(x <= .1 for x in medians)
        contraction_summary.append(dict(model=model, episode=ep, variant=variant, from_layer=a, to_layer=b,
                                        states=len(rs), instances=len(medians),
                                        instance_ratio_p50=float(np.median(medians)),
                                        instances_ratio_le_01=count, strong_contraction=count / len(medians) >= .8))
        for (instance,), ss in group(rs, ['instance']).items():
            contraction_instance.append(dict(model=model, episode=ep, variant=variant, from_layer=a, to_layer=b,
                                             instance=instance, median_relative_rms_ratio=float(np.median([float(x['relative_rms_ratio']) for x in ss]))))
    write('contraction_summary.csv', contraction_summary)
    write('contraction_by_instance.csv', contraction_instance)
    rank_summary = []
    for (model, ep, variant, layer), rs in group(ranks, ['model', 'episode', 'variant', 'layer']).items():
        rank_summary.append(dict(model=model, episode=ep, variant=variant, layer=layer,
                                 anchor_states=len(rs), effective_rank_p50=quant(r['effective_rank'] for r in rs)['p50'],
                                 pc1_energy_p50=quant(r['pc1_energy_fraction'] for r in rs).get('p50')))
    write('rank_summary.csv', rank_summary)
    gradient_summary = []
    for (model, ep, variant, kind, layer), rs in group(grads, ['model', 'episode', 'variant', 'kind', 'layer']).items():
        gradient_summary.append(dict(model=model, episode=ep, variant=variant, kind=kind, layer=layer,
                                     l2_p50=quant(r['l2_norm'] for r in rs)['p50'],
                                     l2_max=quant(r['l2_norm'] for r in rs)['max'],
                                     candidate_centered_l2_p50=quant(r['candidate_centered_l2'] for r in rs).get('p50')))
    write('gradient_summary.csv', gradient_summary)
    perturbation_summary = []
    for (model, ep, variant, layer), rs in group(perturbations, ['model', 'episode', 'variant', 'layer']).items():
        perturbation_summary.append(dict(model=model, episode=ep, variant=variant, layer=layer,
                                         centered_change_rms_p50=quant(r['centered_output_change_rms'] for r in rs)['p50'],
                                         relative_centered_change_p50=quant(r['relative_centered_output_change'] for r in rs).get('p50'),
                                         policy_tv_change_p50=quant(r['policy_tv_change'] for r in rs)['p50'],
                                         policy_tv_change_max=quant(r['policy_tv_change'] for r in rs)['max']))
    write('perturbation_summary.csv', perturbation_summary)
    comparison_summary = []
    for (model, ep), rs in group(paired, ['model', 'episode']).items():
        comparison_summary.append(dict(model=model, episode=ep,
                                       ln2_rms_ratio=quant(r['ln2_rms_ratio'] for r in rs),
                                       relu_rms_ratio=quant(r['relu_rms_ratio'] for r in rs),
                                       relu_pattern_change=quant(r['relu_pattern_change_fraction'] for r in rs),
                                       tv_original_modified=quant(r['tv_original_modified'] for r in rs),
                                       scale_outside_bounds=sum(not(.8 <= float(r['ln2_rms_ratio']) <= 1.25 and .8 <= float(r['relu_rms_ratio']) <= 1.25) for r in rs)))
    precision_summary = []
    for (model, ep), rs in group(precision, ['model', 'episode']).items():
        precision_summary.append(dict(model=model, episode=ep, float32_tv_uniform=quant(r['float32_tv_uniform'] for r in rs),
                                      float64_tv_uniform=quant(r['float64_tv_uniform'] for r in rs),
                                      float32_float64_tv=quant(r['float32_float64_tv'] for r in rs),
                                      float64_logit_spread=quant(r['float64_logit_spread'] for r in rs),
                                      float64_states_tv_ge_001=sum(float(r['float64_tv_uniform']) >= .01 for r in rs)))
    beta_summary = []
    for (model, ep), rs in group(beta, ['model', 'episode']).items():
        beta_summary.append(dict(model=model, episode=ep, beta_tv_uniform=quant(r['beta_tv_uniform'] for r in rs),
                                 beta_cv=quant(r['beta_coefficient_variation'] for r in rs),
                                 reconstruction_max_error=max(float(r['reconstruction_max_error']) for r in rs)))
    consistency = []
    original_contractions = [r for r in contraction_summary if r['variant'] == 'original']
    for a, b in protocol['predeclared_screens']['contraction_comparisons']:
        rr = [r for r in original_contractions if r['from_layer'] == a and r['to_layer'] == b]
        consistency.append(dict(from_layer=a, to_layer=b, strong_in_all_four_checkpoints=len(rr) == 4 and all(r['strong_contraction'] for r in rr),
                                checkpoint_instance_ratios=[r['instance_ratio_p50'] for r in rr],
                                checkpoint_counts=[r['instances_ratio_le_01'] for r in rr]))
    summary = dict(policy=policy_summary, intervention=comparison_summary, precision=precision_summary,
                   bimatching=beta_summary, cross_checkpoint_contractions=consistency,
                   invariance={name:max(float(r[name]) for r in invariants) for name in ['repeat_max_error', 'permutation_max_error', 'padding_max_error', 'padded_probability']},
                   no_visible_preference_recovered=not any(r['states_tv_ge_001'] for r in policy_summary if r['variant'] != 'original'),
                   scale_preservation_passed=not any(r['scale_outside_bounds'] for r in comparison_summary),
                   caveat='Scale/variance/rank/gradient measurements do not establish useful action ranking or solver benefit.')
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    write('policy_summary.csv', [{**{k:v for k,v in r.items() if not isinstance(v,dict)},
                                  'tv_p50':r['tv_uniform']['p50'], 'tv_max':r['tv_uniform']['max'],
                                  'logit_spread_p50':r['logit_spread']['p50']} for r in policy_summary])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
