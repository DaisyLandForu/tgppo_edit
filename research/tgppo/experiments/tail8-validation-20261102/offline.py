"""Four full-Actor two-step updates, complete-buffer GAE, no SCIP optimize."""
from probe_utils import *


def main():
    protected=verify_inputs();results=[];critic_after={}
    monitor=pickle.loads((OUT.parent/'tgppo_20261102_signal_localization/observations.pkl').read_bytes())
    monitor_rows=[r for key,instance,r in monitor]
    manifests=list(csv.DictReader((REPLAY/'batch_manifest.csv').open()))
    for seed in [11,22]:
        payloads=[pickle.loads(p.read_bytes()) for p in sorted((SOURCE/f'B0-s{seed}/collection/e00100/attempt-000').glob('e*.pkl'))]
        for variant in ['original','tail8']:
            agent,cfg=make_initial(seed,variant);d=OUT/'offline'/f'{variant}-s{seed}';d.mkdir(parents=True)
            # Online pilot starts from a separate pristine initial checkpoint.
            rt.save_checkpoint(d/'initial.pt',agent,cfg,progress={'diagnostic_initial':True})
            for payload in payloads:agent.memory.import_dict(payload)
            rows=agent.memory.rows();raw,returns=agent.compute_targets()
            assert len(rows)==(3735 if seed==11 else 5902)
            normalized=(raw-raw.mean())/raw.std(unbiased=False).clamp_min(1e-8)
            chosen=[r for r in manifests if r['model']==f'B0-s{seed}'];ids=np.array([int(r['global_index']) for r in chosen])
            assert len(ids)==256
            selected=[rows[i] for i in ids];adv=normalized[ids].numpy()
            assert np.max(np.abs(adv-np.array([float(r['normalized_advantage']) for r in chosen])))<1e-6
            before=responses(agent.actor_network,selected,adv)
            assert before['tv_max']<1e-7
            rt.seed_all(7000+seed)
            update,grads=bounded_learn(agent,[ids,ids],d)
            after=responses(agent.actor_network,selected,adv);mon=responses(agent.actor_network,monitor_rows)
            exported=roundtrip(agent,cfg,d,selected)
            if variant=='original':critic_after[seed]={k:v.detach().clone() for k,v in agent.critic_network.state_dict().items()}
            else:assert all(torch.equal(v,critic_after[seed][k]) for k,v in agent.critic_network.state_dict().items())
            second=next(x for x in grads if x['network']=='actor' and x['step']==2)['modules']
            row=dict(variant=variant,source_seed=seed,actor_steps=2,critic_steps=2,
                surrogate_change=after['surrogate_loss']-before['surrogate_loss'],train=after,monitor=mon,
                second_actor_module_gradients=second,roundtrip=exported,update=update)
            results.append(row);save(OUT/'offline/summary.json',dict(complete=len(results)==4,results=results))
            print(json.dumps(row),flush=True)
            del agent,rows,raw,returns,normalized
    assert verify_inputs()==protected and not torch.cuda.is_initialized()
    save(OUT/'offline/verification.json',dict(status='passed',protected_files=protected,actor_steps=8,critic_steps=8,
        critic_updates_paired_exact=True,solver_tasks=0,cuda_initialized=False))


if __name__=='__main__':main()
