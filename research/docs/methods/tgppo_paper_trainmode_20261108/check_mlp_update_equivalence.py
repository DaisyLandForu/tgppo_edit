"""Bounded CPU equivalence check on disposable copies; no solves or model saves."""
import ast
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import signal
import time

START = time.monotonic()
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT/'research/tgppo/experiments/paper-aligned-20261106'
NEW = ROOT/'research/tgppo/experiments/paper-trainmode-20261108'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA must be hidden')
    ledger_path = HERE/'PREPARATION_BUDGET.json'
    ledger = json.loads(ledger_path.read_text())
    remaining = 600-ledger['charged_seconds']
    if remaining < 5:
        raise RuntimeError('Cumulative local CPU check budget exhausted')
    attempt = dict(phase='mlp_update_equivalence',started_epoch=time.time(),allocated_seconds=remaining)
    ledger['attempts'].append(attempt)
    write(ledger_path,ledger)
    result = dict(passed=False,real_scip_solves=0,committed_training_episodes=0,
        production_model_writes=0,scope='two trained MLP conditions, four retained transitions each; CPU only')
    def stop(signum,frame):
        raise TimeoutError('Local equivalence deadline')
    signal.signal(signal.SIGALRM,stop)
    signal.setitimer(signal.ITIMER_REAL,max(1,remaining-(time.monotonic()-START)))
    try:
        import torch
        import paper
        from project import runtime as rt
        from engineering import same_state
        torch.set_num_threads(1)
        identities={}
        for p in (OLD,NEW):
            release=json.loads((p/'docs/release.json').read_text())
            changed=[f for f,h in release['files'].items() if sha(p/f)!=h]
            if changed:
                raise ValueError('Changed frozen source: '+str(changed))
            identities[p.name]=dict(files=len(release['files']),changed=changed)
        spec=importlib.util.spec_from_file_location('old_paper_agent',OLD/'code/project/agent/agent.py')
        legacy=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        # The legacy Agent imports the same storage class: verify its bytes.
        common_files=['project/memory/memory.py','project/policy/light_actor.py',
                      'project/policy/critic.py','project/policy/modules.py']
        assert all((OLD/'code'/p).read_bytes()==(NEW/'code'/p).read_bytes() for p in common_files)
        def methods(path):
            return {n.name:ast.dump(n,include_attributes=False) for n in ast.walk(ast.parse(path.read_text()))
                    if isinstance(n,ast.FunctionDef) and n.name in ('_compute_gae','compute_targets','seal_episode')}
        assert methods(OLD/'code/project/agent/agent.py')==methods(NEW/'code/project/agent/agent.py')
        comparisons=[]
        for variant in ('mlp','mlp_bc'):
            path=ROOT/'research/tgppo/runs/paper-aligned-20261106/training'/(variant+'-s22')/'pending.pt'
            raw=path.read_bytes()
            payload=torch.load(io.BytesIO(raw),map_location='cpu')
            source_sha=hashlib.sha256(raw).hexdigest()
            rows=payload['memory']['episodes'][0]['transitions']
            if len(rows)<5:
                raise ValueError('Five retained transitions required')
            outputs=[]
            for update_mode in ('old_eval','new_train'):
                cfg=paper.configuration(paper.protocol(),variant,22,'cpu')
                rt.seed_all(22)
                a=rt.build_agent(cfg)
                if update_mode=='old_eval':
                    a=legacy.Agent(a.actor_network,a.actor_optimizer,a.critic_network,a.critic_optimizer,
                        **{k:cfg[k] for k in rt.PPO_KEYS},device='cpu',state_dims=cfg['state_dims'])
                    a.attention_cell_budget=cfg['attention_cell_budget']
                a.actor_network.load_state_dict(payload['actor'])
                a.critic_network.load_state_dict(payload['critic'])
                a.actor_optimizer.load_state_dict(copy.deepcopy(payload['actor_optimizer']))
                a.critic_optimizer.load_state_dict(copy.deepcopy(payload['critic_optimizer']))
                a.policy_version=payload['policy_version']
                a.update_counter=payload['update_counter']
                assert not any(isinstance(m,(torch.nn.Dropout,torch.nn.modules.batchnorm._BatchNorm))
                               for m in a.actor_network.modules())
                a.memory.begin_episode('equivalence-only',a.policy_version)
                for row in rows[:4]:
                    a.remember(**{k:row[k] for k in ('cands_state','node_state','mip_state',
                        'action','value','log_prob','reward')},done=False)
                bootstrap={k:rows[4][k] for k in ('cands_state','node_state','mip_state')}
                a.seal_episode(False,collector_truncated=True,bootstrap_state=bootstrap,
                               end_reason='offline equivalence slice; not a real training episode')
                targets=copy.deepcopy(a.compute_targets())
                rt.seed_all(334455)
                metrics=a.learn(microbatch_size=8)
                outputs.append(dict(actor=copy.deepcopy(a.actor_network.state_dict()),
                    critic=copy.deepcopy(a.critic_network.state_dict()),
                    actor_optimizer=copy.deepcopy(a.actor_optimizer.state_dict()),
                    critic_optimizer=copy.deepcopy(a.critic_optimizer.state_dict()),rng=rt.rng_state(),
                    targets=targets,metrics=metrics,policy_version=a.policy_version,
                    update_counter=a.update_counter))
                del a
            first,second=outputs
            equality={k:same_state(first[k],second[k]) for k in ('actor','critic','actor_optimizer',
                'critic_optimizer','rng','targets','policy_version','update_counter')}
            equality['common_metrics']=all(first['metrics'][k]==second['metrics'][k]
                for k in ('actor_loss','critic_loss','entropy','clip_fraction','num_updates','num_samples'))
            comparisons.append(dict(variant=variant,training_seed=22,checkpoint=str(path),
                checkpoint_sha256=source_sha,source_episode_cursor=payload['progress']['next_episode'],
                slice_length=4,candidate_counts=[len(r['cands_state']) for r in rows[:4]],
                logical_updates_per_side=first['metrics']['num_updates'],equality=equality,
                max_actor_tensor_difference=max(float((first['actor'][k]-second['actor'][k]).abs().max())
                    for k in first['actor']),
                metrics_old={k:first['metrics'][k] for k in ('actor_loss','critic_loss','clip_fraction')},
                metrics_new={k:second['metrics'][k] for k in ('actor_loss','critic_loss','clip_fraction')}))
            del outputs,first,second,payload,raw
        result.update(passed=all(all(x['equality'].values()) for x in comparisons),
            cuda_initialized=torch.cuda.is_initialized(),frozen_sources=identities,
            standard_gae_methods_identical=True,comparisons=comparisons,
            limits='CPU, two checkpoints and small slices; not proof of cross-GPU or full-trajectory bitwise identity')
        assert not torch.cuda.is_initialized()
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        attempt.update(elapsed_seconds=time.monotonic()-START,passed=result['passed'])
        ledger['charged_seconds']=sum(a.get('elapsed_seconds',a['allocated_seconds']) for a in ledger['attempts'])
        write(ledger_path,ledger)
        result.update(elapsed_seconds=attempt['elapsed_seconds'],cumulative_local_check_seconds=ledger['charged_seconds'])
        write(HERE/'MLP_UPDATE_EQUIVALENCE.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
