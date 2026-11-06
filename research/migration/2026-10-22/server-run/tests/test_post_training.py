import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

LAUNCH=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('post_training',LAUNCH/'post_training.py')
post=importlib.util.module_from_spec(spec);spec.loader.exec_module(post)


class PostTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.run=self.root/'runs';self.work=self.run/'post'
        self.work.mkdir(parents=True)
        self.c=dict(train=['a','b'],solver_seeds=[0,1,2],max_episodes=8,batch_episodes=4)

    def collection(self):
        order=post.server.schedule(11,self.c);batches={}
        for begin,count in [(0,5),(4,70)]:
            batches[begin]=[dict(job_id='e{:05d}-{}-s{}'.format(i,order[i],self.c['solver_seeds'][(i//2)%3]),
                valid=True,task_terminal=True,worker_cpu_complete=True,cuda_initialized=False,
                policy_version=0,num_transitions=count,worker_cpu_seconds=1.) for i in range(begin,begin+4)]
        metrics=[dict(episodes_seen=4,num_samples=20,updated=False,policy_version=0),
                 dict(episodes_seen=8,num_samples=300,updated=True,num_updates=6,policy_version=1)]
        return order,batches,metrics

    def audited(self):
        value=dict(passed=True,protocol_sha256=post.fingerprint(self.c),stack={'cpu_model':'Gold 6148'},
                   jobs=[dict(group=g,seed=s) for g,s in post.MATRIX])
        post.write_json(self.work/'training_audit.json',value)
        return value

    def test_small_batch_is_retained_until_the_next_update(self):
        order,batches,metrics=self.collection()
        result=post.check_collection_history(self.c,11,order,batches,metrics)
        self.assertEqual(result['episodes'],8)
        self.assertEqual(result['transitions'],300)
        self.assertEqual(result['pending_transitions'],0)
        self.assertEqual(result['complete_ppo_updates'],1)

    def test_dropped_or_duplicated_retained_samples_are_rejected(self):
        for n in [280,320]:
            order,batches,metrics=self.collection();metrics[-1]['num_samples']=n
            with self.assertRaisesRegex(ValueError,'lost or duplicated'):
                post.check_collection_history(self.c,11,order,batches,metrics)

    def test_cpu_mismatch_is_rejected_before_selection_runs(self):
        audited=self.audited()
        with patch.object(post.server,'preflight',return_value=dict(stack={'cpu_model':'E5'})):
            with self.assertRaisesRegex(ValueError,'CPU/software differs'):
                post.target_cpu(self.root,self.c,audited)

    def test_preflight_loads_six_actors_without_starting_any_solver(self):
        audited=self.audited()
        self.c['milestones']=[100,200,300,400,500]
        for row in audited['jobs']:
            out=post.server.group_dir(self.run,row['group'],row['seed']);out.mkdir(parents=True)
            for episode in self.c['milestones']:
                (out/('episode-{:04d}.pt'.format(episode))).write_bytes(b'fixture')
            row['checkpoints']={'500':{'sha256':post.sha256(out/'episode-0500.pt')}}
        info=dict(effective_cpu_count=8.,memory_limit_bytes=32*1024**3,stack=audited['stack'])
        with patch.object(post,'load_audit',return_value=audited), \
                patch.object(post,'target_cpu',return_value=info), \
                patch.object(post,'validate_dataset'),patch.object(post.server,'baseline_bundle'), \
                patch.object(post,'load_actor',return_value=(SimpleNamespace(),{'actor':{}})) as load, \
                patch.object(post.server,'select') as select,patch.object(post.server,'evaluate') as evaluate:
            post.preflight(self.root,self.c,self.run,self.work)
            self.assertEqual(load.call_count,6)
            select.assert_not_called();evaluate.assert_not_called()
        receipt=post.read(next((self.work/'preflights').glob('*.json')))
        self.assertTrue(receipt['passed']);self.assertEqual(receipt['solves_started'],0)

    def test_preflight_rejects_missing_cpu_quota_before_loading_models(self):
        audited=self.audited()
        info=dict(effective_cpu_count=4.,memory_limit_bytes=32*1024**3)
        with patch.object(post,'target_cpu',return_value=info),patch.object(post,'load_actor') as load:
            with self.assertRaisesRegex(ValueError,'8 CPU'):
                post.preflight(self.root,self.c,self.run,self.work)
            load.assert_not_called()

    def test_held_out_jobs_never_start_before_six_models_are_selected(self):
        self.audited()
        with patch.object(post,'target_cpu'),patch.object(post.server,'evaluate') as execute:
            with self.assertRaises(post.Pending):
                post.evaluate(self.root,self.c,self.run,self.work,'B0',11)
            execute.assert_not_called()
        self.assertFalse((self.work/'frozen_selection.json').exists())

    def test_changed_frozen_actor_is_rejected(self):
        self.audited()
        actor=self.root/'actor.pt';actor.write_bytes(b'original fixture actor')
        selection=self.root/'selection.json';selection.write_text('{}')
        rows=[dict(group=g,seed=s,selection_path='selection.json',selection_sha256=post.sha256(selection),
                   exports={kind:dict(path='actor.pt',sha256=post.sha256(actor)) for kind in ('best','final')}) for g,s in post.MATRIX]
        post.write_json(self.work/'frozen_selection.json',dict(protocol_sha256=post.fingerprint(self.c),
            training_audit_sha256=post.sha256(self.work/'training_audit.json'),models=rows))
        actor.write_bytes(b'changed fixture actor')
        with self.assertRaisesRegex(ValueError,'Frozen Actor changed'):
            post.freeze(self.root,self.c,self.run,self.work)

    def test_changed_development_cutoff_is_rejected(self):
        directory=self.root/'tgppo/datasets/train_instances';directory.mkdir(parents=True)
        instance=directory/'a.mps.gz';instance.write_bytes(b'fixture instance')
        original=dict(cutoff=5.,status='optimal',source='fixture source')
        audited=dict(datasets={'development':{'a':dict(sha256=post.sha256(instance),cutoff=original)}})
        post.write_json(directory.parent/'cutoffs.json',{'a':dict(original,cutoff=6.)})
        with self.assertRaisesRegex(ValueError,'Instance/cutoff changed'):
            post.validate_dataset(self.root,audited,'development')


if __name__=='__main__':unittest.main(verbosity=2)
