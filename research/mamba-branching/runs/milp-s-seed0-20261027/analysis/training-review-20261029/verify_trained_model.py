from pathlib import Path
import json, sys, time, gc
ROOT = Path('/home/duweiyue25/tgppo-transfer/research/mamba-branching')
sys.path.insert(0, str(ROOT / 'scripts'))
from formal_common import load_run, read_json, sha, hardware, stamp
import torch
from fast_data import MemoryStore, MemoryData
from fast_ops import validate
from network import TreeGateNet, GroupedMamba
run, cfg, provenance = load_run(ROOT / 'runs/milp-s-seed0-20261027')
print('Frozen code/environment verified', flush=True)
selected = read_json(run / 'training/selected.json')
report = {'started_utc':stamp(), 'hardware':hardware(), 'checkpoint_checks':[], 'read_only_verification':True, 'formal_training_updates_performed':0, 'test_instances_used':False}
def finite_count(value):
    if isinstance(value, torch.Tensor):
        assert torch.isfinite(value).all(), 'Nonfinite checkpoint tensor'
        return value.numel()
    if isinstance(value, dict): return sum(finite_count(v) for v in value.values())
    if isinstance(value, (tuple,list)): return sum(finite_count(v) for v in value)
    return 0
for label, pointer_path in [('cl_best','training/cl/selected.json'),('cl_final','training/cl/latest.json'),('il_best_and_final','training/il/selected.json')]:
    pointer = read_json(run / pointer_path)
    path = run / pointer['checkpoint']
    assert sha(path) == pointer['sha256']
    state = torch.load(path, map_location='cpu')
    assert state['config_sha256'] == selected['config_sha256']
    assert state['dataset_sha256'] == selected['dataset_sha256']
    assert state['run_sha256'] == sha(run / 'run.json')
    assert state['fast_execution_sha256'] == selected['fast_execution_sha256']
    assert state['effective_il_microbatch'] == 8
    total = sum(finite_count(state[k]) for k in ['embedding','sequence','optimizer'])
    if label in ['cl_final','il_best_and_final']:
        assert state['progress']['completed'] and state['progress']['epoch'] == 50
        assert state['progress']['cursor'] == 0
        expected = 458400 if label=='cl_final' else 4900
        assert state['progress']['updates'] == expected
    report['checkpoint_checks'].append({'label':label,'checkpoint':pointer['checkpoint'],'sha256':pointer['sha256'],'all_model_optimizer_tensors_finite':True,'elements_checked':total,'progress':state['progress'],'validation':state.get('validation')})
    print('Checkpoint verified: '+label, flush=True)
    if label == 'il_best_and_final': trained=state
    else: del state
assert torch.cuda.is_available() and torch.cuda.device_count()==1
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
report['gpu']=torch.cuda.get_device_name()
embedding, sequence=TreeGateNet(infimum=8).cuda(), GroupedMamba(8,100).cuda()
embedding.load_state_dict(trained['embedding']); sequence.load_state_dict(trained['sequence'])
started=time.monotonic()
store=MemoryStore(run,lambda e:print(json.dumps(e),flush=True))
valid=MemoryData(store,'valid','il')
metric=validate('il',embedding,sequence,valid,8,2,lambda:False)
reference=trained['validation']
report['validation_recheck']={'metric':metric,'reference':reference,'seconds_with_preload':time.monotonic()-started,'loss_relative_error':abs(metric['loss']-reference['loss'])/max(1.,abs(reference['loss'])),'accuracy_absolute_error':abs(metric['action_accuracy']-reference['action_accuracy'])}
assert report['validation_recheck']['loss_relative_error'] < 1e-5
assert report['validation_recheck']['accuracy_absolute_error'] <= 1 / metric['states']
assert metric['states']==42250 and metric['examples']==475
report['passed']=True; report['finished_utc']=stamp()
Path('/tmp/mamba-trained-model-verification-20261029.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['validation_recheck'],indent=2),flush=True)
print('GPU validation recheck passed',flush=True)
