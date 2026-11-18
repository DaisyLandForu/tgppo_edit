import argparse
import ast
import copy
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
import types
import unittest
from unittest import mock

import resource_resume as entry
import recover
paper = recover.boot('trainmode')

# Run the actual frozen preflight function with synthetic hardware. Importing
# the entire solver/Torch dependency stack is unnecessary for a RAM gate test.
tree=ast.parse((paper.PACKAGE/'code/server.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preflight')
namespace=dict(os=os,socket=socket)
cuda=types.SimpleNamespace(is_available=lambda:True,get_device_name=lambda index:'mock V100',
    get_device_properties=lambda index:types.SimpleNamespace(total_memory=32*1024**3),
    get_device_capability=lambda index:(7,0),device_count=lambda:1)
namespace['torch']=types.SimpleNamespace(set_num_threads=lambda count:None,cuda=cuda)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(paper.PACKAGE/'code/server.py'),'exec'),namespace)
preflight=namespace['preflight']


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.c = paper.protocol()
        self.limits = dict(allowed_cpus=[0,1,2,3],effective_cpu_count=4.,memory_limit_bytes=48*1024**3)
        namespace['stack']=lambda root:dict(python_abi='3.8',torch_version='1.13.1+cu117',cpu_model='same CPU')
        namespace['resource_limits']=lambda:dict(self.limits)

    def test_original_rejects48_new_accepts48_without_changing_training_config(self):
        before = copy.deepcopy(self.c)
        with self.assertRaisesRegex(RuntimeError,'below frozen training resource request'):
            preflight(paper.ROOT,self.c,'cuda')
        info = entry.memory_preflight(preflight,48)(paper.ROOT,self.c,'cuda')
        self.assertEqual(self.c,before)
        self.assertEqual(info['device'],'cuda')
        self.assertEqual(info['stack']['cpu_model'],'same CPU')
        self.assertEqual(info['resource_override']['original_minimum_ram_gib'],64)
        self.assertEqual(info['resource_override']['requested_minimum_ram_gib'],48)
        self.assertEqual(info['resource_override']['worker_address_space_limit_bytes'],32*1024**3)
        self.assertNotIn('torch',sys.modules)

    def test_48_minimum_still_rejects32(self):
        self.limits['memory_limit_bytes']=32*1024**3
        with self.assertRaises(RuntimeError):
            entry.memory_preflight(preflight,48)(paper.ROOT,self.c,'cuda')

    def test_64_option_preserves_original_minimum(self):
        with self.assertRaises(RuntimeError):
            entry.memory_preflight(preflight,64)(paper.ROOT,self.c,'cuda')
        self.limits['memory_limit_bytes']=64*1024**3
        info=entry.memory_preflight(preflight,64)(paper.ROOT,self.c,'cuda')
        self.assertEqual(info['resource_override']['requested_minimum_ram_gib'],64)

    def test_cpu_and_software_checks_remain(self):
        self.limits['effective_cpu_count']=2
        with self.assertRaisesRegex(RuntimeError,'CPU quota'):
            entry.memory_preflight(preflight,48)(paper.ROOT,self.c,'cuda')
        self.limits['effective_cpu_count']=4
        with mock.patch.dict(namespace,stack=lambda root:dict(python_abi='3.9',torch_version='wrong')):
            with self.assertRaisesRegex(RuntimeError,'Frozen server environment'):
                entry.memory_preflight(preflight,48)(paper.ROOT,self.c,'cuda')

    def test_memory_monitor_keeps_peak_and_event_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'memory.current').write_text('1048576')
            (root/'memory.peak').write_text('3145728')
            (root/'memory.max').write_text(str(48*1024**3))
            (root/'memory.events').write_text('oom 0\noom_kill 0\n')
            sample=entry.memory_sample(root)
            self.assertEqual(sample['cgroup_peak_bytes'],3145728)
            self.assertEqual(sample['cgroup_events']['oom_kill'],0)
            destination=root/'output/memory.json'
            with mock.patch.object(entry,'memory_sample',return_value=sample):
                with entry.MemoryMonitor(destination,interval=60):pass
            report=json.loads(destination.read_text())
            self.assertEqual(report['samples'],2)
            self.assertEqual(report['observed_peak_bytes'],3145728)
            self.assertEqual(report['latest']['cgroup_limit_bytes'],48*1024**3)

    def test_gpu_alias_forwarded_as_cuda_and_resume_retained(self):
        args=argparse.Namespace(command='run',experiment='trainmode',variant='b0',seed=22,
            device='gpu',ram_gib=48,no_total_wall_limit=True)
        forwarded=entry.forward(args)
        self.assertEqual(forwarded[forwarded.index('--device')+1],'cuda')
        self.assertEqual(forwarded[forwarded.index('--seed')+1],'22')
        self.assertIn('--resume',forwarded)
        self.assertIn('--no-total-wall-limit',forwarded)
        self.assertNotIn('--ram-gib',forwarded)


if __name__=='__main__':
    unittest.main()
