from probe_utils import *
from project.reward.reward import RewardH3
from project.reward.diagnostic import DiagnosticRewardH3


class FakeModel:
    def __init__(self,state):self.state=state;self.calls={}
    def __getattr__(self,name):
        def value():
            self.calls[name]=self.calls.get(name,0)+1
            return self.state[name]
        return value


def main():
    tests=0
    for base_gap in [0.,.2,1e20]:
        a,b=RewardH3(),DiagnosticRewardH3()
        for reward in [a,b]:reward.reset(100,.2 if base_gap==.2 else base_gap,200,'timelimit',20)
        for i in range(5):
            data=dict(getNNodes=10+i*7,getSolvingTime=1.+i*3,getGap=.8/(i+1),
                getPrimalDualIntegral=10.+i*8,getPrimalbound=100.-i,getDualbound=20.+i,
                getStatus='timelimit')
            m,n=FakeModel(data),FakeModel(data)
            assert a.compute(m,i==4)==b.compute(n,i==4)
            assert m.calls==n.calls
            tests+=1
        assert len(b.records)==5
    print('PASS H3 exact-return and identical getter-count checks:',tests,flush=True)
    save(OUT/'preflight.json',dict(status='passed',h3_exact_reward_comparisons=tests,
        no_added_solver_getters=True,baseline_unchanged=True,protected_input_files=verify_inputs()))


if __name__=='__main__':main()
