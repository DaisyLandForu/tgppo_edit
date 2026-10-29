import ecole
import numpy as np

class ExploreThenStrongBranch:
    """
    This custom observation function class will randomly return either strong branching scores (expensive expert)
    or pseudocost scores (weak expert for exploration) when called at every node.
    """

    def __init__(self, expert_probability):
        self.expert_probability = expert_probability
        self.pseudocosts_function = ecole.observation.Pseudocosts()
        self.strong_branching_function = ecole.observation.StrongBranchingScores()

    def before_reset(self, model):
        """
        This function will be called at initialization of the environment (before dynamics are reset).
        """
        self.pseudocosts_function.before_reset(model)
        self.strong_branching_function.before_reset(model)

    def extract(self, model, done):
        """
        Should we return strong branching or pseudocost scores at time node?
        """
        probabilities = [1 - self.expert_probability, self.expert_probability]
        expert_chosen = bool(np.random.choice(np.arange(2), p=probabilities))
        if expert_chosen:
            return (self.strong_branching_function.extract(model, done), True)
        else:
            return (self.pseudocosts_function.extract(model, done), False)

class TreeObs():
    def __init__(
        self, cands_state_mat, node_state, mip_state,
        # variable_features
    ):
        self.cands_state = cands_state_mat
        self.node_state = node_state
        self.mip_state = mip_state
        # self.variable_features = variable_features
class TreeFeature:
    """
    Tree Feature的特征提取
    """
    def __init__(self):
        self.var_dim = 25
        self.node_dim = 8
        self.mip_dim = 53
        self.branchexec_count = 0

    def before_reset(self, model):
        """
        This function will be called 
        at initialization of the environment (before dynamics are reset).
        """
        pass

    def extract(self,model,done):

        self.branchexec_count += 1

        model = model.as_pyscipopt()
        cands, cands_pos, cands_state_mat = model.getCandsState(self.var_dim, self.branchexec_count)
        # cands, cands_pos, variable_state_mat = model.getVariableState(self.var_dim, self.branchexec_count)

        node_state = model.getNodeState(self.node_dim)
        mip_state = model.getMIPState(self.mip_dim)
        
        return (cands_state_mat, node_state, mip_state)
