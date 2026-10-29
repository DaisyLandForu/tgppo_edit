from utils import init_params
import time
import pyscipopt as scip
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import copy

class Tree:
    """
    Container for a B&B search Tree
    """
    def __init__(self):
        self.node_feature = []
        self.mip_feature = []
        self.var_feature = [] # Branching Decision
        self.edge_index = []
        self.cand_features = []
        self.num_node = 0
        self.scipIdMapping = {}
        self.scip_id_list = []

    def update(self, curr_scip_id, parent_scip_id, curr_node_feature, curr_mip_feature, curr_var_feature, curr_cand_features, branch_status=None):
        # pdb.set_trace()
        if curr_scip_id in self.scip_id_list: 
            if branch_status == scip.SCIP_RESULT.BRANCHED:
                # replace
                # pdb.set_trace()
                id = self.scipIdMapping[str(curr_scip_id)]
                self.node_feature[id] = curr_node_feature
                self.mip_feature[id] = curr_mip_feature
                self.var_feature[id] = curr_var_feature
                self.cand_features[id] = curr_cand_features
                
        else:
            id = self.num_node
            self.num_node = self.num_node + 1
            self.node_feature.append(curr_node_feature)
            self.mip_feature.append(curr_mip_feature)
            self.var_feature.append(curr_var_feature)
            self.cand_features.append(curr_cand_features)

            self.scipIdMapping[str(curr_scip_id)] = id
            self.scip_id_list.append(curr_scip_id)
            
            if parent_scip_id is not None:
                parent_id = self.scipIdMapping[str(parent_scip_id)]
                self.edge_index.append([id, parent_id])
                self.edge_index.append([parent_id, id])
    
    def getTreeNodeFeature(self, scip_idx):
        index = self.scipIdMapping[str(scip_idx)]
        return self.node_feature[index], self.mip_feature[index], self.var_feature[index]

class Brancher(scip.Branchrule):
    """
    Base class for scip.Branchrule subclasses.
    Callback method branchexeclp is customized in each subclass.
    """
    def initialize(self):
        pass

    def branchinit(self):
        pass

class MambaEvalBrancher(Brancher):
    """
    Brancher using trained llm policy and SeqEvalEnv.
    Evaluation mode is deterministic.
    """
    def __init__(
        self, model, device, policy,
        llm, state_dims, verbose, max_seq_len, is_init, relpscost_times):
        super(MambaEvalBrancher, self).__init__()

        self.model = model
        self.device = device
        self.policy = policy.to(device)
        self.llm = llm
        self.var_dim = state_dims['var_dim']
        self.node_dim = state_dims['node_dim']
        self.mip_dim = state_dims['mip_dim']
        self.verbose = verbose

        self.branch_count = 0
        self.branchexec_count = 0
        self.episode_rewards = []

        self.seq_len = max_seq_len
        self.input_seq = deque(maxlen=self.seq_len)
        self.is_init = is_init
        self.relpscost_times = relpscost_times

    def choose(self, probs):
        if len(probs.size()) == 0:
            probs = probs.unsqueeze(0)
        confidence_score, branch_decision = probs.max(0)
        return confidence_score, branch_decision

    def branchexeclp(self, allowaddcons):

        self.branchexec_count += 1

        # get state representations
        cands, cands_pos, cands_state_mat = self.model.getCandsState(self.var_dim, self.branchexec_count)
        node_state = self.model.getNodeState(self.node_dim)
        mip_state = self.model.getMIPState(self.mip_dim)

        # torchify states
        cands_state_mat = torch.from_numpy(cands_state_mat.astype('float32')).to(self.device)
        node_state = torch.from_numpy(node_state.astype('float32')).to(self.device)
        mip_state = torch.from_numpy(mip_state.astype('float32')).to(self.device)

        # select action from the policy probs
        # probs = self.policy(cands_state_mat, node_state, mip_state)
        state = torch.cat((node_state, mip_state), dim=0).unsqueeze(0)
        cands_state_mat = cands_state_mat.unsqueeze(0)
        # probs = self.policy(cands_state_mat, state, False)
        
        # new_cans_state_mat.shape = (1, candidate_num, 8)
        new_cands_state_mat, _  = self.policy(
            cands_state_mat, state
        )

 
        current_candidate_num = new_cands_state_mat.shape[1]
        self.input_seq.append(new_cands_state_mat)

        # 把所有元素算一个max_candidate_num维度的向量
        input_seq_list = list(self.input_seq)

        # if self.is_init and len(input_seq_list) < self.seq_len:
        if self.is_init and self.branch_count < self.relpscost_times:
            result = self.model.executeBranchRule('relpscost', allowaddcons)
            if result == scip.SCIP_RESULT.BRANCHED:
                children = self.model.getOpenNodes()[1][0]
                # 仿照getBranchInfos函数
                domchg = children.getDomchg()
                boundchgs = domchg.getBoundchgs()
                nboundchgs = len(boundchgs)
                assert nboundchgs == 1

                chosen_variable = boundchgs[0].getVar()                
                
                # chosen_variable is a SCIP Variable object
                assert chosen_variable is not None
                assert chosen_variable.isInLP()

                action = cands_pos.index(chosen_variable.getCol().getLPPos())
                action = torch.tensor(
                    action, dtype = torch.long,
                    device=new_cands_state_mat.device
                )

                # 序列更新：加入action
                self.input_seq.append(new_cands_state_mat[:,action,:].unsqueeze(0))

                self.branch_count += 1

                if self.verbose:
                    print('\tBranch count: {}. Selected var: {}.'.format(
                        self.branch_count, cands_pos[action.item()]))

            # 如果状态不是branched，没有子节点产生
            else:
                self.input_seq.pop()  # 弹出最后一个元素
        else:
            # max_candidate_num = max([
            #     cands.shape[1] for cands in input_seq_list
            # ])
            candidates_num = [cands.shape[1] for cands in input_seq_list]

            # 长度为7，则4个位置存放states，3个位置存放action
            position_ids = torch.zeros(
                1, sum(candidates_num),
                dtype=torch.long, device=new_cands_state_mat.device
            )
            
            # state和action共享一个cnt
            position_cnt = 0
            for i in range(len(input_seq_list)):
                start_pos = sum(candidates_num[:i])
                states_pos = slice(start_pos, start_pos + candidates_num[i])
                if i % 2 == 0:
                    position_ids[:, states_pos] = position_cnt
                else:
                    position_ids[:, states_pos] = position_cnt
                    position_cnt += 1

                
            input_embeds = torch.cat(input_seq_list, dim=1)

            with torch.no_grad():
                outputs_embeds = self.llm(
                    input_embeds,
                    position_ids=position_ids,
                )


            outputs_embeds = outputs_embeds[:, -current_candidate_num:, :]
            outputs_logits = outputs_embeds.mean(dim=-1).squeeze(0)  # shape: (max_candidates,)
            action = outputs_logits.argmax()

            # 序列更新：加入action
            self.input_seq.append(new_cands_state_mat[:,action,:].unsqueeze(0))

            # define the SCIP branch var
            var = cands[action.item()]
            # branch on the selected variable (SCIP Variable object)
            self.model.branchVar(var)
            self.branch_count += 1

            if self.verbose:
                print('\tBranch count: {}. Selected var: {}.'.format(
                    self.branch_count, cands_pos[action.item()]))

            result = scip.SCIP_RESULT.BRANCHED
            if result == scip.SCIP_RESULT.BRANCHED:
                try:
                    _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
                except:
                    # getOpenNodes, return leaves, children, siblings
                    children = self.model.getOpenNodes()[1][0]

                    # 仿照getBranchInfos函数
                    domchg = children.getDomchg()
                    boundchgs = domchg.getBoundchgs()
                    nboundchgs = len(boundchgs)
                    assert nboundchgs == 1

                    chosen_variable = boundchgs[0].getVar()
                    # _, chosen_variable, *_ = .getBranchInfos()
                    
                assert chosen_variable is not None
                assert chosen_variable.isInLP()
            
            
            
        # probs = probs.squeeze()
        # confidence_score, action = self.choose(probs)  # the chosen variable

        
        return {'result': result}

    def finalize(self):
        pass

    def finalize_zero_branch(self):
        pass


class SeqEvalBrancher(Brancher):
    """
    Brancher using trained llm policy and SeqEvalEnv.
    Evaluation mode is deterministic.
    """
    def __init__(self, model, device, policy, llm, state_dims, verbose):
        super(SeqEvalBrancher, self).__init__()

        self.model = model
        self.device = device
        self.policy = policy.to(device)
        self.llm = llm
        self.var_dim = state_dims['var_dim']
        self.node_dim = state_dims['node_dim']
        self.mip_dim = state_dims['mip_dim']
        self.verbose = verbose

        self.branch_count = 0
        self.branchexec_count = 0
        self.episode_rewards = []

        self.input_seq = None
        self.seq_len = 11

    def choose(self, probs):
        if len(probs.size()) == 0:
            probs = probs.unsqueeze(0)
        confidence_score, branch_decision = probs.max(0)
        return confidence_score, branch_decision

    def branchexeclp(self, allowaddcons):

        self.branchexec_count += 1

        # get state representations
        cands, cands_pos, cands_state_mat = self.model.getCandsState(self.var_dim, self.branchexec_count)
        node_state = self.model.getNodeState(self.node_dim)
        mip_state = self.model.getMIPState(self.mip_dim)

        # torchify states
        cands_state_mat = torch.from_numpy(cands_state_mat.astype('float32')).to(self.device)
        node_state = torch.from_numpy(node_state.astype('float32')).to(self.device)
        mip_state = torch.from_numpy(mip_state.astype('float32')).to(self.device)

        # select action from the policy probs
        # probs = self.policy(cands_state_mat, node_state, mip_state)
        state = torch.cat((node_state, mip_state), dim=0).unsqueeze(0)
        cands_state_mat = cands_state_mat.unsqueeze(0)
        # probs = self.policy(cands_state_mat, state, False)
        
        # new_cans_state_mat.shape = (1, candidate_num, 8)
        new_cands_state_mat, _  = self.policy(
            cands_state_mat, state
        )

        # proj_tree_features.shape = (1,8)
        proj_tree_features = new_cands_state_mat.max(dim=-2)[0]

        if self.input_seq is None:
            self.input_seq = proj_tree_features.unsqueeze(0)     # (1, 1, 896)
        else:
            self.input_seq = torch.cat(
                [
                    self.input_seq[:,-(self.seq_len - 1):,:],
                    proj_tree_features.unsqueeze(0),        # (1, seq_len, 896)
                ], dim=1
            )

        position_ids = torch.arange(
            self.input_seq.shape[1],
            dtype=torch.long,
            device=self.input_seq.device
        ).unsqueeze(0)  # (1, seq_len)
        position_ids = (position_ids // 2).long()

        output = self.llm(
            inputs_embeds=self.input_seq[:,-self.seq_len:,:],
            output_hidden_states=True,
            position_ids=position_ids,
        )

        outputs_embeds = output.hidden_states[-1]

        # pre_action_embedding.shape = [8]
        pre_action_embedding = outputs_embeds[0, -1, :]
        
        # new_cands_state_mat.shape = (1, candidate_num, 8)
        similarity = F.cosine_similarity(
            pre_action_embedding, new_cands_state_mat,
            dim = -1
        )

        _, action = torch.max(similarity, dim=1)

        # 序列更新：加入action
        self.input_seq = torch.cat(
            [
                self.input_seq[:,-(self.seq_len - 1):,:],
                new_cands_state_mat[0, action,:].unsqueeze(0),
            ], dim=1
        )

        # probs = probs.squeeze()
        # confidence_score, action = self.choose(probs)  # the chosen variable

        # define the SCIP branch var
        var = cands[action.item()]
        # branch on the selected variable (SCIP Variable object)
        self.model.branchVar(var)
        self.branch_count += 1

        if self.verbose:
            print('\tBranch count: {}. Selected var: {}.'.format(
                   self.branch_count, cands_pos[action.item()]))

        result = scip.SCIP_RESULT.BRANCHED
        if result == scip.SCIP_RESULT.BRANCHED:
            try:
                _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
            except:
                # getOpenNodes, return leaves, children, siblings
                children = self.model.getOpenNodes()[1][0]

                # 仿照getBranchInfos函数
                domchg = children.getDomchg()
                boundchgs = domchg.getBoundchgs()
                nboundchgs = len(boundchgs)
                assert nboundchgs == 1

                chosen_variable = boundchgs[0].getVar()
                # _, chosen_variable, *_ = .getBranchInfos()
                
            assert chosen_variable is not None
            assert chosen_variable.isInLP()

        return {'result': result}

    def finalize(self):
        pass

    def finalize_zero_branch(self):
        pass

class ILEvalBrancher(Brancher):
    """
    Brancher using trained Imitation Learning policy and ILEvalEnv.
    Evaluation mode is deterministic.
    """

    def __init__(self, model, device, policy, state_dims, verbose, is_init=False, relpscost_times=0):
        super(ILEvalBrancher, self).__init__()

        self.model = model
        self.device = device
        self.policy = policy.to(device)
        self.var_dim = state_dims['var_dim']
        self.node_dim = state_dims['node_dim']
        self.mip_dim = state_dims['mip_dim']
        self.verbose = verbose

        self.branch_count = 0
        self.branchexec_count = 0
        self.episode_rewards = []

        self.is_init = is_init
        self.relpscost_times = relpscost_times

    def choose(self, probs):
        if len(probs.size()) == 0:
            probs = probs.unsqueeze(0)
        confidence_score, branch_decision = probs.max(0)
        return confidence_score, branch_decision

    def branchexeclp(self, allowaddcons):

        self.branchexec_count += 1

        if self.is_init and self.branch_count < self.relpscost_times:
            result = self.model.executeBranchRule('relpscost', allowaddcons)
            if result == scip.SCIP_RESULT.BRANCHED:
                self.branch_count += 1
                
                try:
                    _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
                except:
                    # getOpenNodes, return leaves, children, siblings
                    children = self.model.getOpenNodes()[1][0]

                    # 仿照getBranchInfos函数
                    domchg = children.getDomchg()
                    boundchgs = domchg.getBoundchgs()
                    nboundchgs = len(boundchgs)
                    assert nboundchgs == 1

                    chosen_variable = boundchgs[0].getVar()                
                
                # chosen_variable is a SCIP Variable object
                assert chosen_variable is not None
                assert chosen_variable.isInLP()

            return {'result': result} 

        # get state representations
        cands, cands_pos, cands_state_mat = self.model.getCandsState(self.var_dim, self.branchexec_count)
        node_state = self.model.getNodeState(self.node_dim)
        mip_state = self.model.getMIPState(self.mip_dim)

        # torchify states
        cands_state_mat = torch.from_numpy(cands_state_mat.astype('float32')).to(self.device)
        node_state = torch.from_numpy(node_state.astype('float32')).to(self.device)
        mip_state = torch.from_numpy(mip_state.astype('float32')).to(self.device)

        # select action from the policy probs
        # probs = self.policy(cands_state_mat, node_state, mip_state)
        state = torch.cat((node_state, mip_state), dim=0).unsqueeze(0)
        cands_state_mat = cands_state_mat.unsqueeze(0)
        probs = self.policy(cands_state_mat, state, False)

        
        probs = probs.squeeze()
        confidence_score, action = self.choose(probs)  # the chosen variable

        # define the SCIP branch var
        var = cands[action.item()]
        # branch on the selected variable (SCIP Variable object)
        self.model.branchVar(var)
        self.branch_count += 1

        if self.verbose:
            print('\tBranch count: {}. Selected var: {}.'.format(
                   self.branch_count, cands_pos[action.item()]))

        result = scip.SCIP_RESULT.BRANCHED
        if result == scip.SCIP_RESULT.BRANCHED:
            try:
                _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
            except:
                # getOpenNodes, return leaves, children, siblings
                children = self.model.getOpenNodes()[1][0]

                # 仿照getBranchInfos函数
                domchg = children.getDomchg()
                boundchgs = domchg.getBoundchgs()
                nboundchgs = len(boundchgs)
                assert nboundchgs == 1

                chosen_variable = boundchgs[0].getVar()
                # _, chosen_variable, *_ = .getBranchInfos()
                
            assert chosen_variable is not None
            assert chosen_variable.isInLP()

        return {'result': result}

    def finalize(self):
        pass

    def finalize_zero_branch(self):
        pass



class SCIPCollectBrancher(Brancher):
    """
    Brancher to run SCIP data collection for imitation learning, with SCIPCollectEnv class.
    Instead of a single policy, 'explorer' and 'expert' rules are specified
    (each should be a string corresponding to a SCIP branching rule).
    The explorer policy runs for the top k branching decisions, then the expert takes over.
    Data is collected from expert decisions only.
    """
    def __init__(self, model, explorer, expert, k, state_dims, verbose):
        super(SCIPCollectBrancher, self).__init__()

        self.model = model
        self.explorer = explorer
        self.expert = expert
        self.k = k
        self.var_dim = state_dims['var_dim']
        self.node_dim = state_dims['node_dim']
        self.mip_dim = state_dims['mip_dim']
        self.verbose = verbose

        # counters and data structures
        self.branchexec_count = 0
        self.branch_count = 0
        self.explore = True
        self.explorer_count = 0
        self.collect_count = 0  # data collect counter
        self.collect_dict = OrderedDict()  # data dictionary to be filled with states and labels
        self.nnodes_list = []
        self.nnodesleft_list = []
        
        # self.searchTree = Tree()

    def branchexeclp(self, allowaddcons):

        # determine whether explorer or expert should be run
        if self.branch_count < self.k:
            self.explore = True
        else:
            self.explore = False

        # get state representations
        cands, cands_pos, cands_state_mat = self.model.getCandsState(self.var_dim, self.branchexec_count)
        cands_state_mat.astype('float32')
        node_state = self.model.getNodeState(self.node_dim).astype('float32')
        mip_state = self.model.getMIPState(self.mip_dim).astype('float32')
        
        if self.explore:
            # branch with explorer
            assert isinstance(self.explorer, str)
            self.branchexec_count += 1
            self.nnodes_list.append(self.model.getNNodes())
            self.nnodesleft_list.append(self.model.getNNodesLeft())
            result = self.model.executeBranchRule(self.explorer, allowaddcons)
            if result == scip.SCIP_RESULT.BRANCHED:
                self.explorer_count += 1
                self.branch_count += 1
                
                # # getOpenNodes, return leaves, children, siblings
                # children = self.model.getOpenNodes()[1][0]

                # # 仿照getBranchInfos函数
                # domchg = children.getDomchg()
                # boundchgs = domchg.getBoundchgs()
                # nboundchgs = len(boundchgs)
                # assert nboundchgs == 1

                # chosen_variable = boundchgs[0].getVar()
                # assert chosen_variable is not None
                # assert chosen_variable.isInLP()
                
                # branch_cand_state = cands_state_mat[cands_pos.index(chosen_variable.getCol().getLPPos()), :]
                
                # curr_scip_id = self.model.getCurrentNode().getNumber()
                # parent_scip_id = None if self.model.getNNodes() == 1 else self.model.getCurrentNode().getParent().getNumber()

                # self.searchTree.update(curr_scip_id, parent_scip_id, node_state, mip_state, branch_cand_state, cands_state_mat, result)

                
                if self.verbose:
                    # print(curr_scip_id, parent_scip_id)
                    print('\tExplore count: {} (exec. {}).'.format(self.explorer_count, self.branchexec_count))
        else:
            # branch with expert
            assert isinstance(self.expert, str)
            self.branchexec_count += 1
            self.nnodes_list.append(self.model.getNNodes())
            self.nnodesleft_list.append(self.model.getNNodesLeft())
            result = self.model.executeBranchRule(self.expert, allowaddcons)
            if result == scip.SCIP_RESULT.BRANCHED:
                self.collect_count += 1
                self.branch_count += 1
                
                try:
                    _, chosen_variable, *_ = self.model.getChildren()[0].getBranchInfos()
                except:
                    # getOpenNodes, return leaves, children, siblings
                    children = self.model.getOpenNodes()[1][0]

                    # 仿照getBranchInfos函数
                    domchg = children.getDomchg()
                    boundchgs = domchg.getBoundchgs()
                    nboundchgs = len(boundchgs)
                    assert nboundchgs == 1

                    chosen_variable = boundchgs[0].getVar()                
                
                # chosen_variable is a SCIP Variable object
                assert chosen_variable is not None
                assert chosen_variable.isInLP()

                # branch_cand_state = cands_state_mat[cands_pos.index(chosen_variable.getCol().getLPPos()), :]

                # curr_scip_id = self.model.getCurrentNode().getNumber()
                # parent_scip_id = None if self.model.getNNodes() == 1 else self.model.getCurrentNode().getParent().getNumber()

                # if self.verbose:
                #     print(curr_scip_id, parent_scip_id)
            
            
                self.collect_dict[self.collect_count] = {
                    'cands_state_mat': cands_state_mat,
                    'mip_state': mip_state,
                    'node_state': node_state,
                    'tree_features': np.hstack([node_state, mip_state]),
                    'varLPpos': chosen_variable.getCol().getLPPos(),
                    'varRELpos': cands_pos.index(chosen_variable.getCol().getLPPos()),
                    # 'global_node_feature': copy.deepcopy(self.searchTree.node_feature),
                    # 'global_mip_feature': copy.deepcopy(self.searchTree.mip_feature),
                    # 'global_var_feature': copy.deepcopy(self.searchTree.var_feature),
                    # 'edge_index': copy.deepcopy(self.searchTree.edge_index)
                }
                
                # self.searchTree.update(curr_scip_id, parent_scip_id, node_state, mip_state, branch_cand_state, cands_state_mat, result)
                
                if self.verbose:
                    # if self.collect_count % 1000 == 0:
                    print('\tBranch count: {} (exec. {}). '
                        'Selected varLPpos: {}. '
                        'Selected varRELpos: {}. '
                        'Num cands: {}'.format(self.branch_count, self.branchexec_count,
                                                chosen_variable.getCol().getLPPos(),
                                                cands_pos.index(chosen_variable.getCol().getLPPos()),
                                                len(cands),
                                                ))
                    
                    # num_current_node = self.model.getCurrentNode().getNumber()
                    # num_total = self.model.getNNodes()
                    # print('branchext:{}, num_curr:{}, num_total:{}'.format(self.branchexec_count, num_current_node, num_total))
                    
        return {'result': result}

    def finalize(self):
        pass


class SCIPCollectEnv:
    """
    Environment to run SCIP data collection for imitation learning, with SCIPCollectBrancher class.
    Instead of a single policy, 'explorer' and 'expert' rules are specified
    (each should be a string corresponding to a SCIP branching rule).
    The explorer policy runs for the top k branching decisions, then the expert takes over.
    Data is collected from expert decisions only.
    """

    def __init__(self):
        pass

    def run_episode(self, instance, name, explorer, expert, k, state_dims,
                    scip_seed, cutoff_value, scip_limits, scip_params, verbose, brancher_name='SCIPCollectBrancher'):
        """
        :param instance: str, pathway to instance.mps.gz
        :param name: str, name of the instance (w/o extension)
        :param explorer: str, SCIP branching rule to be used as explorer
        :param expert: str, SCIP branching rule to be used as expert
        :param k: int, number of branching decision to be explored before data collection
        :param state_dims: dict, of state dimensionalities
        :param scip_seed: int, SCIP solver seed
        :param cutoff_value: float, cutoff
        :param scip_limits: dict, specifying SCIP parameter limits
        :param scip_params: dict, specifying SCIP parameter setting
        :param verbose: bool, verbosity
        :param brancher_name: str, name of the brancher to be defined
        :return:
            exp_dict: dict, containing basic statistics on the experiment (run)
            brancher.collect_dict: dict, of data (states, labels) collected by the expert
        """

        print("\nRunning data collection on instance {}".format(name))
        m = scip.Model()

        # set static solver setting (scip seed and cutoff are set below)
        init_params(m, scip_limits, scip_params)

        # set scip parameters as needed (wrt the current episode setting)
        m.setBoolParam('randomization/permutevars', True)
        m.setIntParam('randomization/permutationseed', scip_seed)  # SCIP default at 0

        m.readProblem(instance)

        if scip_params['cutoff']:
            assert cutoff_value is not None
            m.setObjlimit(cutoff_value)

        brancher = SCIPCollectBrancher(
            model=m,
            explorer=explorer,
            expert=expert,
            k=k,
            state_dims=state_dims,
            verbose=verbose
        )
        m.includeBranchrule(
            brancher,
            name=brancher_name,
            desc="bla",
            priority=999999,
            maxdepth=-1,
            maxbounddist=1
        )

        # optimize, i.e., perform the solve
        t0 = time.time()
        t0_process = time.process_time()
        m.optimize()
        t1_process = time.process_time()
        t1 = time.time()

        print("\tInstance {}. SCIP time: {} (wall-clock: {}). Nnodes: {}. FairNNodes: {}. Collected: {}".format(
            name, m.getSolvingTime(), t1 - t0, m.getNNodes(),
            m.getFairNNodes(bytes(brancher_name, 'utf-8')), brancher.collect_count
        ))

        # store episode_data
        exp_dict = {
            'name': name,
            'explorer': explorer,
            'expert': expert,
            'k': k,
            'seed': scip_seed,
            'nnodes': m.getNNodes(),
            'fair_nnodes': m.getFairNNodes(bytes(brancher_name, 'utf-8')),  # needs bytes encoding
            'nnodes_left': m.getNNodesLeft(),
            'nLP_iterations': m.getNLPIterations(),
            'max_depth': m.getMaxDepth(),
            'status': m.getStatus(),
            'gap': m.getGap(),
            'primal_bound': m.getPrimalbound(),
            'dual_bound': m.getDualbound(),
            'primaldualintegral': m.getPrimalDualIntegral(),
            'scip_solve_time': m.getSolvingTime(),
            'scip_presolve_time': m.getPresolvingTime(),
            'opt_time_process': t1_process - t0_process,
            'opt_time_wallclock': t1 - t0,
            # 'nnodes_list': brancher.nnodes_list,
            # 'nnodesleft_list': brancher.nnodesleft_list,
        }

        m.freeProb()

        return exp_dict, brancher.collect_dict
    
class ILEvalEnv:
    """
    Environment to evaluate a trained Imitation Learning policy, using ILEvalBrancher.
    The specified branching policy is a trained IL policy.
    """
    def __init__(self, device, relpscost_times=0):
        self.device = device
        self.relpscost_times = relpscost_times

    def run_episode(self, instance, name, policy, policy_name, state_dims,
                    scip_seed, cutoff_value, scip_limits, scip_params, verbose,
                    is_init = False,
                    brancher_name='ILEvalBrancher'):
        """
        :param instance: str, pathway to instance.mps.gz
        :param name: str, name of the instance (w/o extension)
        :param policy: a trained IL policy
        :param policy_name: str, name of the policy
        :param state_dims: dict, of state dimensionalities
        :param scip_seed: int, SCIP solver seed
        :param cutoff_value: float, cutoff
        :param scip_limits: dict, specifying SCIP parameter limits
        :param scip_params: dict, specifying SCIP parameter setting
        :param verbose: bool, verbosity
        :param brancher_name: str, name of the brancher to be defined
        :return:
            exp_dict: dict, containing basic statistics on the experiment (run)
        """

        print("\nRunning IL evaluation on instance {}".format(name))
        m = scip.Model()

        # set static solver setting (scip seed and cutoff are set below)
        init_params(m, scip_limits, scip_params)

        # set scip parameters as needed (wrt the current episode setting)
        m.setBoolParam('randomization/permutevars', True)
        m.setIntParam('randomization/permutationseed', scip_seed)  # SCIP default at 0

        m.readProblem(instance)

        if scip_params['cutoff']:
            assert cutoff_value is not None
            m.setObjlimit(cutoff_value)

        # define brancher
        brancher = ILEvalBrancher(
            model=m,
            device=self.device,
            policy=policy,
            state_dims=state_dims,
            verbose=verbose,
            is_init = is_init,
            relpscost_times=self.relpscost_times
        )
        m.includeBranchrule(
            brancher,
            name=brancher_name,
            desc="bla",
            priority=999999,
            maxdepth=-1,
            maxbounddist=1
        )

        # perform the episode
        try:
            t0 = time.time()
            t0_process = time.process_time()
            m.optimize()
            t1_process = time.process_time()
            t1 = time.time()
            print("\tInstance: {}. Nnodes: {}. Branch count: {}. Status: {}. Gap: {:.4f}".format(
                name,
                m.getNNodes(),
                brancher.branch_count,
                m.getStatus(),
                m.getGap())
            )
        except:
            print("\tSCIP exception or error.")
            t0 = time.time()
            t0_process = time.process_time()
            t1 = t0
            t1_process = t0_process

        # update exp_dict
        exp_dict = {
            'name': name,
            'policy': policy_name,
            'seed': scip_seed,
            'nnodes': m.getNNodes(),
            'fair_nnodes': m.getFairNNodes(bytes(brancher_name, 'utf-8')),  # needs bytes encoding
            'nnodes_left': m.getNNodesLeft(),
            'nLP_iterations': m.getNLPIterations(),
            'max_depth': m.getMaxDepth(),
            'status': m.getStatus(),
            'gap': m.getGap(),
            'primal_bound': m.getPrimalbound(),
            'dual_bound': m.getDualbound(),
            'primaldualintegral': m.getPrimalDualIntegral(),
            'scip_solve_time': m.getSolvingTime(),
            'scip_presolve_time': m.getPresolvingTime(),
            'opt_time_process': t1_process - t0_process,
            'opt_time_wallclock': t1 - t0,
        }

        m.freeProb()

        return exp_dict
    
    
class SeqEvalEnv:
    """
    Environment to evaluate a trained Imitation Learning policy, using ILEvalBrancher.
    The specified branching policy is a trained IL policy.
    """
    def __init__(self, device, max_seq_len=99, relpscost_times=0):
        self.device = device
        self.max_seq_len = max_seq_len
        self.relpscost_times = relpscost_times

    def run_episode(self, instance, name, policy, policy_name, llm, llm_name,
                    state_dims, scip_seed, cutoff_value, scip_limits,
                    scip_params, verbose, is_init = False,brancher_name='ILEvalBrancher'):
        """
        :param instance: str, pathway to instance.mps.gz
        :param name: str, name of the instance (w/o extension)
        :param policy: a trained IL policy
        :param policy_name: str, name of the policy
        :param llm: a trained LLM policy
        :param llm_name: str, name of the llm policy
        :param state_dims: dict, of state dimensionalities
        :param scip_seed: int, SCIP solver seed
        :param cutoff_value: float, cutoff
        :param scip_limits: dict, specifying SCIP parameter limits
        :param scip_params: dict, specifying SCIP parameter setting
        :param verbose: bool, verbosity
        :param brancher_name: str, name of the brancher to be defined
        :return:
            exp_dict: dict, containing basic statistics on the experiment (run)
        """

        print("\nRunning IL evaluation on instance {}".format(name))
        m = scip.Model()

        # set static solver setting (scip seed and cutoff are set below)
        init_params(m, scip_limits, scip_params)

        # set scip parameters as needed (wrt the current episode setting)
        m.setBoolParam('randomization/permutevars', True)
        m.setIntParam('randomization/permutationseed', scip_seed)  # SCIP default at 0

        # # 初始化时，更靠近strong
        # m.setRealParam("branching/relpscost/minreliable", 5.0)     # 至少 10 次 strong branching 才算可靠
        # m.setRealParam("branching/relpscost/maxreliable", 10.0)     # 最多 20 次 strong branching 也得可靠了
        # m.setIntParam('branching/relpscost/initcand', 200)          


        m.readProblem(instance)

        if scip_params['cutoff']:
            assert cutoff_value is not None
            m.setObjlimit(cutoff_value)

        # define brancher
        # brancher = SeqEvalBrancher(
        brancher = MambaEvalBrancher(
            model=m,
            device=self.device,
            policy=policy,
            llm=llm,
            state_dims=state_dims,
            verbose=verbose,
            max_seq_len=self.max_seq_len,
            is_init=is_init,
            relpscost_times=self.relpscost_times
        )
        m.includeBranchrule(
            brancher,
            name=brancher_name,
            desc="bla",
            priority=999999,
            maxdepth=-1,
            maxbounddist=1
        )

        # perform the episode
        try:
            t0 = time.time()
            t0_process = time.process_time()
            m.optimize()
            t1_process = time.process_time()
            t1 = time.time()
            print("\tInstance: {}. Nnodes: {}. Branch count: {}. Status: {}. Gap: {:.4f}".format(
                name,
                m.getNNodes(),
                brancher.branch_count,
                m.getStatus(),
                m.getGap())
            )
        except:
            print("\tSCIP exception or error.")
            t0 = time.time()
            t0_process = time.process_time()
            t1 = t0
            t1_process = t0_process

        # update exp_dict
        exp_dict = {
            'name': name,
            'policy': policy_name,
            'llm': llm_name, 
            'seed': scip_seed,
            'nnodes': m.getNNodes(),
            'fair_nnodes': m.getFairNNodes(bytes(brancher_name, 'utf-8')),  # needs bytes encoding
            'nnodes_left': m.getNNodesLeft(),
            'nLP_iterations': m.getNLPIterations(),
            'max_depth': m.getMaxDepth(),
            'status': m.getStatus(),
            'gap': m.getGap(),
            'primal_bound': m.getPrimalbound(),
            'dual_bound': m.getDualbound(),
            'primaldualintegral': m.getPrimalDualIntegral(),
            'scip_solve_time': m.getSolvingTime(),
            'scip_presolve_time': m.getPresolvingTime(),
            'opt_time_process': t1_process - t0_process,
            'opt_time_wallclock': t1 - t0,
        }

        m.freeProb()

        return exp_dict