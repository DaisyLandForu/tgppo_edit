from numpy import clip
import torch
from pathlib import Path
import argparse

from dataset_scip import get_data_loader_clip, get_data_loader_transformer
from network import TreeGateNet

from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
import os

from peft import LoraConfig, TaskType
from peft import get_peft_model, PeftModel

from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer,TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
import numpy as np
# import bitsandbytes as bnb

import pynvml

from network import OutputNet

from transformers import get_linear_schedule_with_warmup
import sys
from torch.nn import DataParallel

def has_nan_gradients(model):
    for param in model.parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            return True
    return False
def has_nan_parameters(model):
    for param in model.parameters():
        if torch.isnan(param).any():
            return True
        return False

def get_gpu_memory_utilization(gpu_id=0):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    utilization = (info.used / info.total) * 100
    pynvml.nvmlShutdown()
    return utilization

def setup_multigpu(model, args):
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU!")
        model = DataParallel(model)
    model.to(args.device)
    return model


def finetune_process_large(
    batch, clip_model: TreeGateNet, model: PeftModel, args,
    # output_tree_model: OutputNet, output_var_model: OutputNet
):
    # llm需要大改：要把所有的candidate mat加进来
    
    # # 判断是否会造成显存爆炸，限制candidates的数量
    if batch['candidate_features'].shape[2] > args.max_candidates_num:
        print(f"skip this batch because of too many candidates: {batch['candidate_features'].shape[2]}")
        return None

    # batch.to(args.device)
    # ori_states.shape = (batch_size, max_seq_length, 61)
    ori_states = batch['states'].to(args.device)

    # ori_actions.shape = (batch_size, max_seq_length,)
    strong_actions = batch['actions'].to(args.device)

    # lengths = batch['lengths'].to(args.device)
    # masks.shape = (batch_size, max_seq_length)
    masks = batch['masks'].to(args.device)

    # ori_candidates.shape = (batch_size, max_seq_length, max_candidates, 25)
    ori_candidates = batch['candidate_features'].to(args.device)
    # ori_candidates_mask.shape = (batch_size, max_seq_length, max_candidates)
    candidates_masks = batch['candidate_masks'].to(args.device)

    if ori_candidates.isnan().any() or ori_states.isnan().any() or strong_actions.isnan().any() or candidates_masks.isnan().any() or masks.isnan().any():
        return None
    
    
    # cands_state_mat.shape = (batch_size, max_seq_length, max_candidates, 8)
    cands_state_mat, _  = clip_model(
        ori_candidates, ori_states
    )
    
    states = cands_state_mat
    
    batch_size, max_seq_length, max_candidates, infinum = cands_state_mat.shape
    
    # if max_candidates > args.max_candidates_num:
    #     return None
    
    indices = strong_actions.unsqueeze(-1).unsqueeze(-1)  # shape=(batch_size, max_seq_length, 1, 1)
    indices = indices.expand(-1, -1, -1, infinum)  # shape=(batch_size, max_seq_length, 1, 8)

    # 使用gather收集
    # actions.shape = (batch_size, max_seq_length, 1, infinum)
    actions = torch.gather(
        cands_state_mat, 
        dim=2,  # 在max_candidates维度上收集
        index=indices.long()  # 确保索引是long类型
    )  # shape=(batch_size, max_seq_length, 1, 8)
    
        
    inputs = torch.zeros(batch_size, max_seq_length * (max_candidates + 1), infinum, device=states.device)
    new_masks = torch.ones(batch_size, max_seq_length * (max_candidates + 1), dtype=torch.bool, device=states.device)
    # output_masks对于每个batch都是一样的，都是把actions的位置mask掉
    output_masks = torch.zeros(max_seq_length * (max_candidates + 1), dtype=torch.bool, device=states.device)
    position_ids = torch.zeros(batch_size, max_seq_length * (max_candidates + 1), dtype=torch.long, device=states.device)
    for i in range(max_seq_length):
        # Calculate positions in the output sequence
        start_pos = i * (max_candidates + 1)
        states_pos = slice(start_pos, start_pos + max_candidates)
        action_pos = start_pos + max_candidates
        
        # Insert states and action
        inputs[:, states_pos, :] = states[:, i, :, :]
        inputs[:, action_pos, :] = actions[:, i, 0, :]
        
        # Set mask (1 for states positions)
        output_masks[states_pos] = True
        
        # Set position indices (same for each block of states+action)
        position_ids[:, states_pos] = i
        position_ids[:, action_pos] = i

        # candidates_masks.shape = (batch_size, max_seq_length, max_candidates)
        new_masks[:, states_pos] = candidates_masks[:, i, :]
        
        
    # 输入模型
    if args.seq_name == 'mamba' or args.seq_name == 'transformer':
        
        # 使用示例
        if has_nan_parameters(model) or has_nan_gradients(model):
            print("模型参数或梯度中包含NaN值！")
        
        outputs_embeds = model(inputs, position_ids)
    else:
        output = model(
            inputs_embeds=inputs, output_hidden_states=True,
            attention_mask=new_masks,     
            position_ids=position_ids,  # 关键：传入自定义位置编码
        )
    
        outputs_embeds = output.hidden_states[-1]   # shape: (batch_size, 2*seq_len-1, hidden_size)

    # outputs_embeds.shape = (batch_size , max_candidates*max_seq_length, infinum)
    outputs_embeds = outputs_embeds[:, output_masks, :]

    outputs_embeds = outputs_embeds.view(batch_size, max_seq_length, max_candidates, infinum)
    
    outputs_logits = outputs_embeds.mean(dim=-1)  # shape: (batch_size, max_seq_length, max_candidates)
    # strong_actions.shape = (batch_size, max_seq_length)
    # candidates_masks.shape = (batch_size, max_seq_length, max_candidates)
    # masks.shape = (batch_size, max_seq_length)
    # 这里使用mask，因为不能把candidates对应的那些东西mask掉，不然就跟strong_actions的shape对不上了
    # 而且那些位置有也不会有事儿，那些位置action里肯定不会有
    
    if args.data_type == 'large' or args.data_type == 'mid':
        # outputs_logits = torch.clamp(outputs_logits, min=-100, max=100)
        outputs_logits = torch.clamp(outputs_logits, min=-50, max=50)

    loss = F.cross_entropy(
        outputs_logits[masks], strong_actions[masks]
    )
    
    if args.is_multi_gpu:
        # 如果是多GPU，需要对loss求平均值
        if torch.cuda.device_count() > 1:
            loss = loss.mean()
            
    if loss.item() > 1000:
        print("")
            
    if loss.isnan().any():
        return None
    
    return loss


def real_clip_process(batch, clip_model: TreeGateNet, args):

    actions, state, padded_cands, masks = batch
    state = state.to(args.device)
    actions = actions.to(args.device)
    padded_cands = padded_cands.to(args.device)
    
    if state.isnan().any() or padded_cands.isnan().any():
        return None
    
    strong_action_indices = actions

    padded_cands, _  = clip_model(
        padded_cands, state
    )

    pre_action_feature = padded_cands.max(dim=-2)[0]
    
    strong_action_feature = padded_cands[torch.arange(padded_cands.shape[0]), strong_action_indices]
    weak_action_features = torch.stack([
        torch.cat([batch_cands[:idx], batch_cands[idx+1:]])
        for batch_cands, idx in zip(padded_cands, strong_action_indices)
    ])

    loss = 2 - F.cosine_similarity(pre_action_feature, strong_action_feature, dim=-1).mean() + \
        F.cosine_similarity(pre_action_feature.unsqueeze(1), weak_action_features, dim=-1).mean()


    return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='llm_branch')
    parser.add_argument('--instance_name',type=str, help='案例名称',default='miplib')
    parser.add_argument('--train_instance_size',nargs=2,type=int,default=[100,500], help='用于训练的instance的size')
    parser.add_argument('--seed',type=int, help='随机数种子',default=0)
    parser.add_argument('--gpu',type=int, help='使用哪块gpu',default=1)
    parser.add_argument('--batch_size',type=int, help='batch size',default=32)
    parser.add_argument('--clip_batch_size',type=int, help='clip batch size',default=32)
    parser.add_argument('--run_id',type=int, help='run id',default=11)
    parser.add_argument('--is_clip', action='store_true', help='Whether clip is trained or not')
    parser.add_argument('--max_seq_length',type=int, help='最大序列长度', default=100)
    parser.add_argument('--max_candidates_num',type=int, help='最大candidates数长度',default=20000)
    parser.add_argument('--num_epochs',type=int, help='训练epoch数',default=50)
    parser.add_argument('--clip_num_epochs',type=int, help='训练epoch数',default=50)
    parser.add_argument('--max_clip_samples',type=int, help='最大clip samples数', default=200000)
    parser.add_argument('--seq_name',type=str, help='seq model 名称',default='mamba')
    parser.add_argument('--lr',type=float, help='学习率',default=1e-3)
    parser.add_argument('--clip_lr',type=float, help='学习率',default=1e-4)
    parser.add_argument('--is_multi_gpu', action='store_true', help='是否使用多卡')
    parser.add_argument('--data_type',type=str, help='数据量',default='less')

    args = parser.parse_args()

    args.device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    
    """
    调试
    """

    writer = SummaryWriter(log_dir='logs/{}/{}'.format(args.instance_name, args.run_id))

    if args.is_clip:
        clip_train_loader, clip_valid_loader = get_data_loader_clip(args, args.data_type)
    
  
    clip_model = TreeGateNet(
        infimum = 8
    ).to(args.device)

    save_path = Path("models/{}".format(args.run_id))
    finetune_save_path = Path("sequence_models/{}".format(args.run_id))
    if not save_path.exists():
        save_path.mkdir(parents=True)
    if not finetune_save_path.exists():
        finetune_save_path.mkdir(parents=True)

    """是否对projection进行训练"""    
    if args.is_clip:
        clip_optimizer = torch.optim.Adam(clip_model.parameters(), lr=args.clip_lr)
        for epoch in range(args.clip_num_epochs):
            clip_model.train()
            loss_list = []
            for batch in clip_train_loader:
                loss = real_clip_process(batch, clip_model, args)
                if loss is None:
                    continue
                clip_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_model.parameters(), max_norm=1.0)
                clip_optimizer.step()
                loss_list.append(loss.item())

            clip_model.eval()
            valid_loss_list = []
            with torch.no_grad():
                for batch in clip_valid_loader:
                    valid_loss = real_clip_process(batch, clip_model, args)
                    if valid_loss is None:
                        continue
                    valid_loss_list.append(valid_loss.item())

            print("clip epoch: {}, train loss: {}, valid loss: {}".format(epoch, np.mean(loss_list), np.mean(valid_loss_list)))
            writer.add_scalars(
                "clip_loss", 
                {"train_loss": np.mean(loss_list), "valid_loss": np.mean(valid_loss_list)}, 
                epoch
            )
            torch.save(clip_model.state_dict(), "{}/model_{}.pth".format(save_path, epoch))
        
        # clip 训练完成，退出
        sys.exit(0)

    else:
        clip_model.load_state_dict(
            torch.load(
                "{}/model_{}.pth".format(save_path, args.clip_num_epochs - 1),
                map_location="cpu",
                weights_only=True
            )
        )
        clip_model.to(args.device)
        
    
    """
    训练Transformer
    """    
    train_loader, valid_loader = get_data_loader_transformer(args, args.data_type)

    
    if args.seq_name == 'gpt2' or args.seq_name == 'distilgpt2':
        model_name = "./pretrained_models/{}".format(args.seq_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 节约显存方案
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map = args.device  # 指定 GPU 编号
        )
    elif args.seq_name == 'transformer':
        from network import TransformerDecoder
        model = TransformerDecoder(
            embed_size=8, max_positions = args.max_seq_length,
        ).to(args.device)
    elif args.seq_name == 'mamba':
        from network import GroupedMamba
        model = GroupedMamba(
            d_model=8, max_seq_length=args.max_seq_length
        ).to(args.device)
        
    else:
        raise ValueError("seq_name not supported")

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(clip_model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    
    
    # 多卡
    if args.is_multi_gpu:
        model = setup_multigpu(model, args)
        clip_model = setup_multigpu(clip_model, args)
    
    for epoch in range(args.num_epochs):
        model.train()
        clip_model.train()
        loss_list = []
        for batch in train_loader:            
            # llm训练
            loss = finetune_process_large(
                batch, clip_model, model, args, 
                # output_tree_model, output_var_model
            )
            if loss is None:
                continue
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(clip_model.parameters()),
                max_norm=1.0
            )
            
            optimizer.step()
            loss_list.append(loss.item())

        
        # print("epoch: {}, train_loss: {} proj_loss: {}".format(epoch, loss, proj_loss))
        print("ecpch: {}, train_loss: {}".format(epoch, np.mean(loss_list)))
        
        valid_loss_list = []
        model.eval()
        clip_model.eval()
        with torch.no_grad():
            for batch in valid_loader:
                valid_loss = finetune_process_large(
                    batch, clip_model, model, args,
                    # output_tree_model, output_var_model
                )
                if valid_loss is None:
                    continue
                valid_loss_list.append(valid_loss.item())
        print("epoch: {}, valid_loss: {}".format(epoch, np.mean(valid_loss_list)))
        
        writer.add_scalars(
            "loss", 
            {"train_loss": np.mean(loss_list), "valid_loss": np.mean(valid_loss_list)}, 
            epoch
        )

        if args.seq_name == 'mamba' or args.seq_name == 'transformer':
            if args.is_multi_gpu:
                torch.save(
                    model.module.state_dict(),
                    "{}/{}_{}.pth".format(finetune_save_path, args.seq_name, epoch)
                )
            else:
                torch.save(
                    model.state_dict(),
                    "{}/{}_{}.pth".format(finetune_save_path, args.seq_name, epoch)
                )
        else:
            if args.is_multi_gpu:
                model.module.save_pretrained("{}/{}".format(finetune_save_path, args.run_id))
            else:
                model.save_pretrained("{}/{}".format(finetune_save_path, args.run_id))
                
        if args.is_multi_gpu:
            if Path("{}/model_{}.pth".format(save_path, epoch)).exists():
                torch.save(
                    clip_model.module.state_dict(),
                    "{}/model_finetune_{}.pth".format(save_path, epoch)
                )
            else:
                torch.save(
                    clip_model.module.state_dict(),
                    "{}/model_{}.pth".format(save_path, epoch)
                )
        else:
            if Path("{}/model_{}.pth".format(save_path, epoch)).exists():
                torch.save(
                    clip_model.state_dict(),
                    "{}/model_finetune_{}.pth".format(save_path, epoch)
                )
            else:
                torch.save(
                    clip_model.state_dict(),
                    "{}/model_{}.pth".format(save_path, epoch)
                )
        
    writer.close()