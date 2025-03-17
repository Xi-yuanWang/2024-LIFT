import torch
import torch.utils
from torch import Tensor
from torch.utils.data import Dataset
import torch.utils.data
from transformers import (
    TrainingArguments,
    Trainer,
    PreTrainedTokenizer,
    PreTrainedModel
)
from .context_dataset import ContextDataset
from typing import Optional, Type, Any, List, Dict
from copy import deepcopy
from .gated_memory.model_qwen import GMQwen2ForCausalLM, DistillCache
import os.path as osp
from tqdm import tqdm
from copy import deepcopy
def my_collator(features: List[Dict[str, torch.Tensor]], return_tensors="pt") -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:

        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object

    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.

    if return_tensors == "pt":
        ret = {}
        ret["labels"] = torch.nn.utils.rnn.pad_sequence([_["labels"] for _ in features], batch_first=True, padding_value=-100, padding_side='right')
        ret["input_ids"] = torch.nn.utils.rnn.pad_sequence([_["input_ids"] for _ in features], batch_first=True, padding_value=0, padding_side='right')
        ret["gate_mask"] = torch.nn.utils.rnn.pad_sequence([_["gate_mask"] for _ in features], batch_first=True, padding_value=1, padding_side='right')
        return ret
    else:
        raise NotImplementedError

    

def load_trainer(model: PreTrainedModel, training_dataset: Dataset, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, eval_dataset: Optional[Dataset]=None, gather_batches: bool=False, optimizer: Optional[torch.optim.Optimizer]=None):
    """Load the training and the model (if the model is not instantiated).
    Args:
        model (PreTrainedModel): the model to train.
        training_dataset (Dataset): the training dataset.
        tokenizer (PreTrainedTokenizer): the tokenizer.
        training_args (TrainingArguments): the huggingface training arguments.
        eval_dataset (Dataset): OPTIONAL; the evaluation dataset.
        gather_batches (bool): OPTIONAL, default to `False`; if `gather_batches=True`, it will force the trainer to update the model only once every epoch; it may lead to more stable gradients.
        optimizer (torch.optim.Optimizer): OPTIONAL; the optimizer to use.
    Returns:
        trainer_model_pair (tuple[Trainer, Module]): the trainer and the model to train.
    """
    training_args = deepcopy(training_args)
    if gather_batches:
        training_args.gradient_accumulation_steps = len(training_dataset)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=training_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        optimizers=(optimizer, None),
        data_collator=my_collator
    )
    return trainer, model


def kvtrain(model: GMQwen2ForCausalLM, dataset: ContextDataset, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, kv_epoches: int=0, gather_batches: bool=True):
    """Fine-tune the model and the corresponding tokenizer.
    Args:
        model (PreTrainedModel): the model to fine-tune.
        dataset (ContextDataset): the dataset for fine-tuning.
        tokenizer (PreTrainedTokenizer): the pretrained tokenizer.
        training_args (TrainingArguments): the huggingface training arguments.
        involve_qa_epochs (int): OPTIONAL, default to `0`; the number of epochs to involve QA pairs.
        gather_batches (bool): OPTIONAL, default to `True`; if `gather_batches=True`, it will force the trainer to update the model only once every epoch; it may lead to more stable gradients.
    Returns:
        model_tokenizer_pair (tuple[PreTrainedModel, PreTrainedTokenizer]): the fine-tuned model and the corresponding tokenizer.
    """
    model.eval()
    torch.cuda.empty_cache()  # Manually release memory
    dataset.disable_qa()
    print("kv train numel", sum([_.numel() for _ in model.parameters() if _.requires_grad]))
    optimizer = torch.optim.AdamW([_ for _ in model.parameters() if _.requires_grad], lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    basemodel = model.model.model
    from tqdm import tqdm
    kvcaches = []
    for data in dataset:
        with torch.no_grad():
            input_id = data["input_ids"].unsqueeze(0).to(model.device)
            outputs = model.forward(input_ids=input_id, gate_mask=torch.zeros_like(input_id), use_cache=True)
            kvcaches.append(outputs.past_key_values)
    import random
    for _ in tqdm(range(kv_epoches*10)):
        random.shuffle(kvcaches)
        for kvcache in kvcaches:
            if True:
                memouts = model.forward(input_ids=input_id, gate_mask=torch.zeros_like(input_id), past_key_values=kvcache, use_cache=True, output_memout=True)
                ks = kvcache.key_cache
                vs = kvcache.value_cache
                loss = []
                for layer_idx in range(basemodel.layer_start_idx, basemodel.config.num_hidden_layers-basemodel.layer_end_idx):
                    # basemodel.layers[layer_idx].self_attn.unset_memproj_numgroup()
                    k, v, memout = ks[layer_idx], vs[layer_idx], memouts[layer_idx]
                    loss.append(torch.mean(torch.square(memout - v)))
                    # basemodel.layers[layer_idx].self_attn.set_memproj_numgroup()
                loss = torch.stack(loss).mean()
                loss.backward()
                print(f"kv loss {loss.item():.3e}", flush=True)
                optimizer.step()
                optimizer.zero_grad()
    return model, optimizer


def distilltrain(model: GMQwen2ForCausalLM, dataset: ContextDataset, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, kv_epoches: int=0, gather_batches: bool=True):
    """Fine-tune the model and the corresponding tokenizer.
    Args:
        model (PreTrainedModel): the model to fine-tune.
        dataset (ContextDataset): the dataset for fine-tuning.
        tokenizer (PreTrainedTokenizer): the pretrained tokenizer.
        training_args (TrainingArguments): the huggingface training arguments.
        involve_qa_epochs (int): OPTIONAL, default to `0`; the number of epochs to involve QA pairs.
        gather_batches (bool): OPTIONAL, default to `True`; if `gather_batches=True`, it will force the trainer to update the model only once every epoch; it may lead to more stable gradients.
    Returns:
        model_tokenizer_pair (tuple[PreTrainedModel, PreTrainedTokenizer]): the fine-tuned model and the corresponding tokenizer.
    """
    import torch


    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


    def sdpa_attention_forward(
        num_key_value_groups: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = False,
    ):
        key = repeat_kv(key, num_key_value_groups)
        value = repeat_kv(value, num_key_value_groups)

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )
        attn_output = attn_output.contiguous()
        return attn_output
    
    from copy import deepcopy
    model.eval()
    torch.cuda.empty_cache()  # Manually release memory
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    basemodel = model.model.model
    scaling = basemodel.layers[0].self_attn.scaling
    num_key_value_groups = basemodel.layers[0].self_attn.num_key_value_groups
    from tqdm import tqdm
    kvcaches = []

    torch.cuda.empty_cache()
    with torch.no_grad():
        basetext = dataset[0]["basetext"].unsqueeze(0).to(model.device)
        len_context = dataset[0]["len_context"]
        basecache: DistillCache = model.forward(input_ids=basetext, gate_mask=torch.zeros_like(basetext), past_key_values=DistillCache(), use_cache=True).past_key_values
        basecache.cpu()
        torch.cuda.empty_cache()
        for data in dataset:
            torch.cuda.empty_cache()
            input_id = data["input_ids"].unsqueeze(0).to(model.device)
            kvcache: DistillCache = model.forward(input_ids=input_id, gate_mask=torch.zeros_like(input_id), past_key_values=deepcopy(basecache), use_cache=True).past_key_values
            kvcache.cpu()
            torch.cuda.empty_cache()

            for idx in range(basemodel.layer_start_idx, basemodel.config.num_hidden_layers-basemodel.layer_end_idx):#kvcache.query_cache:
                q, k, v = kvcache.query_cache[idx].to(model.device), kvcache.key_cache[idx].to(model.device), kvcache.value_cache[idx].to(model.device)
                # print(q.shape, k.shape, v.shape)
                k_lower = k[:, :, :len_context]#k[:, :, :len_context]
                v_lower = v[:, :, :len_context]
                kvcache.virtual_attnout_cache[idx] = sdpa_attention_forward(num_key_value_groups, q, k_lower, v_lower, 0.0, scaling, False).cpu()
                q_upper = q[:, :, len_context:]
                k_upper = k[:, :, len_context:]
                v_upper = v[:, :, len_context:]
                kvcache.post_attnout_cache[idx] = sdpa_attention_forward(num_key_value_groups, q_upper, k_upper, v_upper, 0.0, scaling, True)
                kvcache.cpu()
                torch.cuda.empty_cache()
            kvcaches.append((kvcache, len_context))
        torch.cuda.empty_cache()
    import random
    for _ in tqdm(range(kv_epoches)):
        random.shuffle(kvcaches)
        for kvcache, len_context in kvcaches:
            if True:
                meml1loss = []
                memcosloss = []
                gatel1loss = []
                gatecosloss = []
                for layer_idx in range(basemodel.layer_start_idx, basemodel.config.num_hidden_layers-basemodel.layer_end_idx):
                    q = kvcache.query_cache[layer_idx].to(model.device, non_blocking=True)
                    vattnout = kvcache.virtual_attnout_cache[layer_idx].to(model.device, non_blocking=True)
                    memout = basemodel.layers[layer_idx].self_attn.mem_proj(q)
                    tmeml1loss = (memout - vattnout).abs().mean()
                    tmemcosloss = 1 - torch.nn.CosineSimilarity(dim=-1)(memout, vattnout).mean()
                    (tmeml1loss + tmemcosloss).backward()
                    
                    meml1loss.append(tmeml1loss.detach())
                    memcosloss.append(tmemcosloss.detach())

                    q = q[:, :, len_context:]
                    k = None#kvcache.key_cache[layer_idx][:, :, len_context:].to(model.device, non_blocking=True)

                    postattnout = kvcache.post_attnout_cache[layer_idx].to(model.device, non_blocking=True)
                    attnout = kvcache.attnout_cache[layer_idx][:, len_context:].transpose(1, 2).to(model.device, non_blocking=True)
                    vattnout = vattnout[:, :, len_context:]
                    
                    gateout = basemodel.layers[layer_idx].self_attn.gate_proj(q, k)
                    
                    predout = postattnout + gateout * (vattnout-postattnout)
                    tgatel1loss = (predout-attnout).abs().mean()
                    tgatecosloss = 1 - torch.nn.CosineSimilarity(dim=-1)(predout, attnout).mean()
                    (tgatel1loss + tgatecosloss).backward()
                    gatel1loss.append(tgatel1loss.detach())
                    gatecosloss.append(tgatecosloss.detach())
                    kvcache.cpu()
                meml1loss = torch.stack(meml1loss)
                memcosloss = torch.stack(memcosloss)
                gatel1loss = torch.stack(gatel1loss)
                gatecosloss = torch.stack(gatecosloss)
                print("cos mem", memcosloss.cpu().tolist())
                print("cos gate", gatecosloss.cpu().tolist())
                print("l1 mem", meml1loss.cpu().tolist())
                print("l1 gate", gatel1loss.cpu().tolist(), flush=True)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            torch.cuda.empty_cache()
    return model, optimizer



def distilltrain2(model: GMQwen2ForCausalLM, dataset: ContextDataset, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, kv_epoches: int=0, gather_batches: bool=True):
    """Fine-tune the model and the corresponding tokenizer.
    Args:
        model (PreTrainedModel): the model to fine-tune.
        dataset (ContextDataset): the dataset for fine-tuning.
        tokenizer (PreTrainedTokenizer): the pretrained tokenizer.
        training_args (TrainingArguments): the huggingface training arguments.
        involve_qa_epochs (int): OPTIONAL, default to `0`; the number of epochs to involve QA pairs.
        gather_batches (bool): OPTIONAL, default to `True`; if `gather_batches=True`, it will force the trainer to update the model only once every epoch; it may lead to more stable gradients.
    Returns:
        model_tokenizer_pair (tuple[PreTrainedModel, PreTrainedTokenizer]): the fine-tuned model and the corresponding tokenizer.
    """

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def sdpa_attention_forward(
        num_key_value_groups: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dropout: float = 0.0,
        scaling: Optional[float] = None,
        is_causal: Optional[bool] = False,
    ):
        key = repeat_kv(key, num_key_value_groups)
        value = repeat_kv(value, num_key_value_groups)

        query = query.contiguous()
        key = key.contiguous()
        value = value.contiguous()
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=dropout,
            scale=scaling,
            is_causal=is_causal,
        )
        attn_output = attn_output.contiguous()
        return attn_output
    
    model.eval()
    torch.cuda.empty_cache()  # Manually release memory
    basemodel = model.model.model
    scaling = basemodel.layers[0].self_attn.scaling
    num_key_value_groups = basemodel.layers[0].self_attn.num_key_value_groups
    
    q2vattn = {idx: [[], []] for idx in range(basemodel.layer_start_idx, basemodel.config.num_hidden_layers-basemodel.layer_end_idx)}
    q2vpaattn = {idx: [[], [], [], []] for idx in range(basemodel.layer_start_idx, basemodel.config.num_hidden_layers-basemodel.layer_end_idx)}

    torch.cuda.empty_cache()
    with torch.no_grad():
        basetext = dataset[0]["basetext"].unsqueeze(0).to(model.device)
        len_context = dataset[0]["len_context"]
        basecache: DistillCache = model.forward(input_ids=basetext, gate_mask=torch.zeros_like(basetext), past_key_values=DistillCache(), use_cache=True).past_key_values
        basecache.cpu()
        torch.cuda.empty_cache()
        for data in tqdm(dataset, desc="get data"):
            torch.cuda.empty_cache()
            input_id = data["input_ids"].unsqueeze(0).to(model.device)
            kvcache: DistillCache = model.forward(input_ids=input_id, gate_mask=torch.zeros_like(input_id), past_key_values=deepcopy(basecache), use_cache=True).past_key_values
            kvcache.cpu()
            torch.cuda.empty_cache()

            for idx in range(basemodel.layer_start_idx, basemodel.config.num_hidden_layers-basemodel.layer_end_idx):#kvcache.query_cache:
                def execcache():
                    q, k, v, attnout = kvcache.query_cache[idx], kvcache.key_cache[idx], kvcache.value_cache[idx], kvcache.attnout_cache[idx]
                    q_upper = q[:, :, len_context:]
                    k_lower = k[:, :, :len_context]
                    v_lower = v[:, :, :len_context]
                    k_upper = k[:, :, len_context:]
                    v_upper = v[:, :, len_context:]
                    attnout_upper = attnout.transpose(1, 2)[:, :, len_context:]
                    if len(q2vattn[idx][0]) > 0:
                        q = q_upper
                    
                    vattnout = sdpa_attention_forward(num_key_value_groups, q.to(model.device), k_lower.to(model.device), v_lower.to(model.device), 0.0, scaling, False).cpu()
                    if len(q2vattn[idx][0]) > 0:
                        vattnout_upper = vattnout
                    else:
                        vattnout_upper = vattnout[:, :, len_context:]
                    q2vattn[idx][0].append(q)
                    q2vattn[idx][1].append(vattnout)
                    
                    q2vpaattn[idx][0].append(q_upper)
                    q2vpaattn[idx][1].append(vattnout_upper)
                    q2vpaattn[idx][2].append(sdpa_attention_forward(num_key_value_groups, q_upper.to(model.device), k_upper.to(model.device), v_upper.to(model.device), 0.0, scaling, True).cpu())
                    q2vpaattn[idx][3].append(attnout_upper)
                execcache()
                torch.cuda.empty_cache()
            del kvcache
        del basecache
    for idx in range(basemodel.layer_start_idx, basemodel.config.num_hidden_layers-basemodel.layer_end_idx):
        q2vattn[idx][0] = torch.concat(q2vattn[idx][0], dim=2)#.transpose(0, 2)
        q2vattn[idx][1] = torch.concat(q2vattn[idx][1], dim=2)#.transpose(0, 2)

        q2vpaattn[idx][0] = torch.concat(q2vpaattn[idx][0], dim=2)#.transpose(0, 2)
        q2vpaattn[idx][1] = torch.concat(q2vpaattn[idx][1], dim=2)#.transpose(0, 2)
        q2vpaattn[idx][2] = torch.concat(q2vpaattn[idx][2], dim=2)#.transpose(0, 2)
        q2vpaattn[idx][3] = torch.concat(q2vpaattn[idx][3], dim=2)#.transpose(0, 2)


    BATCHSIZE = 16384
    for idx in range(basemodel.layer_start_idx, basemodel.config.num_hidden_layers-basemodel.layer_end_idx):
        memproj = basemodel.layers[idx].self_attn.mem_proj
        optimizer = torch.optim.AdamW(memproj.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        Q, VATTN = q2vattn[idx][0].to(model.device), q2vattn[idx][1].to(model.device)
        LEN = Q.shape[2]
        #dataset = torch.utils.data.TensorDataset(q2vattn[idx][0].to(model.device), q2vattn[idx][1].to(model.device))
        #dataloader = torch.utils.data.DataLoader(dataset, batch_size=32768, shuffle=True)#, pin_memory=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, kv_epoches)
        for _ in tqdm(range(kv_epoches), desc=f"mem {idx}"):
            perms = torch.split(torch.randperm(LEN, device=model.device), BATCHSIZE)
            for perm in perms:
                q: torch.Tensor = Q[:, :, perm]#.transpose(0, 2)
                vattn: torch.Tensor = VATTN[:, :, perm]#.transpose(0, 2)
                memout: torch.Tensor = memproj(q)
                l1loss: torch.Tensor = (memout - vattn).abs().flatten()
                cosloss: torch.Tensor = 1 - torch.nn.CosineSimilarity(dim=-1)(memout, vattn).flatten()
                l1loss = torch.sum(l1loss.clamp_min_(3e-3) * torch.softmax(l1loss.detach(), dim=0))
                cosloss = torch.sum(cosloss.clamp_min_(3e-3) * torch.softmax(cosloss.detach(), dim=0))
                (l1loss + cosloss).backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            print(f"mem {idx} epoch {_} {l1loss.item():.3f} {cosloss.item():.3f}", flush=True)
        del Q, VATTN
        torch.cuda.empty_cache()

    for idx in range(basemodel.layer_start_idx, basemodel.config.num_hidden_layers-basemodel.layer_end_idx):
        gateproj = basemodel.layers[idx].self_attn.gate_proj
        optimizer = torch.optim.AdamW(gateproj.parameters(), lr=training_args.learning_rate, weight_decay=training_args.weight_decay)
        Q, VATTN, PATTN, ATTN = q2vpaattn[idx][0].to(model.device), q2vpaattn[idx][1].to(model.device), q2vpaattn[idx][2].to(model.device), q2vpaattn[idx][3].to(model.device)
        LEN = Q.shape[2]
        #dataset = torch.utils.data.TensorDataset(q2vpaattn[idx][0].to(model.device), q2vpaattn[idx][1].to(model.device), q2vpaattn[idx][2].to(model.device), q2vpaattn[idx][3].to(model.device))
        #dataloader = torch.utils.data.DataLoader(dataset, batch_size=32768, shuffle=True)#, pin_memory=True)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, (kv_epoches//10)*len(dataloader))
        for _ in tqdm(range(kv_epoches), desc=f"gate {idx}"):
            perms = torch.split(torch.randperm(LEN, device=model.device), 16384)
            for perm in perms:
                q: torch.Tensor = Q[:, :, perm]
                vattn: torch.Tensor = VATTN[:, :, perm]
                pattn: torch.Tensor = PATTN[:, :, perm]
                attn: torch.Tensor = ATTN[:, :, perm]
                gateout: torch.Tensor = gateproj(q)
                predout = pattn + gateout * (vattn-pattn)
                l1loss: torch.Tensor = (predout - attn).abs().flatten()
                cosloss: torch.Tensor = 1 - torch.nn.CosineSimilarity(dim=-1)(predout, attn).flatten()
                l1loss = torch.sum(l1loss.clamp_min_(1e-2) * torch.softmax(l1loss.detach(), dim=0))
                cosloss = torch.sum(cosloss.clamp_min_(3e-3) * torch.softmax(cosloss.detach(), dim=0))
                (l1loss + cosloss).backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            print(f"gate {idx} epoch {_} {l1loss.item():.3f} {cosloss.item():.3f}", flush=True)
        del Q, VATTN, PATTN, ATTN
        torch.cuda.empty_cache()
    return model, optimizer


def train(model: PreTrainedModel, dataset: ContextDataset, tokenizer: PreTrainedTokenizer, training_args: TrainingArguments, involve_qa_epochs: int=0, gather_batches: bool=True):
    """Fine-tune the model and the corresponding tokenizer.
    Args:
        model (PreTrainedModel): the model to fine-tune.
        dataset (ContextDataset): the dataset for fine-tuning.
        tokenizer (PreTrainedTokenizer): the pretrained tokenizer.
        training_args (TrainingArguments): the huggingface training arguments.
        involve_qa_epochs (int): OPTIONAL, default to `0`; the number of epochs to involve QA pairs.
        gather_batches (bool): OPTIONAL, default to `True`; if `gather_batches=True`, it will force the trainer to update the model only once every epoch; it may lead to more stable gradients.
    Returns:
        model_tokenizer_pair (tuple[PreTrainedModel, PreTrainedTokenizer]): the fine-tuned model and the corresponding tokenizer.
    """
    # load tokenzier
    torch.cuda.empty_cache()  # Manually release memory
    # Load and finetune the model
    if involve_qa_epochs > 0:
        dataset.disable_qa()
    trainer, model = load_trainer(
        model=model,
        training_dataset=dataset,
        tokenizer=tokenizer,
        training_args=training_args,
        gather_batches=gather_batches,
    )
    if training_args.num_train_epochs > 0:
        trainer.train()
        trainer.save_model(osp.join(trainer.args.output_dir, "after_lift"))
    # Load the dataset with QA pairs and continue-finetune the model
    if involve_qa_epochs > 0:
        dataset.enable_qa()
        training_args_syn = deepcopy(training_args)
        training_args_syn.num_train_epochs = involve_qa_epochs
        #training_args_syn.lr_scheduler_type = "constant"
        #training_args_syn.output_dir="./ckpts",
        #training_args_syn.save_strategy="steps",  # Save every `save_steps`
        #training_args_syn.save_steps=1,         # Save every 500 steps
        #training_args_syn.save_total_limit=2,     # Keep only the last 2 checkpoints
        trainer_syn, model = load_trainer(
            model=model,
            training_dataset=dataset,
            tokenizer=tokenizer,
            training_args=training_args_syn,
            gather_batches=gather_batches,
            optimizer=trainer.optimizer,
        )
        trainer_syn.train()
        trainer_syn.save_model(osp.join(trainer_syn.args.output_dir, "after_syn_qa"))
    # Clear cache
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    torch.cuda.empty_cache()
    return model, tokenizer
