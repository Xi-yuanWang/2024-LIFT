import torch
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from typing import List, Tuple, Optional
from copy import deepcopy
from random import randint


class ContextDataset(Dataset):
    """Given a piece of context, `ContextDataset` creates a torch-Dataset, using the truncation strategy described in our paper.
    """
    def __init__(
        self,
        context: str,
        tokenizer: PreTrainedTokenizer,
        model_max_length: int = 7800,
        block_size: int = 256,
        len_segment: int = 8,
        len_offset: int = 3,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        texts = context.replace('\0', ' ')
        input_ids = self.tokenizer(texts, add_special_tokens=False)['input_ids']
        len_segment = len_segment * block_size
        len_offset = len_offset * block_size
        # Generate datapoints
        self.data = [(input_ids[s:s+len_segment], 0) for s in range(0, len(input_ids), len_offset)]
        self.num_segments = len(self.data)  # record the number of context datapoints

    def __len__(self):
        return self.num_segments

    def preprocessing(self, example: Tuple[List[int], int]):
        input_ids, len_input = example
        labels = deepcopy(input_ids)
        # Clip and truncation
        input_ids = input_ids[:self.model_max_length]
        labels = labels[:self.model_max_length]
        # Transfer to Tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        labels[:len_input] = -100  # mask the unsupervised part
        attention_mask = torch.ones_like(input_ids)
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
    
    def __getitem__(self, index):
        return self.preprocessing(self.data[index])
    
    def enable_qa(self):
        raise NotImplementedError
    
    def disable_qa(self):
        raise NotImplementedError
    
    def generate_task(self):
        raise NotImplementedError


class RandomContextDataset(ContextDataset):
    def __init__(
        self,
        context: str,
        tokenizer: PreTrainedTokenizer,
        model_max_length: int = 7800,
        block_size: int = 256,
        len_segment: int = 8,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.len_segment = len_segment * block_size
        self.model_max_length = model_max_length
        context = context.replace('\0', ' ')
        self.input_ids = [self.tokenizer.bos_token_id] + self.tokenizer(context, add_special_tokens=False)['input_ids'] + [self.tokenizer.eos_token_id]
        self.num_segments = (len(self.input_ids) // self.len_segment + 1) * 3  # num_segments determines the batch size
    
    def __getitem__(self, index):
        st = randint(0, len(self.input_ids) - self.len_segment + 1)
        return self.preprocessing((self.input_ids[st:st+self.len_segment], self.len_segment // 2))
