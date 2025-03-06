"""
Test LIFT on the LooGLE dataset (general cases).
For timeline reorder tasks only please refer to test_loogle_timeline.py.
"""
from transformers import (
    TrainingArguments,
    PreTrainedTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    BitsAndBytesConfig
)
from lift.args import (
    ModelArguments,
    DataTrainingArguments,
    CustomTrainingArguments,
    parse_args
)
from lift.context_dataset import ContextDataset
from lift.model import load_tokenizer, load_model
from lift.train import train
from dataclasses import dataclass, field
from typing import List, Dict, Optional,Tuple
from numpy.random import randint
from nltk import sent_tokenize
import logging
import json
import os
import torch
import tqdm
from torch.utils.data import Dataset
from copy import deepcopy



LOOGLEFORMAT_NON_ICL = "<|im_start|>user\nBased on the article <<{title}>>, please answer the following question concisely and accurately.\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\nAnswer: "
LOOGLEFORMAT_COT = "<|im_start|>user\nBased on the article <<{title}>>, please recite the most relevant original sentence from article of the following question: {question}<|im_end|>\n<|im_start|>assistant\nSentence:"


@dataclass
class TestArguments:
    input_file: str = field(metadata={"help": "The input file for the test."})
    output_file: str = field(metadata={"help": "The output file for the test."})
    overwrite: bool = field(default=False, metadata={"help": "Whether to overwrite the output file."})
    num_syn_qa: int = field(default=0, metadata={"help": "The number of synthetic QA pairs to generate."})
    title_option: int = field(default=1, metadata={"help": "The title option for the LooGLE dataset."})
    generator_name_or_path: Optional[str] = field(default=None, metadata={"help": "The generator model name or path."})
    use_cot: bool = field(default=False, metadata={'help': "Whether to use CoT in syn. QA and test."})
    num_test: Optional[int] = field(default=None, metadata={'help': "Test only the first several articles."})
    use_icl: bool = field(default=True, metadata={'help': "Whether to use ICL when training and testing."})


def LooGLEtrain(context: str, title: str, tokenizer: PreTrainedTokenizer, model_name_or_path: str, training_args: TrainingArguments, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, use_lora: bool=False, lora_rank: Optional[int]=None, use_pissa: bool=False, load_in_4bit: bool=False, involve_qa_epochs: int=0, gather_batches: bool=True, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_gated_memory: bool=False, use_cot: bool=False, use_icl: bool=True, **kwargs):
    model = load_model(model_name_or_path=model_name_or_path, use_lora=use_lora, lora_rank=lora_rank, use_pissa=use_pissa, load_in_4bit=load_in_4bit, vocab_size=len(tokenizer), use_gated_memory=use_gated_memory)
    return model


def prediction(data: List[Dict], training_args: TrainingArguments, lift_args: Dict, output_file: str, num_resumed: int=0, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_cot: bool=False, use_icl: bool=True):
    tokenizer = load_tokenizer(lift_args['tokenizer_name_or_path'])
    title = "Jos\u00e9 Luis Picardo"
    context = ""
    for i, qa in enumerate(data):
        model = LooGLEtrain(context, title, tokenizer, training_args=training_args, num_syn_qa=num_syn_qa, title_option=title_option, generator_name_or_path=generator_name_or_path, use_cot=use_cot, use_icl=use_icl, **lift_args)
        model.eval()
        if not use_cot:
            input_text = LOOGLEFORMAT_NON_ICL.format(title=title, question=qa['Q'])
        else:
            input_text = LOOGLEFORMAT_COT.format(title=title, question=qa['Q'])
        input_ids = tokenizer(input_text, add_special_tokens=False)['input_ids']#[:-1]
        #print(tokenizer.convert_ids_to_tokens(input_ids))
        input_ids = input_ids + tokenizer(" "+qa['A'], add_special_tokens=False)['input_ids'] 
        #print(tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=False))
        #print(tokenizer.convert_ids_to_tokens(tokenizer(input_text+" "+qa['A'], add_special_tokens=False)["input_ids"]))
        input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
        output, memgate = model.forward(input_ids=input_ids, output_memgate=True)
        memgate = [_.cpu() for _ in memgate]
        print([(_.shape, _.abs().mean(dim=[0, 1, 3]).tolist()) for _ in memgate])
        print(input_ids)


def main():
    training_args, test_args, lift_args = parse_args(
        (TrainingArguments, TestArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)),
        no_dict=(TrainingArguments,)
    )
    input_file = test_args.pop('input_file')
    output_file = test_args.pop('output_file')
    overwrite = test_args.pop('overwrite')
    num_test = test_args.pop('num_test')
    use_icl = test_args.pop('use_icl')


    with open(input_file, 'r') as f:
        input_data = [json.loads(line) for line in f]
    prediction(input_data, training_args, lift_args, output_file, num_resumed=1, use_icl=use_icl, **test_args)


if __name__ == '__main__':
    main()
