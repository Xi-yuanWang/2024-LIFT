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
from typing import List, Dict, Optional
from numpy.random import randint
# from nltk import sent_tokenize
import logging
import json
import os
import torch
import tqdm


LOOGLEFORMAT = "The article {title}: \n{input}\nPlease answer the question based on {title}.\nQuestion: {question}\nAnswer: "
LOOGLEFORMAT_COT = """The article {title}:
{input}
Please recall one or several original sentences from article {title} as evidence, and then answer the question solely based on this evidence.
Please answer in the following format:

# Evidence:
- [evidence 1]
- [evidence 2]
...
# Answer:
[answer]
# End of answer

Please DON'T output quotes when outputing evidences.
# Question:
{question}
# Evidence:
"""


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
    do_check_memgate: bool = field(default=False, metadata={'help': "Whether to check memory gate values; only available with --use_gated_memory True."})


class LooGLEGoldQaDataset(ContextDataset):
    def __init__(self, context: str, title: str, gold_qa: list, tokenizer: PreTrainedTokenizer, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_cot: bool=False):
        # Option 1: prepend title before context
        if title_option == 1:
            context = title + '\n' + context
        super().__init__(context, tokenizer, model_max_length, block_size, len_segment, len_offset)
        # Option 2: prepend title before each segment and predict the whole segment
        if title_option == 2:
            snippet = tokenizer(f"A snippet of {title}: ", add_special_tokens=False)['input_ids']
            self.data = [(snippet + input_ids, 0) for input_ids, _ in self.data]
        # Option 3: prepend title before each segment and predict the content
        if title_option == 3:
            snippet = tokenizer(f"A snippet of {title}: ", add_special_tokens=False)['input_ids']
            self.data = [(snippet + input_ids, len(snippet)) for input_ids, _ in self.data]

        # tmp:
        # repeat data again for the correct ratio of qa and article
        self.num_segments *= 2
        self.data = [d for d in self.data for _ in range(2)]
        
        # Generate QA pairs
        if num_syn_qa > 0:
            for sqa in gold_qa:
                self.data.append(self.formatting_qa(sqa, context, title, model_max_length, use_cot))
        self.enable_qa_tag = False

    def formatting_qa(self, sqa: dict, full_context: str, title: str, model_max_length: int, use_cot: bool=False):
        """ S: evidence sentences; Q: question; A: answer """
        evidences = [sqa['S']]
        question = sqa['Q']
        answer = sqa['A']

        if use_cot:
            input_text = LOOGLEFORMAT_COT.format(title=title, input=full_context, question=question)
            answer = '\n'.join(['- ' + e for e in evidences]) + "\n# Answer:\n" + answer.strip() + "\n# End of answer"
        else:
            input_text = LOOGLEFORMAT.format(title=title, input=full_context, question=question)

        example = input_text + ' ' + answer
        input_ids = self.tokenizer(example, add_special_tokens=False)['input_ids']
        input_length = len(self.tokenizer(input_text, add_special_tokens=False)['input_ids'])

        mixin = self.tokenizer("...", add_special_tokens=False)['input_ids']
        output_length = len(input_ids) - input_length
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length//2 - len(mixin)] + mixin + input_ids[-model_max_length//2:]
            input_length = len(input_ids) - output_length
        return (input_ids, input_length)
    
    def enable_qa(self):
        self.enable_qa_tag = True
        
    def disable_qa(self):
        self.enable_qa_tag = False
    
    def __len__(self):
        return len(self.data) if self.enable_qa_tag else self.num_segments
    
    
def LooGLEtrain(context: str, title: str, gold_qa: list, tokenizer: PreTrainedTokenizer, model_name_or_path: str, training_args: TrainingArguments, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, use_lora: bool=False, lora_rank: Optional[int]=None, use_pissa: bool=False, load_in_4bit: bool=False, involve_qa_epochs: int=0, gather_batches: bool=True, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_gated_memory: bool=False, use_cot: bool=False, **kwargs):
    dataset = LooGLEGoldQaDataset(context, title, gold_qa, tokenizer, model_max_length, block_size, len_segment, len_offset, num_syn_qa, title_option, generator_name_or_path, use_cot)
    model = load_model(model_name_or_path=model_name_or_path, use_lora=use_lora, lora_rank=lora_rank, use_pissa=use_pissa, load_in_4bit=load_in_4bit, vocab_size=len(tokenizer), use_gated_memory=use_gated_memory)
    model = train(model, dataset, tokenizer, training_args, involve_qa_epochs, gather_batches)[0]
    return model


def prediction(data: List[Dict], training_args: TrainingArguments, lift_args: Dict, output_file: str, num_resumed: int=0, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_cot: bool=False, do_check_memgate: bool=False):
    if do_check_memgate and not lift_args['use_gated_memory']:
        do_check_memgate = False
        logging.warning("--do_check_mem_gate True should be used with --use_gated_memory True; set --do_check_mem_gate to False")
    tokenizer = load_tokenizer(lift_args['tokenizer_name_or_path'])
    mixin = tokenizer("...", add_special_tokens=False)['input_ids']
    model_max_length = lift_args['model_max_length']
    
    for i, sample in enumerate(tqdm.tqdm(data, desc="Sample")):
        if i < num_resumed:
            continue
        context = sample['input']
        title = sample['title']
        qa_pairs = sample['test_qa_pairs']
        train_qa_pairs = sample['train_qa_pairs']
        model = LooGLEtrain(context, title, train_qa_pairs, tokenizer, training_args=training_args, num_syn_qa=num_syn_qa, title_option=title_option, generator_name_or_path=generator_name_or_path, use_cot=use_cot, **lift_args)
        model.eval()
        for qa_pair in tqdm.tqdm(qa_pairs, desc="QA Pair"):
            if use_cot:
                input_text = LOOGLEFORMAT_COT.format(title=title, input=context, question=qa_pair['Q'])
            else:
                input_text = LOOGLEFORMAT.format(title=title, input=context, question=qa_pair['Q'])
            input_ids = tokenizer(input_text, add_special_tokens=False)['input_ids']
            if len(input_ids) > model_max_length:
                input_ids = input_ids[:model_max_length//2 - len(mixin)] + mixin + input_ids[-model_max_length//2:]
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
            attention_mask = torch.ones_like(input_ids)
            # terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                # eos_token_id=terminators,
                max_new_tokens=200,
                use_cache=True,
                do_sample=False,
            )
            response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
            qa_pair['pred'] = response
        output_case = {
            'title': title,
            'input': context,
            'qa_pairs': qa_pairs
        }
        with open(output_file, 'a') as f:
            f.write(json.dumps(output_case) + '\n')


def main():
    training_args, test_args, lift_args = parse_args(
        (TrainingArguments, TestArguments, (ModelArguments, CustomTrainingArguments, DataTrainingArguments)),
        no_dict=(TrainingArguments,)
    )
    input_file = test_args.pop('input_file')
    output_file = test_args.pop('output_file')
    overwrite = test_args.pop('overwrite')
    num_test = test_args.pop('num_test')
    num_resumed = 0
    if os.path.exists(output_file):
        if overwrite:
            os.remove(output_file)
        else:
            with open(output_file, 'r') as f:
                num_resumed = len(f.readlines())
    with open(input_file, 'r') as f:
        input_data = [json.loads(line) for line in f]
    if num_test is not None:
        input_data = input_data[:num_test]
    prediction(input_data, training_args, lift_args, output_file, num_resumed=num_resumed, **test_args)


if __name__ == '__main__':
    main()

