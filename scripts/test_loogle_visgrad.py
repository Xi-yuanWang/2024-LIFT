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
from lift.train import train_visgrad
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


LOOGLEFORMAT_NON_ICL = "<|im_start|>user\nBased on the article <<{title}>>, please answer the following question concisely and accurately: \nQuestion: {question}<|im_end|>\n<|im_start|>assistant\nAnswer: "
LOOGLEFORMAT_COT = "<|im_start|>user\nBased on the article <<{title}>> and the following question, please first recall original sentences related to the question as evidence, and then answer the question solely based on this evidence: \nQuestion: {question}<|im_end|>\n<|im_start|>assistant\nEvidence: "
LIFT_ICL_PROMPT = "<|im_start|>user\nPlease remember an article and use it to answer a question later. One long segment in the article <<{title}>> is ..."


class ICLContextDataset(Dataset):
    """Given a piece of context, `ContextDataset` creates a torch-Dataset, using the truncation strategy described in our paper.
    """
    def __init__(self, qa_pairs, title: str, context: List[str], tokenizer: PreTrainedTokenizer, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3):
        """
        Args:
            context (str): the context to train on.
            tokenizer (PreTrainedTokenizer): the AutoTokenizer.
            model_max_length (int): OPTIONAL, default to `4096`; the texts will be clipped at the `model_max_length`-th token.
            block_size (int): OPTIONAL, default to `256`; the number of tokens in a block; a block is the unit of segments and offsets.
            len_segment (int): OPTIONAL, default to `8`; the number of units in a segment; the article is divided into segments.
            len_offset (int): OPTIONAL, default to `3`; the number of units per offset; it determines the offset from one segment to the next one.
        """
        self.ignore_index = -100  # The default value for ignored labels in torch
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        texts = (c.replace('\0', ' ') for c in context)
        input_ids = list(self.tokenizer(t, add_special_tokens=False)['input_ids'] for t in texts)
        len_segment = len_segment * block_size
        len_offset = len_offset * block_size

        # Generate datapoints
        mixin = self.tokenizer("...", add_special_tokens=False)['input_ids']
        prompt = self.tokenizer(LIFT_ICL_PROMPT.format(title=title), add_special_tokens=False)['input_ids']

        self.data = []
        for tid in range(len(input_ids)):
            for s in range(0, len(input_ids[tid]) - int(0.5*len_offset), len_offset):
                start_pos = s
                end_pos = min(s + len_segment, len(input_ids[tid]))
                lift_icl = prompt 
                input_len = len(lift_icl)
                self.data.append((tid, lift_icl, start_pos, end_pos, int(0.5*(len_offset)), input_len))
            print("len", len(self.data))
        for qa in qa_pairs:
            self.data.append((torch.tensor(self.tokenizer(LOOGLEFORMAT_COT.format(title=title, question=qa["Q"]), add_special_tokens=False)['input_ids']), torch.tensor(self.tokenizer(qa["A"], add_special_tokens=False)['input_ids'])))
        print("len", len(self.data))
        self.input_ids = input_ids
        #self.num_segments = len(self.data)  # record the number of context datapoints

    def __len__(self):
        return len(self.data)#self.num_segments
    
    def prepare_lift_icl(self, start_pos: int, end_pos: int, input_ids: list[int], len_segment: int, len_offset: int, len_lift_icl: int=4096):

        def get_fix_length_segments(front_lim: int, back_lim: int, tot_len: int):
            """
                return two segments,
                satisfying frist segment length is less than 'front_lim', 
                second segment length is less than 'back_lim'
                and the sum of two segments' length is equal to 'tot_len'.


                The return value is the length of the two segments.
            """
            a = randint(max(0, tot_len - back_lim), min(front_lim, tot_len))
            b = tot_len - a
            return a, b

        front_len, back_len = get_fix_length_segments(len_lift_icl, len_lift_icl, len_lift_icl)
        front_st = randint(0, len_lift_icl - front_len)
        back_ed = randint(0, len_lift_icl - back_len)

        return input_ids[front_st: front_st+front_len], input_ids[-back_ed-back_len:-back_ed] if back_ed != 0 else input_ids[-back_len:]

    def preprocessing(self, example: Tuple[List[int], int]):
        input_ids, len_input = example
        labels = deepcopy(input_ids)
        # Clip and truncation
        input_ids = input_ids[:self.model_max_length]
        labels = labels[:self.model_max_length]
        # Transfer to Tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        labels[:len_input] = self.ignore_index  # mask the unsupervised part
        return {
            'input_ids': input_ids,
            'labels': labels,
        }
    
    def preprocessing_qa(self, example: Tuple[List[int], int]):
        Q, A = example
        input_ids = torch.concat((Q, A), dim=-1)
        labels = input_ids.clone()
        labels[:Q.shape[-1]] = self.ignore_index  # mask the unsupervised part
        return {
            'input_ids': input_ids,
            'labels': labels,
        }
    
    def __getitem__(self, index):
        #print(index)
        if len(self.data[index]) > 2:
            tid, lift_icl, start_pos, end_pos, var, input_len = self.data[index]
            import random 
            offset = random.randint(-var, var)
            def constrainidx(a):
                return min(max(a, 0), len(self.input_ids[tid]))
            ret = self.preprocessing((lift_icl+self.input_ids[tid][constrainidx(start_pos+offset): constrainidx(end_pos+offset)], input_len))
        else:
            ret = self.preprocessing_qa(self.data[index])
        print("input_ids==", self.tokenizer.decode(ret["input_ids"], skip_special_tokens=False))
        print("label==", self.tokenizer.decode(ret["labels"][ret["labels"]>=0], skip_special_tokens=False))
        return ret
    
    def enable_qa(self):
        raise NotImplementedError
    
    def disable_qa(self):
        raise NotImplementedError
    
    def generate_task(self):
        raise NotImplementedError


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


class LooGLEDataset(ICLContextDataset):
    def __init__(self, qa_pairs, context: str, title: str, tokenizer: PreTrainedTokenizer, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_cot: bool=False, use_icl: bool=True):
        # Option 1: prepend title before context
        if title_option == 1:
            context = ("Title: " + title + '\n' + c for c in context)
        super().__init__(qa_pairs, title, context, tokenizer, model_max_length, block_size, len_segment, len_offset)
        # Option 2: prepend title before each segment and predict the whole segment
        if title_option == 2:
            snippet = tokenizer(f"A snippet of {title}: ", add_special_tokens=False)['input_ids']
            self.data = [(snippet + input_ids, 0) for input_ids, _ in self.data]
        # Option 3: prepend title before each segment and predict the content
        if title_option == 3:
            snippet = tokenizer(f"A snippet of {title}: ", add_special_tokens=False)['input_ids']
            self.data = [(snippet + input_ids, len(snippet)) for input_ids, _ in self.data]
        # Generate QA pairs
        self.textdata = self.data
        if num_syn_qa > 0:
            self.qadata = []
            context_sent = sent_tokenize(context)
            assert len(context_sent) >= 16, "The length of the context should be at least 25 sentences."
            generator = AutoModelForCausalLM.from_pretrained(
                generator_name_or_path,
                device_map='auto',
                torch_dtype=torch.bfloat16,
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                    do_sample=False,
                ),
            )
            gen_tokenizer = load_tokenizer(generator_name_or_path)
            generator.eval()
            for _ in range(num_syn_qa):
                result = self.generate_task(generator, gen_tokenizer, context, context_sent, title, model_max_length, use_cot, use_icl=use_icl)
                if result is not None:
                    self.qadata.append(result)
        self.enable_qa_tag = False
    
    @torch.no_grad()
    def generate_task(self, generator: PreTrainedModel, tokenizer: PreTrainedTokenizer, full_context: str, context_sent: List[str], title: str, model_max_length: int, use_cot: bool=False, use_icl: bool=True):
        st_pos = randint(0, len(context_sent) - 8)
        context = ' '.join(context_sent[st_pos:st_pos+8])
        messages = [
            {
                'role': "system",
                'content': "You are a helpful assistant."
            },
            {
                'role': "user", 
                'content': f"You are given a piece of text as the context. You should generate ONLY one question and the corresponding concise answer according to the context. You should also select one or more sentences directly from the original context as the evidence. The evidences must be verbatim sentences from the context. Please answer in the following format: \nQuestion: [question] \nAnswer: [answer] \nEvidence: [evidence]\nPlease DON'T output quotes when outputting evidences. The following is the piece of text: {context}"
            }
        ]
        input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(generator.device)
        mask_attention = torch.ones_like(input_ids)
        # terminators = [tokenizer.eos_token_id, tokenizer.pad_token_id]
        # print("sp token", tokenizer.pad_token_id, tokenizer.eos_token_id)
        for _ in range(5):
            # print(input_ids)
            outputs = generator.generate(
                input_ids=input_ids,
                attention_mask=mask_attention.to(generator.device),
                max_new_tokens=1024,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
            response = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
            question_position = response.find("Question:")
            answer_position = response.find("Answer:")
            evidence_position = response.find("Evidence:")
            
            if question_position == -1 or answer_position == -1 or evidence_position == -1:
                continue
            question = response[question_position + 9:answer_position].strip()
            answer = response[answer_position + 7:evidence_position].strip()
            evidence = response[evidence_position + 9:].strip()
            if evidence not in context:
                pass #continue
            break
        else:
            logging.warning("Fail to generate a QA pair, skip.")
            return None
        if not use_cot:
            input_text = LOOGLEFORMAT_NON_ICL.format(title=title, question=qa_pair['Q'])
        else:
            input_text = LOOGLEFORMAT_COT.format(title=title, question=question)
        example = input_text + ' ' + answer# + self.tokenizer.eos_token
        print(f'syn input text: {example}')
        input_ids = self.tokenizer(example, add_special_tokens=False)['input_ids']
        input_ids = input_ids + [self.tokenizer.eos_token_id]
        input_length = len(self.tokenizer(input_text, add_special_tokens=False)['input_ids']) 
        mixin = self.tokenizer("...", add_special_tokens=False)['input_ids']
        output_length = len(input_ids) - input_length
        if len(input_ids) > model_max_length:
            raise NotImplementedError
            input_ids = input_ids[:model_max_length//2 - len(mixin)] + mixin + input_ids[-model_max_length//2:]
            input_length = len(input_ids) - output_length
        return (input_ids, input_length)
    
    def enable_qa(self):
        self.enable_qa_tag = True
        self.data = self.textdata + self.qadata
        
    def disable_qa(self):
        self.enable_qa_tag = False
        self.data = self.textdata
    
    def __len__(self):
        return len(self.data) #if self.enable_qa_tag else self.num_segments
    
    
def LooGLEtrain(qa_pairs, context: str, title: str, tokenizer: PreTrainedTokenizer, model_name_or_path: str, training_args: TrainingArguments, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, use_lora: bool=False, lora_rank: Optional[int]=None, use_pissa: bool=False, load_in_4bit: bool=False, involve_qa_epochs: int=0, gather_batches: bool=True, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_gated_memory: bool=False, use_cot: bool=False, use_icl: bool=True, **kwargs):
    model = load_model(model_name_or_path=model_name_or_path, use_lora=use_lora, lora_rank=lora_rank, use_pissa=use_pissa, load_in_4bit=load_in_4bit, vocab_size=len(tokenizer), use_gated_memory=use_gated_memory)
    
    dataset = LooGLEDataset(qa_pairs, context, title, tokenizer, model_max_length, block_size, len_segment, len_offset, num_syn_qa, title_option, generator_name_or_path, use_cot, use_icl=use_icl)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
    grad = train_visgrad(model, dataset, tokenizer, training_args, 0, gather_batches)
    pred = []
    '''
    for n in grad:
        mean, var = grad[n].abs().mean().item(), grad[n].std(dim=0).mean().item()
        pred.append((n, mean/var, mean, var))
    pred = sorted(pred, key=lambda x: x[2], reverse=True)
    for n, a, b, c in pred:
        print(n, a, b, c)
    '''
    from torch import Tensor
    import matplotlib.pyplot as plt 
    plot = []
    for n in grad:
        if "token" in n:
            continue
        if "norm" in n:
            continue
        fg: Tensor = grad[n].flatten(1)
        fg = fg/(fg.norm(dim=-1, keepdim=True)+1e-4)
        cov = fg@fg.T
        print(n, cov)
        plot.append((n, cov.float().numpy()))
        #plt.matshow(cov, cmap=plt.cm.Reds)
    
    num_matrices = len(plot)
    rows = 24 #int(num_matrices**0.5)  # Number of rows in subplot grid
    cols = (num_matrices + rows-1)//rows  # Number of columns in subplot grid

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))

    # Flatten axes for easy iteration if rows/cols > 1
    axes = axes.ravel()

    # Plot each matrix in a subplot
    for ax, (name, matrix) in zip(axes, plot):
        # Plot the matrix
        cax = ax.matshow(matrix, cmap='viridis', vmin=0, vmax=1)
        
        # Add title
        ax.set_title(name)
        
        # Add colorbar for each subplot (optional)
        fig.colorbar(cax, ax=ax)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.savefig("gradcov.png")
    return grad


def prediction(data: List[Dict], training_args: TrainingArguments, lift_args: Dict, output_file: str, num_resumed: int=0, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_cot: bool=False, use_icl: bool=True):
    tokenizer = load_tokenizer(lift_args['tokenizer_name_or_path'])
    mixin = tokenizer("...", add_special_tokens=False)['input_ids']
    model_max_length = lift_args['model_max_length']
    
    for i, sample in enumerate(tqdm.tqdm(data, desc="Sample")):
        if i < num_resumed:
            continue
        context = (sample['input'], sample['input2'])
        title = sample['title']
        qa_pairs = eval(sample['qa_pairs'])
        model = LooGLEtrain(qa_pairs, context, title, tokenizer, training_args=training_args, num_syn_qa=num_syn_qa, title_option=title_option, generator_name_or_path=generator_name_or_path, use_cot=use_cot, use_icl=use_icl, **lift_args)
        for qa_pair in tqdm.tqdm(qa_pairs, desc="QA Pair"):
            
            if not use_cot:
                input_text = LOOGLEFORMAT_NON_ICL.format(title=title, question=qa_pair['Q'])
            else:
                input_text = LOOGLEFORMAT_COT.format(title=title, question=qa_pair['Q'])
            print(f'input_text: {input_text}')
            input_ids = tokenizer(input_text, add_special_tokens=False)['input_ids']#[:-1]
            print(tokenizer.decode(input_ids, skip_special_tokens=False))
            if len(input_ids) > model_max_length:
                raise NotImplementedError
                input_ids = input_ids[:model_max_length//2 - len(mixin)] + mixin + input_ids[-model_max_length//2:]
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
            gate_mask = torch.ones_like(input_ids)
            #terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            output = model.generate(
                input_ids=input_ids,
                gate_mask=gate_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=1024,
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
        #print(output_case, flush=True)
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
    use_icl = test_args.pop('use_icl')


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
    prediction(input_data, training_args, lift_args, output_file, num_resumed=num_resumed, use_icl=use_icl, **test_args)


if __name__ == '__main__':
    main()
