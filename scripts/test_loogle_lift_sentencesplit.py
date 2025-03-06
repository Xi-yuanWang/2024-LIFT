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
from gensim.parsing import remove_stopwords
import string


PUNC_TRANS = str.maketrans(string.punctuation, ' '*len(string.punctuation))
def remove_punctuation(text: str):
    text = text[:1].lower() + text[1:]
    no_punct = text.replace("'s", " ").translate(PUNC_TRANS)
    return no_punct

LIFT_ICL_PROMPT = "<|im_start|>user\nBased on the article <<{title}>>, please recite the most relevant original sentence from article of the following keywords: {keywords}<|im_end|>\n<|im_start|>assistant\nSentence:"
LOOGLEFORMAT_NON_ICL = "<|im_start|>user\nBased on the article <<{title}>>, please answer the following question concisely and accurately.\nQuestion: {question}<|im_end|>\n<|im_start|>assistant\nAnswer:"
LOOGLEFORMAT_COT = "<|im_start|>user\nBased on the article <<{title}>>, please recite the most relevant original sentence from article of the following question: {question}<|im_end|>\n<|im_start|>assistant\nSentence:"

class ICLContextDataset(Dataset):
    """Given a piece of context, `ContextDataset` creates a torch-Dataset, using the truncation strategy described in our paper.
    """
    def __init__(self, title: str, context: str, tokenizer: PreTrainedTokenizer, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3):
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
        texts = context.replace('\0', ' ')
        sents = sent_tokenize(texts)
        self.title = title
        self.data = []
        for s in range(len(sents)):
            if len(sents[s].split()) < 5:
                continue
            prompt = LIFT_ICL_PROMPT#.format(title=title)  
            keywords = list(remove_stopwords(remove_punctuation(sents[s])).split())
            text = self.tokenizer(sents[s].strip() + "<|im_end|>", add_special_tokens=False)['input_ids']
            self.data.append((prompt, keywords, text))

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
        attention_mask = torch.zeros_like(input_ids)
        attention_mask[len_input:] = 1
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
        }
    
    def __getitem__(self, index):
        #print(index)
        if len(self.data[index]) > 2:
            prompt, keywords, text_id = self.data[index]
            import random 
            prompt = prompt.format(title=self.title, keywords=" ,".join(random.sample(keywords, k=random.randint(1, len(keywords))))+".")
            prompt = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
            ret = self.preprocessing((prompt + text_id, len(prompt)))
        else:
            ret = self.preprocessing(self.data[index])
        #print(self.tokenizer.decode(ret["input_ids"], skip_special_tokens=False), flush=True)
        #print(self.tokenizer.decode(ret["labels"][ret["labels"]>=0], skip_special_tokens=False), flush=True)
        return ret
    
    def enable_qa(self):
        raise NotImplementedError
    
    def disable_qa(self):
        raise NotImplementedError
    
    def generate_task(self):
        raise NotImplementedError



#"<|im_start|>user\nBased on the article <<{title}>>, please answer the following question.\nQuestion: {question}.\nLet's think step by step. First, please recite the most relevant three original sentences from the article <<{title}>> as evidence, and then provide an concise answer based on the sentences.<|im_end|>\n<|im_start|>assistant\n"

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
    def __init__(self, context: str, title: str, tokenizer: PreTrainedTokenizer, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_cot: bool=False, use_icl: bool=True):
        # Option 1: prepend title before context
        if title_option == 1:
            context = "Title: " + title + '.\n' + context
        super().__init__(title, context, tokenizer, model_max_length, block_size, len_segment, len_offset)
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
            for _ in range(len(context_sent)):
                if len(context_sent[_].split()) < 5:
                    continue
                result = self.generate_task(generator, gen_tokenizer, context, context_sent[_:_+1], title, model_max_length, use_cot, use_icl=use_icl)
                if result is not None:
                    self.qadata.append(result)
        self.enable_qa_tag = False

    @torch.no_grad()
    def generate_task(self, generator: PreTrainedModel, tokenizer: PreTrainedTokenizer, full_context: str, context_sent: List[str], title: str, model_max_length: int, use_cot: bool=False, use_icl: bool=True):
        context = ' '.join(context_sent)
        messages = [
            {
                'role': "system",
                'content': "You are a helpful assistant."
            },
            {
                'role': "user",
                'content': f"{context}\nGiven the sentence in article <<{title}>> above, please generate a question, whose answer is in the sentence, in the following format: \nQuestion: [question]. Please do not generate other text."
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

            if question_position == -1:
                continue
            question = response[question_position + 9:].strip().split("Note:")[0].strip()
            break
        else:
            logging.warning("Fail to generate a QA pair, skip.")
            return None
        if not use_cot:
            input_text = LOOGLEFORMAT_NON_ICL.format(title=title, question=question)
        else:
            input_text = LOOGLEFORMAT_COT.format(title=title, question=question)
        answer = context
        example = input_text# + ' ' + answer# + self.tokenizer.eos_token
        # print(f'syn input text: {example}', flush=True)
        input_ids = self.tokenizer(example, add_special_tokens=False)['input_ids']
        input_ids = input_ids + self.tokenizer(answer, add_special_tokens=False)['input_ids']
        input_ids = input_ids + [self.tokenizer.eos_token_id]
        input_length = len(self.tokenizer(input_text, add_special_tokens=False)['input_ids'])
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
    
    
def LooGLEtrain(context: str, title: str, tokenizer: PreTrainedTokenizer, model_name_or_path: str, training_args: TrainingArguments, model_max_length: int=4096, block_size: int=256, len_segment: int=8, len_offset: int=3, use_lora: bool=False, lora_rank: Optional[int]=None, use_pissa: bool=False, load_in_4bit: bool=False, involve_qa_epochs: int=0, gather_batches: bool=True, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_gated_memory: bool=False, use_cot: bool=False, use_icl: bool=True, **kwargs):
    model = load_model(model_name_or_path=model_name_or_path, use_lora=use_lora, lora_rank=lora_rank, use_pissa=use_pissa, load_in_4bit=load_in_4bit, vocab_size=len(tokenizer), use_gated_memory=use_gated_memory)
    if use_lora or use_gated_memory:
        dataset = LooGLEDataset(context, title, tokenizer, model_max_length, block_size, len_segment, len_offset, num_syn_qa, title_option, generator_name_or_path, use_cot, use_icl=use_icl)
        model = train(model, dataset, tokenizer, training_args, involve_qa_epochs, gather_batches)[0]
    return model


def prediction(data: List[Dict], training_args: TrainingArguments, lift_args: Dict, output_file: str, num_resumed: int=0, num_syn_qa: int=0, title_option: int=1, generator_name_or_path: Optional[str]=None, use_cot: bool=False, use_icl: bool=True):
    tokenizer = load_tokenizer(lift_args['tokenizer_name_or_path'])
    model_max_length = lift_args['model_max_length']
    
    for i, sample in enumerate(tqdm.tqdm(data, desc="Sample")):
        if i < num_resumed:
            continue
        context = sample['input']
        title = sample['title']
        # qa_pairs = sample['test_qa_pairs']
        qa_pairs = eval(sample['qa_pairs'])
        #from copy import deepcopy
        #tmp = deepcopy(qa_pairs[-1])
        #tmp["Q"] = "Where is the capital of China?"
        #qa_pairs.append(tmp) 
        #tmp = deepcopy(qa_pairs[-1])
        #tmp["Q"] = "Who is Xiyuan Wang?"
        #qa_pairs.append(tmp)
        model = LooGLEtrain(context, title, tokenizer, training_args=training_args, num_syn_qa=num_syn_qa, title_option=title_option, generator_name_or_path=generator_name_or_path, use_cot=use_cot, use_icl=use_icl, **lift_args)
        model.eval()
        for qa_pair in tqdm.tqdm(qa_pairs, desc="QA Pair"):
            
            if not use_cot:
                input_text = LOOGLEFORMAT_NON_ICL.format(title=title, question=qa_pair['Q'])
            else:
                input_text = LOOGLEFORMAT_COT.format(title=title, question=qa_pair['Q'])
            input_ids = tokenizer(input_text, add_special_tokens=False)['input_ids']#[:-1]
            print(tokenizer.decode(input_ids, skip_special_tokens=False))
            if len(input_ids) > model_max_length:
                raise NotImplementedError
                input_ids = input_ids[:model_max_length//2 - len(mixin)] + mixin + input_ids[-model_max_length//2:]
            len_input = len(input_ids)
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
            attention_mask = torch.zeros_like(input_ids)
            #terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=1024,
                use_cache=True,
                do_sample=False,
            )
            response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=False)
            qa_pair['pred'] = response
        '''
        qa_pairs2 = deepcopy(qa_pairs)        
        for qa_pair in tqdm.tqdm(qa_pairs2, desc="QA Pair"):
            keywords = list(remove_stopwords(remove_punctuation(qa_pair["Q"])).split())             
            input_text = LIFT_ICL_PROMPT.format(title=title, keywords=" ,".join(keywords)+".")
            print(f'input_text: {input_text}')
            input_ids = tokenizer(input_text, add_special_tokens=False)['input_ids']#[:-1]
            print(tokenizer.decode(input_ids, skip_special_tokens=False))
            if len(input_ids) > model_max_length:
                raise NotImplementedError
                input_ids = input_ids[:model_max_length//2 - len(mixin)] + mixin + input_ids[-model_max_length//2:]
            len_input = len(input_ids)
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=model.device).unsqueeze(0)
            attention_mask = torch.zeros_like(input_ids)
            #terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=1024,
                use_cache=True,
                do_sample=False,
            )
            response = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=False)
            qa_pair['pred'] = response
        qa_pairs = qa_pairs + qa_pairs2
        '''
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
