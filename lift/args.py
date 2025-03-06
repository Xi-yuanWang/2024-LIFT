from dataclasses import dataclass, field
from typing import Optional, Any, List, Union, Literal
from transformers import HfArgumentParser


@dataclass
class ModelArguments:
    model_name_or_path: str
    tokenizer_name_or_path: Optional[str] = field(default=None)
    model_max_length: Optional[int] = field(default=None)
    
    def __post_init__(self):
        if self.tokenizer_name_or_path is None:
            self.tokenizer_name_or_path = self.model_name_or_path


@dataclass
class DataTrainingArguments:
    block_size: Optional[int] = field(
        default=256,
        metadata={'help': "The number of tokens in a block (a block is the unit of segments and offsets)."},
    )
    len_segment: int = field(
        default=8,
        metadata={'help': "The number of blocks in a segment."}
    )
    len_offset: int = field(
        default=3,
        metadata={'help': "The number of blocks in an offset from one segment to the next one."}
    )
    use_random_segment: bool = field(
        default=False,
        metadata={'help': "Randomly sample batches of segments during LIFT."}
    )


@dataclass
class CustomTrainingArguments:
    use_lora: bool = field(
        default=False,
        metadata={'help': "Use LoRA."}
    )
    lora_rank: int = field(
        default=128,
        metadata={'help': "The rank of LoRA adapters."}
    )
    lora_target_modules: List[str] = field(
        default_factory=lambda: ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        metadata={'help': "The target modules of LoRA adapters."}
    )
    use_pissa: bool = field(
        default=False,
        metadata={'help': "Use PiSSA intialization for LoRA. Require model_name_or_path to be a PiSSA checkpoint."}
    )
    use_gated_memory: bool = field(
        default=False,
        metadata={'help': "Use Gated Memory."}
    )
    use_prefix_tuning: bool = field(
        default=False,
        metadata={'help': "Use prefix-tuning."}
    )
    num_virtual_tokens: Optional[int] = field(
        default=None,
        metadata={'help': "The number of learnable tokens in prefix-tuning."}
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={'help': "Load in 4bit."}
    )
    load_in_8bit: bool = field(
        default=False,
        metadata={'help': "Load in 8bit."}
    )
    gather_batches: bool = field(
        default=True,
        metadata={'help': "Update only once per epoch. Implemented with gradient accumulate."}
    )
    involve_qa_epochs: int = field(
        default=0,
        metadata={'help': "The number of epochs of the second LIFT stage (incorporate auxiliary tasks)."}
    )
    
    def __post_init__(self):
        if len(self.lora_target_modules) == 1 and self.lora_target_modules[0] == 'all-linear':
            self.lora_target_modules = 'all-linear'
        assert not self.load_in_8bit, "8-bit loading is not supported yet."
        if self.use_pissa:
            assert self.use_lora, "LoRA must be enabled when using PiSSA."
        assert int(self.use_gated_memory) + int(self.use_lora) <= 1, "LoRA and the gated-memory technique cannot be used simultaneously."
        if self.use_prefix_tuning and self.num_virtual_tokens is None:
            raise ValueError("Use prefix-tuning but '--num_virtual_tokens' is not provided.")


def parse_args(class_clusters: tuple[Any|tuple[Any]], no_dict: tuple[Any], return_config: bool=False):
    class_set = set()
    for cluster in class_clusters:
        if isinstance(cluster, tuple):
            class_set.update(set(cluster))
        else:
            class_set.add(cluster)
    class_tuple = tuple(class_set)
    parser = HfArgumentParser(class_tuple)
    arg_list = parser.parse_args_into_dataclasses()
    arg_dict = {c: a for c, a in zip(class_tuple, arg_list)}
    returns = ()
    for cluster in class_clusters:
        if isinstance(cluster, tuple):
            temp = {}
            for item in cluster:
                temp.update(dict(vars(arg_dict[item])))
            returns += (temp,)
        else:
            if cluster in no_dict:
                returns += (arg_dict[cluster],)
            else:
                returns += (dict(vars(arg_dict[cluster])),)
    if return_config:
        config = {}
        for arg in arg_list:
            config.update({k: v for k, v in dict(vars(arg)).items() if isinstance(v, int|float|bool|str)})
        return returns, config
    return returns
