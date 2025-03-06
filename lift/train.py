import torch
import torch.utils
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
        # print(ret)
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
        trainer.save_model(trainer.args.output_dir)
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
        trainer_syn.save_model(trainer.args.output_dir)
    # Clear cache
    for param in model.parameters():
        if param.requires_grad:
            param.grad = None
    torch.cuda.empty_cache()
    return model, tokenizer
