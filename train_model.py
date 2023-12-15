import argparse
import torch
from functools import partial
from transformers import AutoTokenizer, TrainingArguments, Trainer
from utils.dataset import DialogDataset
from utils.dialog_model import DialogModel
from utils.dialog_utils import Tokens

model_name = 'distilgpt2'
model_max_len = 1024
model = DialogModel(model_name)


dataset = DialogDataset()
tok = AutoTokenizer.from_pretrained(model_name)
tok.add_special_tokens({"additional_special_tokens":[Tokens.CHAR_TOKEN, Tokens.CONTEXT_TOKEN, Tokens.INPUT_TOKEN, Tokens.RESPONSE_TOKEN]})
if tok.pad_token == None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

collate = partial(DialogDataset.collate, tokenizer=tok)

training_args = TrainingArguments(
                                  save_strategy  = "no",
                                  warmup_steps = len(dataset)//2,
                                  logging_steps = 500,
                                  weight_decay = 0.0,
                                  num_train_epochs = 4,
                                  learning_rate = 5e-5,
                                  fp16 = True if torch.cuda.is_available() else False,
                                  ddp_backend = "nccl",
                                  lr_scheduler_type="cosine",
                                  logging_dir = './logs',
                                  output_dir = './results',
                                  per_device_train_batch_size = 2)

trainer = Trainer(model = model,
                  args = training_args,
                  train_dataset= dataset,
                  data_collator=collate).train()


model.save_LM("dialog_model_LM")
