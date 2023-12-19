from transformers import AutoModelForCausalLM, AutoTokenizer
from .dialog_utils import Tokens
import torch

class DialogModel(torch.nn.Module):
    def __init__(self, pretrained_model,max_len:int = 1024, resize_now=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model,output_hidden_states=True)
        if resize_now:
            self.model.resize_token_embeddings(50257 + len([Tokens.CHAR_TOKEN, Tokens.CONTEXT_TOKEN, Tokens.INPUT_TOKEN, Tokens.RESPONSE_TOKEN]))
        self.character = f"{Tokens.CHAR_TOKEN} "
        self.max_model_len = max_len
        self.context = []
        
    def forward(self, input_ids=None, labels=None, attention_mask=None, *args, **kwargs):
        outputs = self.model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)
        return {'loss': outputs.loss, "model_output": outputs}
    
    def set_character(self, name, bio):
        self.character = f"{Tokens.CHAR_TOKEN} {name}, Bio: {bio}"
        
    def save_LM(self, LM_path):
        self.model.save_pretrained(LM_path, safe_serialization=False)

    
    def generate(self, tokenizer: AutoTokenizer, prompt):
        length_stop = True
        while length_stop:
            model_input = f"{self.character} {Tokens.CONTEXT_TOKEN} {' '.join(self.context)} {Tokens.INPUT_TOKEN} {prompt}"
            input_ids: torch.Tensor = tokenizer.encode(model_input, return_tensors='pt' )['input_ids']
            if input_ids.size(-1) > self.max_model_len:
                if len(self.context) > 0:
                    self.context.pop(0)
                    length_stop = True
                else:
                    input_ids = input_ids[:,:self.max_model_len]
                    length_stop = False
            else:
                 length_stop = False
        response_out = self.model.generate(input_ids,  
                                    max_new_tokens=  min(self.max_model_len - input_ids.size(-1), 150),
                                    do_sample=True,
                                    top_k=50,
                                    early_stopping=True,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id)
        response:str = tokenizer.decode(response_out[0], skip_special_tokens=False).split(Tokens.RESPONSE_TOKEN)[-1].split(tokenizer.eos_token)[0]
        self.context.append(prompt)
        self.context.append(response)
        return response
    
    def generate_self(self,  tokenizer: AutoTokenizer):
        length_stop = True
        while length_stop:
            model_input = f"{self.character} {Tokens.CONTEXT_TOKEN} {' '.join(self.context)}"
            input_ids: torch.Tensor = tokenizer.encode(model_input, return_tensors='pt' )
            if input_ids.size(-1) > self.max_model_len:
                if len(self.context) > 0:
                    self.context.pop(0)
                    length_stop = True
                else:
                    input_ids = input_ids[:,:self.max_model_len]
                    length_stop = False
            else:
                length_stop = False
        response_out = self.model.generate(input_ids,  
                                    max_new_tokens= min(self.max_model_len - input_ids.size(-1), 150),
                                    do_sample=True,
                                    top_k=50,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id)
        response:str = tokenizer.decode(response_out[0], skip_special_tokens=False).split(Tokens.RESPONSE_TOKEN)[-1].split(tokenizer.eos_token)[0]
        self.context.append(response)
        return response
        