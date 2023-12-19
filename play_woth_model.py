
from utils.dialog_utils import Tokens
from utils.dialog_model import DialogModel
from transformers import AutoTokenizer, AutoModelForCausalLM

model = DialogModel('dialog_model_LM_E32', resize_now=False)

tok = AutoTokenizer.from_pretrained('distilgpt2')
tok.add_special_tokens({"additional_special_tokens":[Tokens.CHAR_TOKEN, Tokens.CONTEXT_TOKEN, Tokens.INPUT_TOKEN, Tokens.RESPONSE_TOKEN]})
if tok.pad_token == None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id
tok.model_max_length = 1024


char_name =  input("Welcome to RPGPT, input character name\nUSER>").strip()
bio = input(f"{char_name}? Nice to meet you! Tell me something about your self! Age, Gender, anything is fine\nUSER>").strip()
model.set_character(char_name, bio)

while True:
    args = input("What would you like to do?\nAvailable Commands:\n - exit\n - dialog\n - generate\n - reset context\n - reset character\n\nUSER>").strip()
    if "exit" == args:
        break
    elif "dialog" == args:
        print("Welcome to dialog, you will be prompted to complete dialog. To exit dialog type in EXIT, to reset context type CONTEXT")
        print("Dialog Start")
        while True:
            user_in = input(f"USER>").strip()
            if "EXIT" == user_in:
                break
            elif "CONTEXT" == user_in:
                model.context = []
            else:
                print(f"SYSTEM>{model.generate(tok, user_in)}")
    elif "generate" == args:
        generate_input= input("For how long to generate? DEfaults to 5\nUSER>").strip()
        gens = 5
        if generate_input.isdecimal():
            gens = int(generate_input)
        for _ in range(gens):
            print(f"SYSTEM>{model.generate_self(tok)}")
    elif "reset context" == args:
        model.context = []
    elif "reset character" == args:
        char_name = input("Input new character name.\nUSER>").strip()
        bio = input(f"{char_name}. Input new character information.\nUSER>").strip()
        model.set_character(char_name, bio)
        model.context = []
    
    
    
    
    
print("Thank you for playing with RPGPT!")