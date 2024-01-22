# Change this file to change the model and tokenizer used.
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path)
    print("Model loaded")
    return model
    

def get_tokenizer(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    # Set the padding token to be the same as the end-of-sentence (EOS) token for the tokenizer.
    tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")
    return tokenizer

'''
from transformers import BertTokenizer, BertModel

def get_model(model_path):
    model = BertModel.from_pretrained(model_path)
    print("Model loaded")
    return model

def get_tokenizer(model_path):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    print("Tokenizer loaded")
    return tokenizer
'''