import torch
import transformers
from transformers import AutoModel, AutoTokenizer
import esm


MODEL_NAME = 'facebook/esm1v_t33_650M_UR90S_1'
SAVE_DIRECTORY = "/models_dl" 

model = transformers.AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)


# Save the tokenizer and model to disk for later reuse
tokenizer.save_pretrained(SAVE_DIRECTORY)
model.save_pretrained(SAVE_DIRECTORY)

