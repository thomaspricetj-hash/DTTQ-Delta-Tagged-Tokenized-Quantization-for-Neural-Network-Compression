from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
model = AutoModelForCausalLM.from_pretrained('facebook/opt-125m', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained('facebook/opt-125m')
inputs = tokenizer('Hello world', return_tensors='pt')
out = model(**inputs, use_cache=True)
past = out.past_key_values
layer = past.layers[0]
print('keys type', type(layer.keys))
print('keys shape', layer.keys.shape if hasattr(layer.keys, 'shape') else 'N/A')
print('values shape', layer.values.shape if hasattr(layer.values, 'shape') else 'N/A')
