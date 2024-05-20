from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
)
import torch
import re

def get_max_memory_dict():
    max_memory = {}
    max_cpu_memory = '99GiB'
    max_memory['cpu'] = f'{max_cpu_memory}GiB' if not re.match('.*ib$', max_cpu_memory.lower()) else max_cpu_memory

    return max_memory if len(max_memory) > 0 else None

params = {
    'low_cpu_mem_usage': True,
    'trust_remote_code': True,
    'torch_dtype': torch.float32,
    'use_safetensors': True,
    'device_map': 'auto',
    'max_memory': get_max_memory_dict(),
}
config = AutoConfig.from_pretrained('ruDialoGPT-small', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('ruDialoGPT-small', **params)
tokenizer = AutoTokenizer.from_pretrained('ruDialoGPT-small', trust_remote_code=True, use_fast=False, device_map='cpu')
new_utt = ''
mem_string=''
new_utt=input('Юзер: ')
while new_utt != 'quit':
    if new_utt=='`':
        from stt import stt
        print("sst")
        print("="*80)
        stt()
        print("sst done")
        print("="*80)
        f = open("prompt.txt", "r")
        prompt=f.read()
        print(prompt)
        print("="*80)
        new_utt=prompt
    from emotion import emotion
    emo=emotion("../emotion/TinyLlama-Q2_K.gguf", mem_string, "<@@ПЕРВЫЙ@@:\""+new_utt+"\">")
    print(emo)
    print("="*80)
    print("User:", new_utt + "\t("+emo+")")
    mem_string=(mem_string+' @@ПЕРВЫЙ@@ ' + new_utt).strip() + ' @@ВТОРОЙ@@'
    while len(mem_string)>1024:
        mem_string=mem_string[mem_string[10:].index("@@ПЕРВЫЙ@@")+10:]
    inputs = tokenizer(mem_string, return_tensors='pt')
    inputs = inputs.to('cpu')
    generated_token_ids = model.generate(
        **inputs,
        top_k=10,
        top_p=0.95,
        num_beams=30,
        num_return_sequences=1,
        do_sample=True,
        no_repeat_ngram_size=2,
        temperature=1.2,
        repetition_penalty=1.2,
        length_penalty=1.0,
        eos_token_id=50257,
        pad_token_id=50257,
        max_new_tokens=120,
    )
    context_with_response = [tokenizer.decode(sample_token_ids) for sample_token_ids in generated_token_ids]
    print("Бот:", context_with_response[0][len(mem_string):].split("@@ВТОРОЙ@@")[0].rstrip(" @@ПЕРВЫЙ@@"))
    mem_string+=context_with_response[0][len(mem_string):].split("@@ВТОРОЙ@@")[0].rstrip(" @@ПЕРВЫЙ@@")
    new_utt=input('Юзер: ')
print('quit')
