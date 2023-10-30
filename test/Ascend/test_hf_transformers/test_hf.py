import torch.jit._shape_functions as shape_functions
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
import torch_dipu
from patch.torch_deepcopy_patch import deepcopy_to_fake_tensor_hf_hook_patched

dipu_device_str = torch_dipu.dipu.device.__diputype__

try:
    import torch._dynamo.config
    torch._dynamo.config.cache_size_limit = 4096
except Exception:
    print("torch._dynamo not found")

pretrained_path = "./llama-7b-hf/"

tokenizer = LlamaTokenizer.from_pretrained(pretrained_path)
model = LlamaForCausalLM.from_pretrained(pretrained_path, device_map="auto", torch_dtype=torch.float16).to(dipu_device_str) # "dipu")

model.generate = torch.compile(model.generate, backend='ascendgraph', dynamic=True)
prompts_list = ["很久很久以前", "在流星划过的天际", "我们的故事要从十年前的一个雨夜说起"]
response_list = []

for prompt in prompts_list:
    tokenized_prompt = tokenizer(prompt, return_tensors="pt").to(dipu_device_str) # "dipu")
    token_promt = tokenized_prompt["input_ids"].to(dipu_device_str)
    print(f"tokenized_prompt: {tokenized_prompt}")
    with deepcopy_to_fake_tensor_hf_hook_patched():
        tokenized_response = model.generate(token_promt, temperature=0.8,
                                            top_k=20, do_sample=True, top_p=0.95,
                                            max_new_tokens=64, repetition_penalty=1.1).cpu()
    print(f"tokenized_response: {tokenized_response}")
    response = tokenizer.decode(tokenized_response[0])
    response_list.append(response)

for idx, response in enumerate(response_list):
    print('Prompt #{}:'.format(idx))
    print(response)
