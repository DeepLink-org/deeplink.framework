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

pretrained_path = "/daoxin/llama-7b-hf/"

tokenizer = LlamaTokenizer.from_pretrained(pretrained_path)
model = LlamaForCausalLM.from_pretrained(pretrained_path, device_map='cpu', torch_dtype=torch.float32)
model.generate = torch.compile(model.generate, backend='ascendgraph', dynamic=True)
prompts_list = ["long long ago", "under the sky meteor crossing", "our story started ten years ago"]
response_list = []

for prompt in prompts_list:
    tokenized_prompt = tokenizer(prompt, return_tensors="pt")
    token_promt = tokenized_prompt["input_ids"]
    print(f"tokenized_prompt: {tokenized_prompt}")
    with deepcopy_to_fake_tensor_hf_hook_patched():
        tokenized_response = model.generate(token_promt, temperature=1e-4, # 0.8,
                                            top_k=20, do_sample=True, top_p=0.95,
                                            max_new_tokens=256, repetition_penalty=1.1).cpu()
    print(f"tokenized_response: {tokenized_response}")
    response = tokenizer.decode(tokenized_response[0])
    response_list.append(response)

for idx, response in enumerate(response_list):
    print('Prompt #{}:'.format(idx))
    print(response)

