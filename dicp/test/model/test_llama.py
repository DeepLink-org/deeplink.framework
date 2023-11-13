import pytest
import torch
import os
import torch._dynamo as dynamo
from common import utils
import torch_dipu
import json
from pathlib import Path
from model.llama.generation import Tokenizer, LLaMA
from model.llama.model import Transformer, ModelArgs
dynamo.config.cache_size_limit = 128
utils.update_dynamo_config(False)
device = utils.get_device()
torch_dipu.dipu.set_device(device)
models_dir = os.environ.get("LLAMA_MODEL_DIR")
assert models_dir is not None


class ModelPath():
    def __init__(self, ckpt_dir, tokenizer_path) -> None:
        self.ckpt_dir = ckpt_dir
        self.tokenizer_path = tokenizer_path


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    max_seq_len: int,
    max_batch_size: int,
    max_prompt_size: int,
    backend: str,
    dynamic: bool,
) -> LLaMA:
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    ckpt_path = checkpoints[0]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size,
        max_prompt_size=max_prompt_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words

    torch._C._set_default_tensor_type(torch.HalfTensor)
    dicp_model = Transformer(model_args).to(device)
    torch._C._set_default_tensor_type(torch.FloatTensor)
    dicp_model.load_state_dict(checkpoint, strict=False)
    compiled_model = torch.compile(dicp_model, backend=backend, dynamic=dynamic)

    dicp_generator = LLaMA(compiled_model, tokenizer)

    return dicp_generator


class TestLlama():
    @pytest.mark.parametrize("model_path", [ModelPath(f"{models_dir}/target_7B/7B", f"{models_dir}/target_7B/tokenizer.model")])
    def test_inference(
        self,
        model_path: ModelPath,
        backend: str,
        dynamic: bool,
        temperature: float = 0.0,
        top_p: float = 0.95,
        max_seq_len: int = 256,
        max_batch_size: int = 32,
        max_gen_len: int = 128,
        max_prompt_size: int = 32,
    ):
        dicp_generator = load(
            model_path.ckpt_dir, model_path.tokenizer_path, max_seq_len, max_batch_size,
            max_prompt_size, backend, dynamic
        )

        prompts = [
            ["I'm "],
            ["You are pretty"],
            ["What's your name"]
        ]

        cuda_results = [
            ["I'm 20 years old and I'm from the Netherlands. I'm a student of International Business and Management. I'm a very open minded person and I love to travel. I'm a very active person and I love to do sports. I'm a very positive person and I love to make people laugh. I'm a very honest person and I'm very loyal. I'm a very good listener and I'm very good at giving advice. I'm a very good friend and I'm very good at making friends. I'm a very good person to talk to and I'"],

            ["You are pretty much guaranteed to get a good deal on a used car in the UK.",
             "The UK is a great place to buy a used car.",
             "The UK is a great place to buy a used car. The country has a large number of used car dealerships, and the market is very competitive. This means that you are pretty much guaranteed to get a good deal on a used car.",
             "The UK is a great place to buy a used car. The country has a large number of used car dealerships, and the market is very competitive. This means that you are pretty much guaranteed to get a good deal on"],

            ["What's your name? What's your name?",
             "What's your name? What's your name? What's your name?",
             "What's your name? What's your name? What's your name? What's your name?",
             "What's your name? What's your name? What's your name? What's your name? What's your name?",
             "What's your name? What's your name? What's your name? What's your name? What's your name? What's your name?",
             "What's your name? What'"]
        ]

        for i, prompt in enumerate(prompts):
            dicp_result = dicp_generator.generate(
                prompt, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, device=device
            )
            dicp_result = dicp_result[0].split("\n")
            assert dicp_result == cuda_results[i]
