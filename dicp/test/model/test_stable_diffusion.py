import pytest
import torch
import os
import torch._dynamo as dynamo
from ..common import utils
import torch_dipu
from diffusers import StableDiffusionPipeline
dynamo.config.cache_size_limit = 128
utils.update_dynamo_config(False)
device = utils.get_device()
torch_dipu.dipu.set_device(device)
models_dir = os.environ.get("STABLE_DIFFUSION_MODEL_DIR")
assert models_dir is not None


class TestStableDiffusion():
    @pytest.mark.parametrize("model_path", [f"{models_dir}/stable-diffusion-2"])
    @pytest.mark.parametrize("num_inference_steps", [50])
    def test_inference(
        self,
        model_path: str,
        backend: str,
        dynamic: bool,
        num_inference_steps: int
    ):
        prompt = "A photo of an astronaut riding a horse on mars."
        utils.update_dynamo_config(dynamic=dynamic)
        torch_dipu.dipu.set_device(device)

        dicp_pipe = StableDiffusionPipeline.from_pretrained(model_path).to(device)
        dicp_pipe.text_encoder = torch.compile(dicp_pipe.text_encoder, backend=backend)
        dicp_pipe.unet = torch.compile(dicp_pipe.unet, backend=backend)
        dicp_image = dicp_pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        if backend == "ascendgraph":
            with open("stable_diffusion/topsgraph_output.txt", "r") as f:
                standard_output = eval(f.read())
        elif backend == "topsgraph":
            with open("stable_diffusion/topsgraph_output.txt", "r") as f:
                standard_output = eval(f.read())
        else:
            raise ValueError("backend should in (ascendgrap, topsgraph)")
        dicp_output = list(dicp_image.getdata())
        assert dicp_output == standard_output
