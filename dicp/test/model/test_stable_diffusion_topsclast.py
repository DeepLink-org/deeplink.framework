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
os.environ["DICP_SD_CLAST"] = "True"


def get_similarity(cpu_image, dicp_image):
    cpu_rgb = list(cpu_image.getdata())
    cpu_gray = [rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114 for rgb in cpu_rgb]
    cpu_gray_tensor = torch.tensor(cpu_gray)

    dicp_rgb = list(dicp_image.getdata())
    dicp_gray = [rgb[0] * 0.299 + rgb[1] * 0.587 + rgb[2] * 0.114 for rgb in dicp_rgb]
    dicp_gray_tensor = torch.tensor(dicp_gray)

    similarity = (torch.abs(torch.sub(cpu_gray_tensor, dicp_gray_tensor)) <= 2).sum() / len(cpu_rgb)
    return similarity


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

        # CPU
        torch.manual_seed(1)
        cpu_pipe = StableDiffusionPipeline.from_pretrained(model_path)
        cpu_image = cpu_pipe(prompt, num_inference_steps=num_inference_steps).images[0]

        # DICP
        torch.manual_seed(1)
        dicp_pipe = StableDiffusionPipeline.from_pretrained(model_path).to(device)
        for name, param in dicp_pipe.unet.named_parameters():
            if "conv" in name and "weight" in name and len(param.shape) == 4:
                data_shape = param.data.shape
                param.data = param.data.permute((2, 3, 1, 0)).contiguous()
                param.data = torch.as_strided(param.data, size=data_shape, stride=(1, data_shape[0], data_shape[0] * data_shape[1] * data_shape[3], data_shape[0] * data_shape[1]))
        dicp_pipe = dicp_pipe.to(device)
        dicp_pipe.text_encoder = torch.compile(dicp_pipe.text_encoder, backend=backend, dynamic=dynamic)
        dicp_pipe.unet = torch.compile(dicp_pipe.unet, backend=backend, dynamic=dynamic)

        dicp_image = dicp_pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        os.environ["DICP_SD_CLAST"] = "False"

        similarity = get_similarity(cpu_image, dicp_image)
        assert similarity > 0.94
