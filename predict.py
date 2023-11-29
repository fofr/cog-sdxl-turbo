import os
import time
import torch
from typing import List
from cog import BasePredictor, Input, Path
from diffusers import AutoPipelineForText2Image
from weights_downloader import WeightsDownloader

SDXL_MODEL_CACHE = "./sdxl-cache"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl-turbo/sd_xl_turbo_fp16.tar"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        start = time.time()
        WeightsDownloader.download_if_not_exists(SDXL_URL, SDXL_MODEL_CACHE)

        print("Loading sdxl turbo txt2img pipeline...")
        self.txt2img_pipe = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/sdxl-turbo",
            cache_dir=SDXL_MODEL_CACHE,
            torch_dtype=torch.float16,
            use_safetensors=True,
            watermark=None,
            safety_checker=None,
            variant="fp16",
        )

        self.txt2img_pipe.to("cuda")
        print("setup took: ", time.time() - start)

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default="An astronaut riding a rainbow unicorn",
        ),
        agree_to_research_only: bool = Input(
            description="You must agree to use this model only for research. It is not for commercial use.",
            default=False,
        ),
        width: int = Input(
            description="Width of output image",
            default=512,
        ),
        height: int = Input(
            description="Height of output image",
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output",
            ge=1,
            le=30,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=4, default=1
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        start = time.time()

        """Run a single prediction on the model."""
        if not agree_to_research_only:
            raise Exception(
                "You must agree to use this model for research-only, you cannot use this model comercially."
            )

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")
        print(f"Prompt: {prompt}")

        inference_start = time.time()
        output = self.txt2img_pipe(
            prompt=[prompt] * num_outputs,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            guidance_scale=0.0,
            generator=torch.Generator("cuda").manual_seed(seed),
        )
        print(f"Inference took: {time.time() - inference_start}")
        output_paths = []

        for i, image in enumerate(output.images):
            output_path = f"/tmp/out-{i}.jpg"
            image.save(output_path)
            output_paths.append(Path(output_path))

        print(f"Prediction took: {time.time() - start}")
        return output_paths
