# coding=utf-8
# Copyright 2023 HuggingFace Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gc
import tempfile
import unittest

import numpy as np
import torch

from diffusers import VersatileDiffusionTextToImagePipeline
from diffusers.utils.testing_utils import nightly, require_torch_gpu, torch_device


torch.backends.cuda.matmul.allow_tf32 = False


class VersatileDiffusionTextToImagePipelineFastTests(unittest.TestCase):
    pass


@nightly
@require_torch_gpu
class VersatileDiffusionTextToImagePipelineIntegrationTests(unittest.TestCase):
    def tearDown(self):
        # clean up the VRAM after each test
        super().tearDown()
        gc.collect()
        torch.cuda.empty_cache()

    def test_remove_unused_weights_save_load(self):
        pipe = VersatileDiffusionTextToImagePipeline.from_pretrained("shi-labs/versatile-diffusion")
        # remove text_unet
        pipe.remove_unused_weights()
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger "
        generator = torch.manual_seed(0)
        image = pipe(
            prompt=prompt, generator=generator, guidance_scale=7.5, num_inference_steps=2, output_type="numpy"
        ).images

        with tempfile.TemporaryDirectory() as tmpdirname:
            pipe.save_pretrained(tmpdirname)
            pipe = VersatileDiffusionTextToImagePipeline.from_pretrained(tmpdirname)
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        generator = generator.manual_seed(0)
        new_image = pipe(
            prompt=prompt, generator=generator, guidance_scale=7.5, num_inference_steps=2, output_type="numpy"
        ).images

        assert np.abs(image - new_image).sum() < 1e-5, "Models don't have the same forward pass"

    def test_inference_text2img(self):
        pipe = VersatileDiffusionTextToImagePipeline.from_pretrained(
            "shi-labs/versatile-diffusion", torch_dtype=torch.float16
        )
        pipe.to(torch_device)
        pipe.set_progress_bar_config(disable=None)

        prompt = "A painting of a squirrel eating a burger "
        generator = torch.manual_seed(0)
        image = pipe(
            prompt=prompt, generator=generator, guidance_scale=7.5, num_inference_steps=50, output_type="numpy"
        ).images

        image_slice = image[0, 253:256, 253:256, -1]

        assert image.shape == (1, 512, 512, 3)
        expected_slice = np.array([0.3367, 0.3169, 0.2656, 0.3870, 0.4790, 0.3796, 0.4009, 0.4878, 0.4778])

        assert np.abs(image_slice.flatten() - expected_slice).max() < 1e-2
