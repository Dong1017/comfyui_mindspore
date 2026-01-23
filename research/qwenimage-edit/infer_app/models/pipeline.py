# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
#
# This code is adapted from https://github.com/huggingface/diffusers
# with modifications to run diffusers on mindspore.
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

from typing import Optional, Union

import numpy as np
import mindspore as ms

# Import from mindone diffusers for QwenImageEditPipeline
from mindone.diffusers import QwenImageEditPipeline

from models.transformer import QwenImageTransformer2DModel
from models.vae import AutoencoderKLQwenImage
from models.text_encoder import Qwen2_5_VLForConditionalGeneration


class QwenImageEditPipelineWrapper(QwenImageEditPipeline):
    """
    Wrapper for QwenImageEditPipeline that uses locally imported modules.
    This provides compatibility with the comfyui_mindspore research folder structure.
    """

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Load the QwenImageEditPipeline with local model classes.
        """
        # Create component instances first using our local classes
        transformer = kwargs.pop('transformer', None)
        vae = kwargs.pop('vae', None)
        text_encoder = kwargs.pop('text_encoder', None)
        tokenizer = kwargs.pop('tokenizer', None)
        processor = kwargs.pop('processor', None)

        mindspore_dtype = kwargs.pop('mindspore_dtype', ms.bfloat16)

        # Load components if not provided
        if transformer is None:
            transformer = QwenImageTransformer2DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                mindspore_dtype=ms.float32  # Use float32 for transformer
            )

        if vae is None:
            vae = AutoencoderKLQwenImage.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="vae",
                mindspore_dtype=ms.float16
            )

        if text_encoder is None:
            text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="text_encoder",
                mindspore_dtype=mindspore_dtype
            )

        # Import Qwen2Tokenizer and Qwen2VLProcessor if modules not provided
        if tokenizer is None or processor is None:
            from transformers import Qwen2Tokenizer, Qwen2VLProcessor
            if tokenizer is None:
                tokenizer = Qwen2Tokenizer.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder="tokenizer"
                )
            if processor is None:
                processor = Qwen2VLProcessor.from_pretrained(
                    pretrained_model_name_or_path,
                    subfolder="text_encoder"
                )

        # Create the pipeline
        pipeline = cls(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            processor=processor,
            **kwargs
        )

        return pipeline


# Re-export the class for compatibility
__all__ = ["QwenImageEditPipelineWrapper"]