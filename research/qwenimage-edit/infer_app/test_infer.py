import argparse
from functools import partial

import numpy as np
from PIL import Image as PILImage

from models.pipeline import QwenImageEditPipelineWrapper
from models.transformer import QwenImageTransformer2DModel
from models.vae import AutoencoderKLQwenImage
from models.text_encoder import Qwen2_5_VLForConditionalGeneration

import mindspore as ms
import mindspore.mint.distributed as dist
from mindspore.communication import GlobalComm

from mindone.trainers.zero import prepare_network


def parse_args():
    parser = argparse.ArgumentParser(description="QwenImage-Edit inference example")
    parser.add_argument(
        "--model_id",
        type=str,
        default="/mnt/disk3/dxw/Qwen-Image-Edit",
        help="The model id in huggingface or the local path of the model's weights.",
    )
    parser.add_argument(
        "--input_image",
        type=str,
        default="/home/dxw/qwenimage/modular/cute_cat.png",
        help="Path to the input image to edit.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="qwenimage_edit_output.png",
        help="The output path where the edited image will be written.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Make the image brighter and more vibrant",
        help="The prompt for image editing.",
    )
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible generation.")

    return parser.parse_args()


def main():
    args = parse_args()

    # dist.init_process_group()
    # ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)

    # local_rank = dist.get_rank()

    # Load components
    transformer = QwenImageTransformer2DModel.from_pretrained(
        args.model_id, subfolder="transformer", mindspore_dtype=ms.bfloat16
    )
    vae = AutoencoderKLQwenImage.from_pretrained(args.model_id, subfolder="vae", mindspore_dtype=ms.bfloat16)
    text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_id, subfolder="text_encoder", mindspore_dtype=ms.bfloat16
    )

    # Create pipeline
    pipe = QwenImageEditPipelineWrapper.from_pretrained(
        args.model_id,
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,
        mindspore_dtype=ms.bfloat16,
    )

    # Apply zero3
    # shard_fn = partial(prepare_network, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
    # pipe.transformer = shard_fn(pipe.transformer)
    # pipe.text_encoder = shard_fn(pipe.text_encoder)

    # dist.barrier()

    # Load input image
    input_image = PILImage.open(args.input_image).convert("RGB")

    # Generate edited image
    edited_image = pipe(
        image=input_image,
        prompt=args.prompt,
        negative_prompt=" ",
        num_inference_steps=50,
        true_cfg_scale=4.0,
        generator=np.random.Generator(np.random.PCG64(seed=args.seed)),
        return_dict=False,
    )[0][0]

    # if local_rank == 0:
    #     edited_image.save(args.output_path)
    #     print(f"Edited image saved to {args.output_path}")


if __name__ == "__main__":
    main()