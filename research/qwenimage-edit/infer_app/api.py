import argparse
import base64
import os
import threading
from functools import partial
from io import BytesIO

import numpy as np
from flask import Blueprint, Flask, request
from flask_restful import Api, Resource
from PIL import Image as PILImage

import mindspore as ms
import mindspore.mint.distributed as dist
from mindspore.communication import GlobalComm

from mindone.trainers.zero import prepare_network
from models.pipeline import QwenImageEditPipelineWrapper
from models.transformer import QwenImageTransformer2DModel
from models.vae import AutoencoderKLQwenImage
from models.text_encoder import Qwen2_5_VLForConditionalGeneration


def parsed_args():
    parser = argparse.ArgumentParser(description="QwenImage-Edit API Functions")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    return args


class QwenImageEditAppPipeline(Resource):
    def __init__(self, model_id):
        # Prepare components with given dtype
        transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", mindspore_dtype=ms.float32
        )
        vae = AutoencoderKLQwenImage.from_pretrained(model_id, subfolder="vae", mindspore_dtype=ms.float16)
        text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, subfolder="text_encoder", mindspore_dtype=ms.float16
        )

        self.model = QwenImageEditPipelineWrapper.from_pretrained(
            model_id,
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            mindspore_dtype=ms.float16,
        )

        # Apply zero3
        shard_fn = partial(prepare_network, zero_stage=3, optimizer_parallel_group=GlobalComm.WORLD_COMM_GROUP)
        self.model.transformer = shard_fn(self.model.transformer)
        self.model.text_encoder = shard_fn(self.model.text_encoder)

        # Wait for all NPUs
        dist.barrier()
        print("Loaded QwenImage-Edit pipeline, start to warm up")

        # Warm up with a dummy edit
        dummy_prompt = "Make the image brighter"
        dummy_image = PILImage.new('RGB', (512, 512), color='white')
        self.generate(
            image=dummy_image,
            prompts=dummy_prompt,
            num_inference_steps=12
        )
        print("Finished warm up!")

    def generate(self, image, prompts, *args, **kwargs):
        """
        Generate edited image using the pipeline.
        """
        seed = kwargs.get("seed", 42)
        result = self.model(
            image=image,
            prompt=prompts,
            negative_prompt=kwargs.get("negative_prompt", " "),
            num_inference_steps=kwargs.get("num_inference_steps", 50),
            true_cfg_scale=kwargs.get("true_cfg_scale", 4.0),
            width=kwargs.get("width", None),
            height=kwargs.get("height", None),
            generator=np.random.Generator(np.random.PCG64(seed=seed)),
            return_dict=False,
        )

        return result[0][0]


lock = threading.Lock()


class QwenImageEditAPI(Resource):
    def __init__(self, qwenimage_edit_pipeline):
        self.qwenimage_edit_pipeline = qwenimage_edit_pipeline

    def post(self):
        with lock:
            try:
                data = request.get_json()

                if not data:
                    return {"error": "No data provided"}, 400

                # Decode base64 image
                image_data = data.get("image_data")
                if not image_data:
                    return {"error": "No image_data provided"}, 400

                img_data = base64.b64decode(image_data)
                image = PILImage.open(BytesIO(img_data)).convert("RGB")

                # Extract parameters
                feature = {
                    "prompts": data.get("prompts", ""),
                    "num_inference_steps": data.get("num_inference_steps", 50),
                    "negative_prompt": data.get("negative_prompt", " "),
                    "true_cfg_scale": data.get("true_cfg_scale", 4.0),
                    "width": data.get("width", None),
                    "height": data.get("height", None),
                    "seed": data.get("seed", 42),
                }

                # Generate edited image
                image = self.qwenimage_edit_pipeline.generate(image=image, **feature)

                # Convert to base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                buffered.seek(0)
                img_str = base64.b64encode(buffered.getvalue()).decode()

                response_data = {"status": "success", "image_data": img_str, "format": "png"}

                return response_data, 200

            except Exception as e:
                import traceback
                traceback.print_exc()
                return {"error": str(e)}, 500


class RemoteServer(object):
    def __init__(self, args) -> None:
        self.app = Flask(__name__)
        root = Blueprint("root", __name__)
        self.app.register_blueprint(root)
        api = Api(self.app)

        self.qwenimage_edit_pipeline = QwenImageEditAppPipeline(model_id=os.path.join(args.model_dir))
        api.add_resource(
            QwenImageEditAPI,
            "/qwenimage-edit-api",
            resource_class_args=[self.qwenimage_edit_pipeline],
        )

    def run(self, host="127.0.0.1", port=5000):
        self.app.run(host, port=port, threaded=True, debug=False)


if __name__ == "__main__":
    args = parsed_args()

    ms.set_context(
        mode=ms.PYNATIVE_MODE,
        device_target="Ascend",
        jit_config={"jit_level": "O0"},
        deterministic="ON",
        pynative_synchronize=True,
        memory_optimize_level="O1",
        max_device_memory="59GB",
    )

    dist.init_process_group()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)
    ms.launch_blocking()

    flask_server = RemoteServer(args)
    port_for_rank = int(args.port) + dist.get_rank()
    flask_server.run(host="127.0.0.1", port=port_for_rank)