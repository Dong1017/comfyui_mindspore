"""
QwenImage-Edit Generator for ComfyUI - Async API-based custom node.

This node communicates with a QwenImage-Edit inference server to edit images.
"""

import mindspore as ms
import numpy as np
from PIL import Image
from io import BytesIO
import base64
import asyncio
import aiohttp
import concurrent.futures


class AsyncQwenImageEditGenerator:
    """
    ComfyUI node that calls an async QwenImage-Edit inference API.
    Pattern matches: comfyui_mindspore/custom_nodes/qwenimage_generator.py
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Make the image brighter and more vibrant"}),
                "num_inference_steps": ("INT", {"default": 50, "min": 1, "max": 200}),
                "true_cfg_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "num_workers": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "base_seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "base_port": ("INT", {"default": 5000, "min": 1000, "max": 9999}),
                "timeout": ("INT", {"default": 30000, "min": 1000, "max": 300000, "step": 1000}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate_async"
    CATEGORY = "QwenImage"

    def __init__(self):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)

    def _tensor_to_base64(self, image_tensor: ms.Tensor) -> str:
        """
        Convert a ComfyUI image tensor to base64 string.
        Input tensor: (B, H, W, C) with values in [0, 1]
        """
        # Convert to numpy
        if isinstance(image_tensor, ms.Tensor):
            image_np = image_tensor.asnumpy()
        else:
            image_np = np.array(image_tensor)

        # Handle batch dimension
        if len(image_np.shape) == 4:
            image_np = image_np[0]  # Take first image from batch

        # Scale to [0, 255]
        image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

        # Create PIL Image
        image = Image.fromarray(image_np, mode='RGB')

        # Convert to base64
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return img_str

    def call_api_sync(self, url, payload, timeout):
        """Synchronously calling the API in a new thread"""
        try:
            async def _call_api():
                async with aiohttp.ClientSession() as session:
                    aiohttp_timeout = aiohttp.ClientTimeout(total=timeout)

                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp_timeout,
                        headers={'Content-Type': 'application/json'}
                    ) as response:

                        if response.status == 200:
                            result = await response.json()

                            if result.get("status") == "success" and "image_data" in result:
                                img_data = base64.b64decode(result["image_data"])
                                return Image.open(BytesIO(img_data))
                            else:
                                raise Exception(f"API error: {result.get('error', 'Unknown error')}")
                        else:
                            error_text = await response.text()
                            raise Exception(f"HTTP {response.status}: {error_text}")

            # Running asynchronous code in a new thread
            return asyncio.run(_call_api())

        except asyncio.TimeoutError:
            raise Exception(f"Request timeout after {timeout}s")
        except Exception as e:
            raise Exception(f"API call failed: {str(e)}")

    def _generate_parallel_sync(self, prompt, negative_prompt, image_base64,
                                num_inference_steps, true_cfg_scale,
                                num_workers, base_port, base_seed,
                                timeout):
        """Generate images in parallel using multiple workers"""
        # Create worker URLs
        urls = [f"http://127.0.0.1:{base_port + i}/qwenimage-edit-api" for i in range(num_workers)]

        # Payload for each worker
        payloads = []
        for i in range(num_workers):
            payload = {
                "image_data": image_base64,
                "prompts": prompt.strip(),
                "num_inference_steps": num_inference_steps,
                "negative_prompt": negative_prompt.strip() if negative_prompt.strip() else " ",
                "true_cfg_scale": true_cfg_scale,
                "seed": base_seed + i
            }
            payloads.append(payload)

        # Use a thread pool to execute all requests in parallel
        futures = []
        for url, payload in zip(urls, payloads):
            future = self.executor.submit(self.call_api_sync, url, payload, timeout)
            futures.append(future)

        # Collect results
        successful_images = []
        errors = []

        for i, future in enumerate(futures):
            try:
                result = future.result(timeout=timeout)
                if result is not None:
                    successful_images.append(result)
                else:
                    errors.append(f"Worker {i}: No result returned")
            except Exception as e:
                errors.append(f"Worker {i}: {str(e)}")

        return successful_images, errors

    def generate_async(self, image: ms.Tensor, prompt: str,
                      num_inference_steps: int, true_cfg_scale: float,
                      num_workers: int = 4, negative_prompt: str = "",
                      base_seed: int = 42, base_port: int = 5000,
                      timeout: int = 30000):
        try:
            print(f"[QwenEdit] Editing image with prompt: {prompt[:50]}...")

            # Convert input image to base64
            image_base64 = self._tensor_to_base64(image)

            successful_images, errors = self._generate_parallel_sync(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_base64=image_base64,
                num_inference_steps=num_inference_steps,
                true_cfg_scale=true_cfg_scale,
                num_workers=num_workers,
                base_port=base_port,
                base_seed=base_seed,
                timeout=timeout
            )

            if not successful_images:
                error_msg = "All workers failed: " + "; ".join(errors)
                print(f"[QwenEdit] {error_msg}")
                return (self.create_error_tensor(image.shape[1], image.shape[2]), error_msg)

            # Convert to tensor
            image_tensors = []
            for edited_image in successful_images:
                # to RGB
                if edited_image.mode != 'RGB':
                    edited_image = edited_image.convert('RGB')

                # to numpy
                image_np = np.array(edited_image).astype(np.float32) / 255.0

                # (H, W, C) -> (1, H, W, C)
                if len(image_np.shape) == 3:
                    image_tensor = ms.from_numpy(image_np).unsqueeze(0)
                # (B, H, W, C)
                elif len(image_np.shape) == 4:
                    image_tensor = ms.from_numpy(image_np)

                image_tensors.append(image_tensor)

            # Collect tensors
            if len(image_tensors) > 1:
                final_tensor = image_tensors[0]
            else:
                final_tensor = image_tensors[0]

            # Create info string
            info = f"Generated {len(successful_images)} edited images | Steps: {num_inference_steps} | CFG: {true_cfg_scale} | Workers: {num_workers}"
            if errors:
                info += f" | Errors: {len(errors)}"

            return (final_tensor, info)

        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            print(f"[QwenEdit] {error_msg}")
            import traceback
            traceback.print_exc()

            # Get height/width from input image
            height = image.shape[1] if len(image.shape) >= 2 else 512
            width = image.shape[2] if len(image.shape) >= 3 else 512
            return (self.create_error_tensor(height, width), error_msg)

    def create_error_tensor(self, height: int, width: int) -> ms.Tensor:
        """Create an error placeholder tensor"""
        error_image = np.zeros((height, width, 3), dtype=np.float32)
        error_image[:, :, 0] = 1.0  # red channel
        return ms.from_numpy(error_image).unsqueeze(0)


# Register node
NODE_CLASS_MAPPINGS = {
    "AsyncQwenImageEditGenerator": AsyncQwenImageEditGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AsyncQwenImageEditGenerator": "Async Qwen Image Edit Generator",
}

print(f"[QwenEdit] Available nodes: {list(NODE_CLASS_MAPPINGS.keys())}")