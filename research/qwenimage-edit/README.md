# Qwen-Image-Edit

This repository provides the inference server support for [Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit), the image editing variant of Qwen-Image.

-----

## ✨ Key Features

* **Consistent Image Editing:** Qwen-Image-Edit achieves exceptional performance in preserving both semantic meaning and visual realism during editing operations.

* **Superior Text Editing:** Capable of editing text in images with high fidelity, including multiline layouts and complex typography.

* **Quality Preservation:** Maintains image quality and style consistency while applying edits.


## 📑 Todo List
- Qwen-Image-Edit (Image Editing Model)
  - [x] Inference server support
  - [x] ComfyUI custom node


## 📦 Requirements

| mindspore | ascend driver | cann               |
| :-------: | :-----------: | :----------------: |
| >=2.7.0   |  >=25.2.0     | >=8.2.RC1          |


## 🚀 Quick Start

### Installation
Clone the repo:
```sh
git clone https://github.com/mindspore-lab/comfyui_mindspore.git
cd comfyui_mindspore/research/qwenimage-edit
```

Download Model Weights:
```bash
# Download from HuggingFace
hf download Qwen/Qwen-Image-Edit
```

### Run Qwen-Image-Edit Inference Server

Try the inference example:
```bash
cd comfyui_mindspore/research/qwenimage-edit/
python infer_app/example_inference.py --input_image path/to/input.png --prompt "Make the image brighter"
```

Start the inference server:
```bash
cd comfyui_mindspore/research/qwenimage-edit/
python infer_app/api.py --model_dir path/to/Qwen-Image-Edit --port 5000
```

When the server is ready, showing "* Running on http://...", enjoy the comfyui_mindspore!
```bash
cd comfyui_mindspore/

# run on ascend 310p, forcing fp16 precision
python main.py --listen 0.0.0.0 --port 9001 --force-fp16 --fp16-vae
```

#### Note:
1. Start a qwenimage-edit server api requires the distributed environment (`4 * ascend 310p`).
2. Wait for all NPUs ready. If you use `4 * ascend 310p`, four urls will be opened for calling.
3. Run comfyui_mindspore, add `AsyncQwenImageEditGenerator` to calling the qwenimage-edit server api.

#### Some configurations you may be interested in `qwenimage_edit_generator.py`:

| configurations          | Description                                                  | Default     |
| ----------------------- | ------------------------------------------------------------ | ----------- |
| `image`                 | Input image to edit (ComfyUI Image tensor)                  | (Required)  |
| `prompt`                | Prompt to guide the image editing                            | (Required)  |
| `num_inference_steps`   | Diffusion infer steps                                        | `50`        |
| `true_cfg_scale`        | Scale for true classifier-free guidance with negative_prompt | `4.0`       |
| `num_workers`           | Number of used NPUs for distributed environment              | `4`         |
| `negative_prompt`       | The prompt not to guide the image editing                    | None        |
| `base_seed`             | Random seed for image editing                                | `42`        |
| `base_port`             | Initial Port for starting server api for calling             | `5000`      |
| `timeout`               | Wait seconds for generating an image (> infer_steps * cost)  | `30000`     |