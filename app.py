# gradio 4.16.0
import hashlib
import gradio as gr
import numpy as np
from PIL import Image
import requests
import json
from Outpaint import Outpaint
from utils import pil_to_bs64, bs64_to_pil
import time


def sketches2coordinates(all_layers):
    combined_image = np.zeros_like(all_layers[0])
    for layer in all_layers:
        combined_image += np.array(layer)

    non_transparent_pixels = np.column_stack(np.where(combined_image[:, :, 3] > 0))

    if len(non_transparent_pixels) > 0:
        top_left_corner = non_transparent_pixels.min(axis=0)
        bottom_right_corner = non_transparent_pixels.max(axis=0)

        top_left_coords = (top_left_corner[1], top_left_corner[0])
        bottom_right_coords = (bottom_right_corner[1], bottom_right_corner[0])
        return [top_left_coords, bottom_right_coords]
    else:
        return all_layers[0].size[::-1]


def rgba_to_rgb(img_pil):
    if img_pil.mode == 'RGBA':
        img_pil = img_pil.convert('RGB')
    return img_pil


def apply_mask_numpy(image_np, mask_np):
    if image_np.shape[:2] != mask_np.shape[:2]:
        raise ValueError("Image and mask must have the same shape.")

    if len(mask_np.shape) == 2:
        mask_np = np.expand_dims(mask_np, axis=2)
        mask_np = np.repeat(mask_np, 3, axis=2)

    masked_image_np = image_np * (mask_np / 255)

    return masked_image_np.astype(np.uint8)


def rgb2palette(palette):
    color_image = np.repeat(palette, 32, axis=0)
    color_image = np.tile(color_image[np.newaxis, ...], (32, 1, 1))

    return color_image


def get_result_with_retry(url, headers, get_result_body, max_retries=3, retry_interval=4):
    retries = 0
    print("get image start")
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, data=json.dumps({"body": get_result_body}))
            result_json = json.loads(json.loads(response.text)["body"])
            if "image_b64" in result_json:
                return result_json["image_b64"]
            elif "image_b64_list" in result_json:
                return result_json["image_b64_list"][0]
            else:
                raise Exception("Any Elements in result")
        except Exception as e:
            print(f"An error occurred: {e}")
            retries += 1
            time.sleep(retry_interval)
    raise TimeoutError("Max retries exceeded")


# Function 1 Outpainting
prev_outpaint = Outpaint()
def outpaint(img_pil, mask_pil, checkbox):
    global prev_outpaint
    # same image, check box changed
    if prev_outpaint.img_pil == img_pil and prev_outpaint.mask_pil == mask_pil:
        prev_outpaint.checkbox = checkbox
        print("Same Input. checkbox:", checkbox)
        if checkbox:
            return prev_outpaint.composite_pil
        else:
            return prev_outpaint.result_pil

    img_base64 = pil_to_bs64(img_pil)
    mask_base64 = pil_to_bs64(mask_pil)

    url = "http://192.168.219.114:8000/diffusion/outpaint/"
    headers = {'Content-Type': 'application/json'}

    request_id = hashlib.sha256(img_base64.encode()).hexdigest()
    outpaint_body = {"image_b64":img_base64, "mask_b64":mask_base64, "request_id":request_id}
    response = requests.post(url, headers=headers, data=json.dumps({"body":outpaint_body}))

    url = "http://192.168.219.114:8000/get_result/"
    get_result_body = {"request_id":request_id}

    try:
        result_base64 = get_result_with_retry(url, headers, get_result_body, max_retries=10, retry_interval=5)
        result_pil = bs64_to_pil(result_base64)
        result_pil = result_pil.resize(img_pil.size)

        prev_outpaint.img_pil = img_pil
        prev_outpaint.mask_pil = mask_pil
        prev_outpaint.result_pil = result_pil
        prev_outpaint.product_pil = prev_outpaint.get_product_pil()
        prev_outpaint.composite_pil = prev_outpaint.outpaint_origin_product()
        prev_outpaint.checkbox = checkbox
        
        return prev_outpaint.composite_pil if checkbox else prev_outpaint.result_pil
    except TimeoutError as e:
        print(f"Failed to get result within retries limit: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Function 2 Composition
def composition(img_list, mask_pil):
    img_pil = rgba_to_rgb(img_list["background"])
    img_base64 = pil_to_bs64(img_pil)
    mask_pil = rgba_to_rgb(mask_pil)
    mask_base64 = pil_to_bs64(mask_pil)

    coordinates = sketches2coordinates(img_list["layers"])

    url = 'http://192.168.219.114:8000/diffusion/composition/'
    headers = {'Content-Type': 'application/json'}

    request_id = hashlib.sha256(img_base64.encode()).hexdigest()
    composition_body = {"image_b64":img_base64, "mask_b64":mask_base64, "request_id":request_id}
    response = requests.post(url, headers=headers, data=json.dumps({"body":composition_body}))

    url = "http://192.168.219.114:8000/get_result/"
    get_result_body = {"request_id": request_id}

    try:
        result_base64 = get_result_with_retry(url, headers, get_result_body, max_retries=10, retry_interval=4)
        result_pil = bs64_to_pil(result_base64)
        return result_pil
    except TimeoutError as e:
        print(f"Failed to get result within retries limit: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Function 3-1 Template Augmentation Style
def template_augmentation_style(template_pil, style_pil):
    template_pil = rgba_to_rgb(template_pil)
    template_base64 = pil_to_bs64(template_pil)
    style_pil = rgba_to_rgb(style_pil)
    style_pil = style_pil.resize(template_pil.size)
    style_base64 = pil_to_bs64(style_pil)

    url = "http://192.168.219.114:8000/diffusion/augmentation/style/"
    headers = {"Content-Type": "application/json"}

    request_id = hashlib.sha256(template_base64.encode()).hexdigest()
    augmentation_style_body = {"image_b64_base":template_base64, "image_b64_style":style_base64, "request_id":request_id}
    response = requests.post(url, headers=headers, data=json.dumps({"body":augmentation_style_body}))

    url = "http://192.168.219.114:8000/get_result/"
    get_result_body = {"request_id":request_id}

    try:
        result_base64 = get_result_with_retry(url, headers, get_result_body, max_retries=10, retry_interval=5)
        result_pil = bs64_to_pil(result_base64)
        return result_pil
    except TimeoutError as e:
        print(f"Failed to get result within retries limit: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Function 3-2 Template Augmentation Text
def template_augmentation_text(img_pil, color, concept):
    img_pil = rgba_to_rgb(img_pil)
    img_base64 = pil_to_bs64(img_pil)

    url = "http://192.168.219.114:8000/diffusion/augmentation/text/"
    headers = {'Content-Type': 'application/json'}

    request_id = hashlib.sha256(img_base64.encode()).hexdigest()
    augmentation_text_body = {"image_b64":img_base64, "color":color, "concept":concept, "request_id":request_id}
    response = requests.post(url, headers=headers, data=json.dumps({"body":augmentation_text_body}))
    print(response.text)
    url = "http://192.168.219.114:8000/get_result/"
    get_result_body = {"request_id":request_id}

    try:
        result_base64 = get_result_with_retry(url, headers, get_result_body, max_retries=10, retry_interval=4)
        result_pil = bs64_to_pil(result_base64)
        return result_pil
    except TimeoutError as e:
        print(f"Failed to get result within retries limit: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Function 4-1 Remove Background
def remove_bg(img_list, post_processing):
    img_pil = rgba_to_rgb(img_list["background"])
    img_base64 = pil_to_bs64(img_pil)
    coordinates = sketches2coordinates(img_list["layers"])

    url = 'http://192.168.219.114:8000/utils/remove_bg/'
    headers = {'Content-Type': 'application/json'}

    remove_bg_body = {"image_b64":img_base64, "post_process":post_processing}
    response = requests.post(url, headers=headers, data=json.dumps({"body":remove_bg_body}))

    mask_base64 = json.loads(json.loads(response.text)["body"])["image_b64"].split("base64,")[1]
    mask_pil = bs64_to_pil(mask_base64)

    masked_pil = apply_mask_numpy(np.array(img_pil), np.array(mask_pil))

    return mask_pil, masked_pil


# Function 4-2 Recommend Colors
def color_recommendation(img_pil, mask_pil):
    img_pil = rgba_to_rgb(img_pil)
    img_base64 = pil_to_bs64(img_pil)
    
    mask_base64 = None if mask_pil is None else pil_to_bs64(mask_pil)

    url = 'http://192.168.219.114:8000/utils/recommend_colors/'
    headers = {'Content-Type': 'application/json'}

    recommend_colors_body = {"image_b64":img_base64, "mask_b64": mask_base64}
    response = requests.post(url, headers=headers, data=json.dumps({"body":recommend_colors_body}))
    colors_and_weights = json.loads(json.loads(response.text)["body"])
    
    similar_colors = colors_and_weights["similar_colors"]
    similar_weights = colors_and_weights["similar_weights"]
    similar_palette = rgb2palette(similar_colors)
    creative_colors = colors_and_weights["creative_colors"]
    creative_weights = colors_and_weights["creative_weights"]
    creative_palatte = rgb2palette(creative_colors)
    return (
        similar_colors,
        similar_weights,
        similar_palette,
        creative_colors,
        creative_weights,
        creative_palatte,
    )


# Function 4-3 Super Resolution
def super_resolution(img_pil):
    img_pil = rgba_to_rgb(img_pil)
    img_np = np.array(img_pil)
    if(img_np.shape[0] > 512 and img_np.shape[1] > 512):
        print("Image size too big")
        return None
    
    img_base64 = pil_to_bs64(img_pil)

    url = "http://192.168.219.114:8000/utils/super_resolution/"
    headers = {'Content-Type': 'application/json'}

    request_id = hashlib.sha256(img_base64.encode()).hexdigest()
    super_resolution_body = {"image_b64":img_base64, "request_id":request_id}
    response = requests.post(url, headers=headers, data=json.dumps({"body":super_resolution_body}))

    url = "http://192.168.219.114:8000/get_result/"
    get_result_body = {"request_id": request_id}
    
    try:
        result_base64 = get_result_with_retry(url, headers, get_result_body, max_retries=10, retry_interval=4)
        result_pil = bs64_to_pil(result_base64)
        return result_pil
    except TimeoutError as e:
        print(f"Failed to get result within retries limit: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Function 4-4 Color Enhancement
def color_enhancement(img_pil):
    img_pil = rgba_to_rgb(img_pil)
    img_base64 = pil_to_bs64(img_pil)

    url = 'http://192.168.219.114:8000/utils/color_enhancement/'
    headers = {'Content-Type': 'application/json'}

    color_enhancement_body = {"image_b64":img_base64, "gamma":0.75, "factor":1.2}
    response = requests.post(url, headers=headers, data=json.dumps({"body":color_enhancement_body}))

    result_base64 = json.loads(json.loads(response.text)["body"])["image_b64"]
    result_pil = bs64_to_pil(result_base64)

    return result_pil

width = 500
with gr.Blocks() as demo:
    with gr.Tab("Outpaint"):
        iface1 = gr.Interface(
            fn=outpaint,
            inputs=[
                gr.Image(type="pil", label="Image", width=width),
                gr.Image(type="pil", label="Mask", width=width),
                gr.Checkbox(label="Outpainting Original Product"),
            ],
            outputs=gr.Image(type="pil", label="Result", width=width),
            title="Outpaint",
            allow_flagging="never",
        )

    with gr.Tab("Composition"):
        iface2 = gr.Interface(
            fn=composition,
            inputs=[
                gr.ImageEditor(type="pil", label="Image", width=width),
                gr.Image(type="pil", label="Mask", width=width),
            ],
            outputs=gr.Image(type="pil", label="Result", width=width),
            title="Composition",
            allow_flagging="never",
        )

    with gr.Tab("Augmentation"):
        with gr.Column():
            iface3_1 = gr.Interface(
                fn=template_augmentation_style,
                inputs=[
                    gr.Image(type="pil", label="Template", width=width),
                    gr.Image(type="pil", label="Style", width=width),
                ],
                outputs=gr.Image(type="pil", label="Result", width=width),
                title="Template Augmentation Style",
                allow_flagging="never",
            )

            iface3_2 = gr.Interface(
                fn=template_augmentation_text,
                inputs=[
                    gr.Image(type="pil", label="Template", width=width),
                    gr.Text(label="Color"),
                    gr.Text(label="Concept"),
                ],
                outputs=gr.Image(type="pil", label="Result", width=width),
                title="Template Augmentation Text",
                allow_flagging="never",
            )

    with gr.Tab("Features"):
        with gr.Column():
            iface4_1 = gr.Interface(
                fn=remove_bg,
                inputs=[
                    gr.ImageEditor(type="pil", label="Image", width=width),
                    gr.Checkbox(label="Post Process"),
                ],
                outputs=[gr.Image(type="pil", label="Mask", width=width),
                         gr.Image(type="pil", label="Masked Image", width=width)],
                title="Remove Background",
                allow_flagging="never",
            )

            iface4_2 = gr.Interface(
                fn=color_recommendation,
                inputs=[gr.Image(type="pil", label="Image", width=width),
                        gr.Image(type="pil", label="Mask", width=width)],
                outputs=[
                    gr.Text(label="Similar Colors"),
                    gr.Text(label="Similar Colors Weight"),
                    gr.Image(label="Similar Colors Palette", width=width),
                    gr.Text(label="Creative Colors"),
                    gr.Text(label="Creative Colors Weight"),
                    gr.Image(label="Creative Colors Palette", width=width),
                ],
                title="Recommend Colors",
                allow_flagging="never",
            )

            iface4_3 = gr.Interface(
                fn=super_resolution,
                inputs=gr.Image(type="pil", label="Image", width=width),
                outputs=gr.Image(type="pil", label="Super Resolution", width=1000),
                title="Super Resolution",
                allow_flagging="never",
            )
            iface4_4 = gr.Interface(
                fn=color_enhancement,
                inputs=gr.Image(type="pil", label="Image", width=width),
                outputs=gr.Image(type="pil", label="Enhanced", width=width),
                title="Color Enhancement",
                allow_flagging="never",
            )

demo.launch()
