# gradio 4.16.0
import hashlib
import gradio as gr
import numpy as np
from PIL import Image
import requests
import json
from utils import composing_output, pil_to_bs64, bs64_to_pil
import time


def sketches2coordinates(all_layers):
    combined_image = np.zeros_like(resize512(all_layers[0]))
    for layer in all_layers:
        layer = resize512(layer)
        combined_image += np.array(layer)

    non_transparent_pixels = np.column_stack(np.where(combined_image[:, :, 3] > 0))

    if len(non_transparent_pixels) > 0:
        top_left_corner = non_transparent_pixels.min(axis=0).tolist()
        bottom_right_corner = non_transparent_pixels.max(axis=0).tolist()

        return [top_left_corner[1], top_left_corner[0], bottom_right_corner[1], bottom_right_corner[0]]
    else:
        bottom_right = all_layers[0].size
        return [0, 0, bottom_right[0], bottom_right[1]]


def rgba_to_rgb(img_pil):
    if img_pil.mode == 'RGBA':
        img_pil = img_pil.convert('RGB')
    return img_pil


def rgb_to_rgba(img_pil):
    if img_pil.mode == "RGB":
        data = img_pil.getdata()
        all_gray = all(r == g == b for r, g, b in data)
        if all_gray:
            alpha_pixels = [(r, g, b, r) for r, g, b in data]
            img_pil = Image.new("RGBA", img_pil.size)
            img_pil.putdata(alpha_pixels)
        else:
            img_pil = img_pil.convert("RGBA")
    return img_pil


def reverse_mask(img_pil):
    mask = np.array(img_pil)
    reverse = 255 - mask

    return Image.fromarray(reverse).convert("RGB")


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


def resize512(img_pil):
    width, height = img_pil.size

    if width < height:
        new_width = 512
        new_height = int(height * (512 / width))
    else:
        new_height = 512
        new_width = int(width * (512 / height))

    resized_img = img_pil.resize((new_width, new_height))
    return resized_img


def get_result_with_retry(url, headers, get_result_body, max_retries=3, retry_interval=4):
    retries = 0
    print("get image start")
    while retries < max_retries:
        try:
            response = requests.post(url, headers=headers, data=json.dumps({"body": get_result_body}))
            result_json = json.loads(json.loads(response.text)["body"])
            if "image_b64" in result_json:
                print("Request Succeed!")
                return result_json["image_b64"]
            elif "image_b64_list" in result_json:
                print("Request Succeed!")
                return result_json["image_b64_list"][0]
            else:
                raise Exception("Any Elements in result")
        except Exception as e:
            print(f"An error occurred: {e}")
            retries += 1
            time.sleep(retry_interval)
    raise TimeoutError("Max retries exceeded")


# Function 1 Outpainting
def outpaint(img_pil, mask_pil, checkbox):
    img_pil = resize512(img_pil)
    if img_pil.size != mask_pil.size:
        mask_pil = mask_pil.resize(img_pil.size)
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
        result_pil_rgba = rgb_to_rgba(result_pil)
        img_pil_rgba = rgb_to_rgba(img_pil)
        mask_pil_rgba = rgb_to_rgba(mask_pil)

        img_pil_rgba = img_pil_rgba.resize(result_pil_rgba.size)
        mask_pil_rgba = mask_pil_rgba.resize(result_pil_rgba.size)

        gray = mask_pil_rgba.convert("L")

        r, g, b, a = img_pil_rgba.split()
        product_pil = Image.merge("RGBA", (r, g, b, gray))

        composite_pil = composing_output(result_pil_rgba, product_pil, mask_pil_rgba)
       
        if checkbox:
            return composite_pil, result_pil, gr.Checkbox(label="Outpainting Original Product", visible=True, value=True)
        return result_pil, composite_pil, gr.Checkbox(label="Outpainting Original Product", visible=True, value=False)
    except TimeoutError as e:
        print(f"Failed to get result within retries limit: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def switch_origin_product(img_a, img_b):
    return img_b, img_a


# Function 2 Composition
def composition(img_pil, mask_pil, checkbox):
    img_pil = rgba_to_rgb(img_pil)
    mask_pil = rgba_to_rgb(mask_pil)
    img_pil = resize512(img_pil)
    mask_pil = resize512(mask_pil)
    mask_pil_reverse = reverse_mask(mask_pil)

    if img_pil.size != mask_pil:
        mask_pil = mask_pil.resize(img_pil.size)

    img_base64 = pil_to_bs64(img_pil)
    mask_base64 = pil_to_bs64(mask_pil_reverse)

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
        result_pil_rgba = rgb_to_rgba(result_pil)
        img_pil_rgba = rgb_to_rgba(img_pil)
        mask_pil_rgba = rgb_to_rgba(mask_pil)

        img_pil_rgba = img_pil_rgba.resize(result_pil_rgba.size)
        mask_pil_rgba = mask_pil_rgba.resize(result_pil_rgba.size)

        gray = mask_pil_rgba.convert("L")

        r, g, b, a = img_pil_rgba.split()
        product_pil = Image.merge("RGBA", (r, g, b, gray))

        composite_pil = composing_output(result_pil_rgba, product_pil, mask_pil_rgba)

        if checkbox:
            return composite_pil, result_pil, gr.Checkbox(label="Composition Original Product", visible=True, value=True)
        return result_pil, composite_pil, gr.Checkbox(label="Composition Original Product", visible=True, value=False)
    except TimeoutError as e:
        print(f"Failed to get result within retries limit: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Function 3-1 Template Augmentation Style
def template_augmentation_style(template_pil, style_pil):
    template_pil = rgba_to_rgb(template_pil)
    template_pil = resize512(template_pil)
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
    img_pil = resize512(img_pil)
    img_base64 = pil_to_bs64(img_pil)
    coordinates = sketches2coordinates(img_list["layers"])

    url = 'http://192.168.219.114:8000/utils/remove_bg/'
    headers = {'Content-Type': 'application/json'}

    remove_bg_body = {"image_b64":img_base64, "post_process":post_processing, "box":coordinates}
    response = requests.post(url, headers=headers, data=json.dumps({"body":remove_bg_body}))

    mask_base64 = json.loads(json.loads(response.text)["body"])["image_b64"].split("base64,")[1]
    mask_pil = bs64_to_pil(mask_base64)

    masked_pil = apply_mask_numpy(np.array(img_pil), np.array(mask_pil))
    return mask_pil, masked_pil


# Function 4-2 Recommend Colors
def color_recommendation(img_pil, mask_pil):
    img_pil = rgba_to_rgb(img_pil)
    
    if img_pil.size != mask_pil.size:
        img_pil = resize512(img_pil)
        mask_pil = mask_pil.resize(img_pil.size)
    else:
        img_pil = resize512(img_pil)
        mask_pil = resize512(img_pil)
    
    img_base64 = pil_to_bs64(img_pil)
    
    mask_base64 = None if mask_pil is None else pil_to_bs64(resize512(mask_pil))

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
    # img_pil = resize512(img_pil)
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
        with gr.Blocks():
            gr.Markdown("""# Outpaint""")
            with gr.Accordion("""Outpaint""", open=False):
                gr.Markdown(
                    """
                    ![outpaint](https://github.com/sehyeon518/Favorfit-Gradio/assets/84698896/8bc3ce6f-15dc-462b-a6c8-7702f23ba317)
                    """
                )
            with gr.Row():
                with gr.Column():
                    img_pil = gr.Image(type="pil", label="Image", width=width)
                    mask_pil = gr.Image(type="pil", label="Mask", width=width)
                    submit = gr.Button(value="Submit", variant="primary")
                with gr.Column():
                    result_pil = gr.Image(type="pil", label="Result", width=width, interactive=False)
                    checkbox = gr.Checkbox(label="Outpainting Original Product", visible=False)
                    substitute = gr.Image(type="pil", label="substitute", visible=False)
            submit.click(outpaint, [img_pil, mask_pil, checkbox], [result_pil, substitute, checkbox])
            checkbox.select(switch_origin_product, [result_pil, substitute], [result_pil, substitute])

    with gr.Tab("Composition"):
        with gr.Blocks():
            gr.Markdown("""# Composition""")
            with gr.Accordion("""Composition""", open=False):
                gr.Markdown(
                    """
                    ![composition](https://github.com/sehyeon518/Favorfit-Gradio/assets/84698896/60f15213-03de-458c-bad4-038f8080a1c9)
                    """
                )
            with gr.Row():
                with gr.Column():
                    img_pil = gr.Image(type="pil", label="Image", width=width)
                    mask_pil = gr.Image(type="pil", label="Mask", width=width)
                    submit = gr.Button(value="Submit", variant="primary")
                with gr.Column():
                    result_pil = gr.Image(type="pil", label="Result", width=width, interactive=False)
                    checkbox = gr.Checkbox(label="Composition Original Product", visible=False)
                    substitute = gr.Image(type="pil", label="substitute", visible=False)
            submit.click(composition, [img_pil, mask_pil, checkbox], [result_pil, substitute, checkbox])
            checkbox.select(switch_origin_product, [result_pil, substitute], [result_pil, substitute])

    with gr.Tab("Augmentation"):
        with gr.Column():
            gr.Markdown("""# Template Augmentation Style""")
            with gr.Accordion("""Template Augmentation - Style""", open=False):
                gr.Markdown(
                    """
                    ![augmentation_style](https://github.com/sehyeon518/Favorfit-Gradio/assets/84698896/aeaf3259-e393-4d9d-8471-422bf125c89b)
                    """
                )     
            with gr.Row():
                with gr.Column():
                    img_pil = gr.Image(type="pil", label="Template", width=width)
                    style_pil = gr.Image(type="pil", label="Style", width=width)
                    submit = gr.Button(value="Submit", variant="primary")
                with gr.Column():
                    result_pil = gr.Image(type="pil", label="Result", width=width)
                submit.click(template_augmentation_style, [img_pil, style_pil], result_pil)
            
            gr.Markdown("""# Template Augmentation Text""")
            with gr.Accordion("""Template Augmentation - Text""", open=False):
                gr.Markdown(
                    """
                    ![augmentation_text](https://github.com/sehyeon518/Favorfit-Gradio/assets/84698896/6cf96628-b5a1-466f-a82c-707a93cf0fbc)
                    """
                )
            with gr.Row():
                with gr.Column():
                    img_pil = gr.Image(type="pil", label="Template", width=width)
                    color_txt = gr.Text(label="Color")
                    concept_txt = gr.Text(label="Concept")
                    submit = gr.Button(value="Submit", variant="primary")
                with gr.Column():
                    result_pil = gr.Image(type="pil", label="Result", width=width)
                submit.click(template_augmentation_text, [img_pil, color_txt, concept_txt], result_pil)

    with gr.Tab("Features"):
        with gr.Column():
            gr.Markdown("""# Remove Background""")
            with gr.Accordion("""Remove Background""", open=False):
                gr.Markdown(
                    """
                    ![remove_background](https://github.com/sehyeon518/Favorfit-Gradio/assets/84698896/9dad4f24-2dbe-4c26-8b7c-f9e76b7daae3)
                    """
                )
            with gr.Row():
                with gr.Column():
                    img_pil = gr.ImageEditor(type="pil", label="Image", width=width)
                    checkbox = gr.Checkbox(label="Post Process")
                    submit = gr.Button(value="Submit", variant="primary")
                with gr.Column():
                    mask_pil = gr.Image(type="pil", label="Mask", width=width)
                    masked_pil = gr.Image(type="pil", label="Masked Image", width=width)
                submit.click(remove_bg, [img_pil, checkbox], [mask_pil, masked_pil])

            gr.Markdown("""# Recommend Colors""")
            with gr.Accordion("""Recommend Colors""", open=False):
                gr.Markdown(
                    """
                    ![recommend_colors](https://github.com/sehyeon518/Favorfit-Gradio/assets/84698896/8b15b971-8ab6-4dd0-b4e5-179a7b7d2444)
                    """
                )
            with gr.Row():
                with gr.Column():
                    img_pil = gr.Image(type="pil", label="Image", width=width)
                    mask_pil = gr.Image(type="pil", label="Mask", width=width)
                    submit = gr.Button(value="Submit", variant="primary")
                with gr.Column():
                    similar_colors = gr.Text(label="Similar Colors")
                    similar_weights = gr.Text(label="Similar Colors Weight")
                    similar_palette = gr.Image(label="Similar Colors Palette", width=width)
                    creative_colors = gr.Text(label="Creative Colors")
                    creative_weights = gr.Text(label="Creative Colors Weight")
                    creative_palette = gr.Image(label="Creative Colors Palette", width=width)
                submit.click(color_recommendation, [img_pil, mask_pil], [similar_colors, similar_weights, similar_palette, creative_colors, creative_weights, creative_palette])

            gr.Markdown("""# Super Resolution""")
            with gr.Accordion("""Super Resolution""", open=False):
                gr.Markdown(
                    """
                    ![super_resolution](https://github.com/sehyeon518/Favorfit-Gradio/assets/84698896/71595e68-e03f-492b-8e2c-a4df6e87a6f6)
                    """
                )
            with gr.Row():
                with gr.Column():
                    img_pil = gr.Image(type="pil", label="Image", width=width)
                    submit = gr.Button(value="Submit", variant="primary")
                with gr.Column():
                    result_pil = gr.Image(type="pil", label="Super Resolution", width=1000)
                submit.click(super_resolution, img_pil, result_pil)

            gr.Markdown("""# Color Enhancement""")
            with gr.Accordion("""Color Enhancement""", open=False):
                gr.Markdown(
                    """
                    <img src="https://github.com/sehyeon518/Favorfit-Gradio/assets/84698896/f6215a13-a837-4749-ba55-991d132023af" width="1000">
                    """
                )
            with gr.Row():
                with gr.Column():
                    img_pil = gr.Image(type="pil", label="Image", width=width)
                    submit = gr.Button(value="Submit", variant="primary")
                with gr.Column():
                    result_pil = gr.Image(type="pil", label="Enhanced", width=width)
                submit.click(color_enhancement, img_pil, result_pil)

demo.launch()
