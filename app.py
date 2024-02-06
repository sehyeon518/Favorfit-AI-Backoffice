# gradio 4.16.0
import gradio as gr
import numpy as np
from PIL import Image
import requests
import json
from utils import pil_to_bs64, bs64_to_pil


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
    image_size = (len(palette), 1)
    image = Image.new("RGB", image_size)
    image.putdata([tuple(color) for color in palette])
    return image


def checkbox_boolean(checkbox_value):
    return checkbox_value


# Function 1 Outpainting
def outpaint(img_pil, mask_pil):
    img_base64 = pil_to_bs64(img_pil)
    mask_base64 = pil_to_bs64(mask_pil)

    # TODO: Outpainting
    result_base64 = img_base64
    result_pil = bs64_to_pil(result_base64)

    return result_pil


# Function 2 Composition
def composition(img_pil, mask_pil):
    img_base64 = pil_to_bs64(img_pil["background"])
    coordinates = sketches2coordinates(img_pil["layers"])
    print(coordinates)
    mask_base64 = pil_to_bs64(mask_pil)

    # TODO: Composition
    result_base64 = None
    # result_pil = base2pil(result_base64)

    return img_pil


# Function 3-1 Template Augmentation Style
def template_augmentation_style(img_pil, template_pil):
    img_base64 = pil_to_bs64(img_pil)
    template_base64 = pil_to_bs64(template_pil)

    # TODO: Template Augmentation Style
    result_base64 = None
    result_pil = bs64_to_pil(result_base64)
    return result_pil


# Function 3-2 Template Augmentation Text
def template_augmentation_text(img_pil, color, concept):
    img_base64 = pil_to_bs64(img_pil)

    # TODO: Template Augmentation Text
    result = None
    return


# Function 4-1 Remove Background
def remove_bg(img_list, post_processing):
    img_pil = rgba_to_rgb(img_list["background"])
    img_base64 = pil_to_bs64(img_list["background"].convert("RGB"))
    coordinates = sketches2coordinates(img_list["layers"])

    url = 'http://192.168.219.114:8000/utils/remove_bg/'
    headers = {'Content-Type': 'application/json'}

    remove_bg_body = {"image_b64":img_base64, "post_process":post_processing}
    response = requests.post(url, headers=headers, data=json.dumps({"body":remove_bg_body}))

    mask_base64 = json.loads(json.loads(response.text)["body"])["image_b64"].split("base64,")[1]
    mask_pil = bs64_to_pil(mask_base64)

    masked_pil = apply_mask_numpy(np.array(img_pil), np.array(mask_pil))

    return mask_pil, masked_pil


# Function 4-2 Recommend Color
def color_recommendation(img_pil):
    img_base64 = pil_to_bs64(img_pil)

    # TODO: Recommend Color
    similar_colors = None
    creative_colors = None
    similar_weights = None
    creative_weights = None
    similar_palette = rgb2palette(similar_colors)
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
    img_base64 = pil_to_bs64(img_pil)

    # TODO: Super Resolution
    result_base64 = None
    result_pil = bs64_to_pil(result_base64)

    return result_pil


# Function 4-4 Color Enhancement
def color_enhancement(img_pil):
    result_np = None
    return result_np


with gr.Blocks() as demo:
    with gr.Tab("Outpaint"):
        iface1 = gr.Interface(
            fn=outpaint,
            inputs=[
                gr.Image(type="pil", label="Image"),
                gr.Image(type="pil", label="Mask"),
            ],
            outputs=gr.Image(type="pil", label="Result"),
            title="Outpaint",
            allow_flagging="never",
        )

    with gr.Tab("Composition"):
        iface2 = gr.Interface(
            fn=composition,
            inputs=[
                gr.ImageEditor(type="pil", label="Image"),
                gr.Image(type="pil", label="Mask"),
            ],
            outputs=gr.Image(type="pil", label="Result"),
            title="Composition",
            allow_flagging="never",
        )

    with gr.Tab("Augmentation"):
        with gr.Column():
            iface3_1 = gr.Interface(
                fn=template_augmentation_style,
                inputs=[
                    gr.Image(type="pil", label="Template"),
                    gr.Image(type="pil", label="Style"),
                ],
                outputs=gr.Image(type="pil", label="Result"),
                title="Template Augmentation Style",
                allow_flagging="never",
            )

            iface3_2 = gr.Interface(
                fn=template_augmentation_text,
                inputs=[
                    gr.Image(type="pil", label="Template"),
                    gr.Text(label="Color"),
                    gr.Text(label="Concept"),
                ],
                outputs=gr.Text(label="Result"),
                title="Template Augmentation Text",
                allow_flagging="never",
            )

    with gr.Tab("Features"):
        with gr.Column():
            iface4_1 = gr.Interface(
                fn=remove_bg,
                inputs=[
                    gr.ImageEditor(type="pil", label="Image"),
                    gr.Checkbox(label="Post Process"),
                ],
                outputs=[gr.Image(type="pil", label="Mask"),
                         gr.Image(type="pil", label="Masked Image")],
                title="Remove Background",
                allow_flagging="never",
            )

            iface4_2 = gr.Interface(
                fn=color_recommendation,
                inputs=gr.Image(type="pil", label="Image"),
                outputs=[
                    gr.Text(label="Similar Colors"),
                    gr.Text(label="Similar Colors Weight"),
                    gr.Image(label="Similar Colors Palette"),
                    gr.Text(label="Creative Colors"),
                    gr.Text(label="Creative Colors Weight"),
                    gr.Image(label="Creative Colors Palette"),
                ],
                title="Color Recommendation",
                allow_flagging="never",
            )

            iface4_3 = gr.Interface(
                fn=super_resolution,
                inputs=gr.Image(type="pil", label="Image"),
                outputs=gr.Image(type="pil", label="Super Resolution"),
                title="Super Resolution",
                allow_flagging="never",
            )
            iface4_4 = gr.Interface(
                fn=color_enhancement,
                inputs=gr.Image(type="pil", label="Image"),
                outputs=gr.Image(type="pil", label="Enhanced"),
                title="Color Enhancement",
                allow_flagging="never",
            )

demo.launch()
