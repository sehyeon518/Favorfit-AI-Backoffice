import gradio as gr
import numpy as np
from PIL import Image
import base64
from io import BytesIO


def base2pil(base64_string):
    image_bytes = base64.b64decode(base64_string)
    image_pil = Image.open(BytesIO(image_bytes))
    return image_pil


def pil2base(image_pil):
    image_bytes = BytesIO()
    image_pil.save(image_bytes, format="JPEG")

    base64_string = base64.b64decode(image_bytes.getvalue()).decode("utf-8")
    return base64_string


# Function 1 Outpainting
def outpaint(img_pil, mask_pil):
    img_base64 = pil2base(img_pil)
    mask_base64 = pil2base(mask_pil)

    # TODO: Outpainting
    result_base64 = None
    result_pil = base2pil(result_base64)

    return result_pil


iface1 = gr.Interface(
    fn=outpaint,
    inputs=[gr.Image(type="pil", label="image"), gr.Image(type="pil", label="mask")],
    outputs=gr.Image(type="pil", label="result"),
)


# Function 2 Composition
def composition(img_pil, mask_pil):
    img_base64 = pil2base(img_pil)
    mask_base64 = pil2base(mask_pil)

    # TODO: Composition
    result_base64 = None
    result_pil = base2pil(result_base64)

    return result_pil


iface2 = gr.Interface(
    fn=composition,
    inputs=[gr.Image(type="pil", label="image"), gr.Image(type="pil", label="mask")],
    outputs=gr.Image(type="pil", label="result"),
)


# Function 3-1 Template Augmentation Style
def template_augmentation_style(img_pil, template_pil):
    img_base64 = pil2base(img_pil)
    template_base64 = pil2base(template_pil)

    # TODO: Template Augmentation Style
    result_base64 = None
    result_pil = base2pil(result_base64)
    return result_pil


iface3_1 = gr.Interface(
    fn=template_augmentation_style,
    inputs=[
        gr.Image(type="pil", label="image"),
        gr.Image(type="pil", label="template"),
    ],
    outputs=gr.Image(type="pil", label="result"),
)


# Function 3-2 Template Augmentation Text
def template_augmentation_text(img_pil, concept, color):
    img_base64 = pil2base(img_pil)

    # TODO: Template Augmentation Text
    result = None
    return


iface3_2 = gr.Interface(
    fn=template_augmentation_text,
    inputs=[gr.Image(type="pil", label="image"), "text", "text"],
    outputs="text",
)


# Function 4-1 Remove Background
def remove_bg(img_pil, mask_pil):
    img_base64 = pil2base(img_pil)
    mask_base64 = pil2base(mask_pil)

    # TODO: Remove Background
    result_base64 = None
    result_pil = base2pil(result_base64)

    return result_pil


iface4_1 = gr.Interface(
    fn=remove_bg,
    inputs=[gr.Image(type="pil", label="image"), gr.Image(type="pil", label="mask")],
    outputs=gr.Image(type="pil", label="result"),
)


# Function 4-2 Recommend Color
def color_recommendation(img_pil):
    img_base64 = pil2base(img_pil)

    # TODO: Recommend Color
    result = None

    return result


iface4_2 = gr.Interface(
    fn=color_recommendation,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Text(), gr.Text()],
)


# Function 4-3 Super Resolution
def super_resolution(img_pil):
    img_base64 = pil2base(img_pil)

    # TODO: Super Resolution
    result_base64 = None
    result_pil = base2pil(result_base64)

    return result_pil


iface4_3 = gr.Interface(
    fn=super_resolution,
    inputs=gr.Image(type="pil", label="image"),
    outputs=gr.Image(type="pil", label="SR"),
)


iface1.launch()
iface2.launch()
iface3_1.launch()
iface3_2.launch()
iface4_1.launch()
iface4_2.launch()
iface4_3.launch()