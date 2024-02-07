from utils import make_outpaint_condition
from PIL import Image


class Outpaint:
    def __init__(self):
        self.img_pil = None
        self.mask_pil = None
        self.result_pil = None
        self.product_pil = None
        self.composite_pil = None
        self.checkbox = False

    def get_product_pil(self):
        self.product_pil = make_outpaint_condition(self.img_pil, self.mask_pil)
        return self.product_pil

    def outpaint_origin_product(self):
        self.product_pil = self.get_product_pil()

        mask_pil_gray = self.mask_pil.convert("L")
        self.composite_pil = Image.new("RGBA", self.result_pil.size)
        self.composite_pil.paste(self.result_pil, (0, 0))
        self.composite_pil.paste(self.product_pil, (0, 0), mask_pil_gray)
        return self.composite_pil
