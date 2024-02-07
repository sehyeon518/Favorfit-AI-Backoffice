# import numpy as np
# import PIL.Image as Image
# import matplotlib.pyplot as plt

# mask = Image.open("/home/sehyeon/Documents/Favorfit-Color-Equalization/Images/mask.png")
# img = Image.open("/home/sehyeon/Documents/Favorfit-Color-Equalization/Images/milk.png")
# mask_np = np.array(mask)
# img_np = np.array(img)
# black_image = np.zeros_like(img_np)
# result_np = black_image * (1 - mask_np/255) + img_np * mask_np/255
# plt.imshow(result_np)
# # background_np = np.full_like(img_np, [151, 99, 145])

# # result_np = background_np * (255-mask_np) + img_np * (mask_np)
# # plt.imshow(mask_np)
# plt.show()

import numpy as np
from PIL import Image

def resize_pil(img_pil, new_width):
    width, height = img_pil.size
    
    new_height = int(height * (new_width / width))
    
    resized_img = img_pil.resize((new_width, new_height))
    
    return resized_img

src = Image.open(Image.open("/home/sehyeon/Documents/Favorfit-Color-Equalization/Images/perfume.jpg"))
dst = resize_pil(src, src.size[0])
import cv2
cv2.imshow("dst", np.array(dst))
cv2.waitKey()
cv2.destroyAllWindows()