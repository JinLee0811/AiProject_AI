import albumentations
import numpy as np
from PIL import Image
from io import BytesIO


def image_preprocessing(img_data):

    aug_image = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225],
                                 max_pixel_value=255.0,
                                 p=1.0)], p=1.0)

    # BytesIO 객체로 변환하여 PIL 이미지 열기!
    image = Image.open(BytesIO(img_data))
    np_img = np.array(image)
    preprocessed_image = aug_image(image=np_img)['image']

    return preprocessed_image