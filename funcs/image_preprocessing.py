from PIL import Image
from io import BytesIO
import torchvision.transforms as transforms


def image_preprocessing(img_data):

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # BytesIO 객체로 변환하여 PIL 이미지 열기
    image = Image.open(BytesIO(img_data))
    preprocessed_image = data_transforms(image)

    return preprocessed_image