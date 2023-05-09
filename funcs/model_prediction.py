import torch
import torch.nn as nn


# load_model: 모델 불러오기
def load_model():
    # 모델 경로 및 device 설정
    model_path = "./models/baseline_1.ckpt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 모델 불러오기
    model_weights = torch.load(model_path, map_location=device)

    # 모델 생성
    model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=True)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, 20)
    model.load_state_dict(model_weights)
    model = model.to(device)

    return model



# get_prediction: 예측
def get_prediction(preprocessed_image):
    model = load_model()
    new_shape = (1, 3, 224, 224)
    pred_image = torch.tensor(preprocessed_image)
    pred_image = pred_image.permute(2, 0, 1)  # channel을 맨 앞으로 보내기 위해 permute
    pred_image = pred_image.view(new_shape)

    pred = model(pred_image)
    print(pred)
    probability, solution_id = torch.max(pred, dim=1)
    print(probability)

    # pred_prob = torch.max(torch.nn.functional.softmax(probability))


    probability, solution_id = float(probability.cpu())*100, int(solution_id.cpu())+1

    return round(probability, 2), solution_id