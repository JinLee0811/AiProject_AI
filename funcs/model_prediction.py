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
    model = torch.hub.load('pytorch/vision:v0.9.0', 'mobilenet_v2', pretrained=False)
    num_ftrs = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(num_ftrs, 20)
    model.load_state_dict(model_weights)
    model = model.to(device)

    return model



# get_prediction: 예측
def get_prediction(preprocessed_image):
    print(preprocessed_image.shape)
    model = load_model()
    model.eval() # 추론 모드로 변경

    pred_image = preprocessed_image.view(1, 3, 224, 224)

    pred = model(pred_image)
    pred_prob = torch.max(torch.nn.functional.softmax(pred))
    probability, solution_id = torch.max(pred_prob, dim=1)

    probability, solution_id = float(probability.cpu())*100, int(solution_id.cpu())+1

    return round(probability, 2), solution_id