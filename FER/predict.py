import os

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import shutil
import torch.nn.functional as F


N_CLASSES = 6
LABEL_MAP =  ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

THIS_DIR = os.path.dirname(os.path.realpath(__file__))
MODEL_FILE = './checkpoints/resnet18/2020-06-09-21-49/model_best_by_accu.pth'
predict_folder = './out/PrivateTest'


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self._model = models.resnet18()
        self._model.fc = nn.Linear(512, N_CLASSES)

    def forward(self, x):
        return self._model(x)


class Classify:
    def __init__(self):
        self._model = Network()
        self._model.load_state_dict(torch.load(
            os.path.join(THIS_DIR, MODEL_FILE), map_location='cpu'))
        self._transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
                                ])
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self._model.to(self.device)
        self._model.eval()

    def predict(self, img_arr):
        image = Image.fromarray(img_arr)
        tensor = self._transform(image)
        tensor = torch.unsqueeze(tensor, 0)
        tensor = tensor.to(self.device)
        logit = self._model(tensor)
        prediction = torch.argmax(logit[0]).cpu().item()
        score = torch.max(F.softmax(logit).detach())
        return score, LABEL_MAP[prediction]


if __name__ == '__main__':
    folder_name = ''
    if not os.path.isdir(folder_name):
        os.makedirs(folder_name)
    print(folder_name,':\t', len(os.listdir(folder_name)))
    model = Classify()
    model.predict(folder_name)
