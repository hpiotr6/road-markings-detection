import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2


class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.drop1 = nn.Dropout2d(p=0.5)
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.drop1 = nn.Dropout2d(p=0.5)

        self.fc1 = torch.nn.LazyLinear(120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1)  # usnunięto tu batch size
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def _transform_image(img_path):
    test_transform = A.Compose(
        [
            A.ToGray(always_apply=True),
            A.Normalize(),
            ToTensorV2(),
        ]
    )
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = test_transform(image=image)["image"]
    return image


def inference(model, checkpoint_path, img_path):
    image = _transform_image(img_path)
    model.load_state_dict(torch.load(checkpoint_path))
    output = model(image)
    return torch.argmax(output)


if __name__ == "__main__":
    LABELS_MAP = {
        0: "Background",
        1: "20 sign",
        2: "30-sign",
    }
    WEIGHTS_PATH = os.path.join("checkpoints", "model_weights.pth")
    IMG_PATH = os.path.join("imgs_merged", "38f3f7b468e8806a.jpg")
    model = LeNet()

    label = inference(model=model, checkpoint_path=WEIGHTS_PATH, img_path=IMG_PATH)

    print(label)
