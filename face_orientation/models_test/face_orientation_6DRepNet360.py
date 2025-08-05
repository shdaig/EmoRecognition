import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks import python
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
# import datasets
import utils
from PIL import Image

import math
import warnings
warnings.filterwarnings("ignore")


class SixDRepNet360(nn.Module):
    def __init__(self, block, layers, fc_layers=1):
        self.inplanes = 64
        super(SixDRepNet360, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)

        self.linear_reg = nn.Linear(512 * block.expansion, 6)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.linear_reg(x)
        out = utils.compute_rotation_matrix_from_ortho6d(x)

        return out


cap = cv2.VideoCapture(0)

model = SixDRepNet360(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 6)
transformations = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
saved_state_dict = torch.load("../../models/6DRepNet360_Full-Rotation_300W_LP+Panoptic.pth")
model.load_state_dict(saved_state_dict)
model.eval()

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path='../../models/blaze_face_short_range.tflite'),
    running_mode=VisionRunningMode.IMAGE)

with FaceDetector.create_from_options(options) as detector:
    while cap.isOpened():
        _, image = cap.read()
        image = cv2.flip(image, 1)
        img_h, img_w, img_c = image.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        detection_result = detector.detect(mp_image)

        for detection in detection_result.detections[:1]:
            bbox = detection.bounding_box
            k = 1.5
            center_point = [int(bbox.origin_x + bbox.width / 2), int(bbox.origin_y + bbox.height / 2)]
            start_point = [int(center_point[0] - (bbox.width / 2 * k)), int(center_point[1] - (bbox.height / 2 * (k + 0.5)))]
            end_point = [int(center_point[0] + (bbox.width / 2 * k)), int(center_point[1] + (bbox.height / 2 * k))]

            start_point[0] = np.clip(start_point[0], 0, img_w)
            start_point[1] = np.clip(start_point[1], 0, img_h)
            end_point[0] = np.clip(end_point[0], 0, img_w)
            end_point[1] = np.clip(end_point[1], 0, img_h)
            print(f"{start_point[0]}:{end_point[0]}, {start_point[1]}:{end_point[1]}")
            img = image[start_point[0]:end_point[0], start_point[1]:end_point[1]].copy()
            img = Image.fromarray(img)
            img = transformations(img)
            img = torch.Tensor(img[None, :])
            R_pred = model(img)
            euler = utils.compute_euler_angles_from_rotation_matrices(R_pred) * 180 / np.pi
            p_pred_deg = euler[:, 0].cpu()
            y_pred_deg = euler[:, 1].cpu()
            r_pred_deg = euler[:, 2].cpu()

            cv2.rectangle(image, start_point, end_point, (255, 0, 0), 3)
            utils.draw_axis(image, y_pred_deg, p_pred_deg, r_pred_deg)

        cv2.imshow('live cam', image)
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
