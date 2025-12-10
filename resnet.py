import torch
import torch.nn as nn
import torchvision.models as models

import numpy as np
RESNETS = { 18: (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1), 
            50: (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V1)}



class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, bn_order, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.bn_order = bn_order

        if num_layers not in RESNETS:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        # if num_input_images > 1: # TODO: add multi-image input support

        model, weights = RESNETS[num_layers]
        self.encoder = model(weights=weights)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        encoder = self.encoder
        features = []
        x = (input_image - 0.45) / 0.225
        x = encoder.conv1(x)

        if self.bn_order == "pre_bn":
            # Concatenating pre-norm features allows us to 
            # keep the scale and shift of RGB colours 
            # and recover them at output
            features.append(x)
            x = encoder.bn1(x)
            x = encoder.relu(x)
        elif self.bn_order == "monodepth":
            # Batchnorm gets rid of constants due to colour shift
            # will make the network not able to recover absolute colour shift
            # of the input image
            x = encoder.bn1(x)
            x = encoder.relu(x)
            features.append(x)
        else:
            assert False

        features.append(encoder.layer1(encoder.maxpool(x)))
        features.append(encoder.layer2(features[-1]))
        features.append(encoder.layer3(features[-1]))
        features.append(encoder.layer4(features[-1]))

        return features