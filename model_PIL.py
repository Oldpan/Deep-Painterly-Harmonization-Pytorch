import math

import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import cuda_utils._ext.cuda_util as cu
import torchvision.transforms as transforms

import matplotlib.cm as cm
import matplotlib.pyplot as plt


device = torch.device("cuda:0")

content_layers_default = ['relu_9']
style_layers_default = ['relu_5', 'relu_9', 'relu_13']


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


loader = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image


def PIL_to_tensor(image):
    image = loader(image).unsqueeze(0)

    return image.to(device, torch.float)


def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)

    return image


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def test_show(tensor):
    c = tensor_to_PIL(tensor).convert('L')
    image_array = np.array(c)
    plt.subplot(2, 1, 1)
    plt.imshow(c, cmap=cm.gray)
    plt.axis("off")
    plt.subplot(2, 1, 2)
    plt.hist(image_array.flatten(), 256)  # flatten可以将矩阵转化成一维序列
    plt.show()


def gram_matrix(input):
    a, b, c, d = input.size()

    features = input.view(a * b, c * d)

    G = torch.mm(features, features.t())

    # return G.div(a * b * c * d)
    return G


class ContentLoss(nn.Module):

    def __init__(self, target, mask, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.mask = mask.clone()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        mask = self.mask.clone().expand_as(input)
        self.loss = F.mse_loss(input*mask, self.target) * self.weight
        # self.loss = self.loss / (input.size(1) * mask.sum())

        return input

    def content_hook(self, module, grad_input, grad_output):
        mask = self.mask.clone().expand_as(grad_input[0])

        grad_input_1 = grad_input[0]
        grad_input_1 = grad_input_1 * mask
        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        # grad_input = tuple([grad_input_1, grad_input[1], grad_input[2]])
        # grad_input_1 = grad_input_1 * self.weight
        grad_input = tuple([grad_input_1])
        return grad_input


class StyleLoss(nn.Module):

    def __init__(self, target, mask, weight):
        super(StyleLoss, self).__init__()
        self.target = target.detach()
        self.mask = mask.clone()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        mask = self.mask.clone().expand_as(input)

        G = gram_matrix(input * mask)
        # G = G.div(mask.sum())
        # self.target = self.target.div(mask.sum())
        self.loss = F.mse_loss(G, self.target) * self.weight

        return input

    def style_hook(self, module, grad_input, grad_output):
        mask = self.mask.clone().expand_as(grad_input[0])

        grad_input_1 = grad_input[0]
        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 2) + 1e-8)

        grad_input_1 = grad_input_1 * mask

        # grad_input = tuple([grad_input_1, grad_input[1], grad_input[2]])

        # grad_input_1 = grad_input_1 * self.weight
        grad_input = tuple([grad_input_1])
        return grad_input


class TVLoss(nn.Module):
    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength
        self.x_diff = torch.Tensor()
        self.y_diff = torch.Tensor()

    def forward(self, input):
        self.x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
        self.y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        return input


def get_model_and_losses(cnn, normalization_mean, normalization_std,
                         style_img, content_img, mask_image, tmask_image,
                         style_weight=100, content_weight=5, tv_weight=1e-3,
                         content_layers=content_layers_default,
                         style_layers=style_layers_default):
    content_losses = []
    style_losses = []

    normalization = Normalization(normalization_mean, normalization_std).to(device)
    model = nn.Sequential(normalization)
    tv_loss = None

    if tv_weight > 0:
        tv_loss = TVLoss(tv_weight)
        model.add_module('tv_loss', tv_loss)

    i = 0
    j = 1
    for layer in cnn.children():

        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_" + str(i)
            sap = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            if not isinstance(mask_image, torch.Tensor):
                mask_image = PIL_to_tensor(mask_image).to(device)
            mask_image = sap(mask_image).clone()

            model.add_module(name, layer)

        # why every time we resize the mask image to a smaller image,
        # because later we need mask image to fit input image in deep layers
        # vgg19 only shrink image size in pooling layer and the rate is 1/2!
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(j)
            if isinstance(mask_image, torch.Tensor):
                mask_image = tensor_to_PIL(mask_image)
            resize = transforms.Resize((math.floor(mask_image.height / 2), math.floor(mask_image.width / 2)))
            mask_image = resize(mask_image)
            mask_image = PIL_to_tensor(mask_image).to(device)
            print('mask image size {} after {}'.format(mask_image.size(), name))

            model.add_module(name, layer)
            j += 1

        elif isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, nn.ReLU(inplace=False))

        if name in content_layers:
            print('-----Setting up content {} layer-----'.format(name))
            target = model(content_img).clone()

            mask = mask_image.clone()
            mask = mask.expand_as(target)
            target = target * mask

            content_loss = ContentLoss(target, mask_image, content_weight)
            content_loss.register_backward_hook(content_loss.content_hook)
            model.add_module("content_loss_" + str(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            print('-----Setting up style {} layer-----'.format(name))

            if name in ['relu_5', 'relu_9', 'relu_13']:

                input_feature = model(content_img).clone()
                target_feature = model(style_img).clone()

                mask = mask_image.clone()
                mask = mask.expand_as(target_feature)
                match = input_feature.clone()

                cu.patchmatch_r(input_feature, target_feature, match, 3, 1)

                print('match size at style {} layer'.format(name), match.size())
                match = match * mask
                match = gram_matrix(match)
                # match = match.div(mask.sum())
                style_loss = StyleLoss(match, mask_image, style_weight)

            else:

                target_feature = model(style_img).clone()
                mask = mask_image.clone()
                mask = mask.expand_as(target_feature)
                target_feature = target_feature * mask
                style_loss = StyleLoss(target_feature, mask_image, style_weight)

            style_loss.register_backward_hook(style_loss.style_hook)
            model.add_module("style_loss" + str(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

        model = model[:i]

    return model, style_losses, content_losses, tv_loss


def original_color(content, generated):
    generated_y = cv2.cvtColor(generated, cv2.COLOR_BGR2YUV)[:, :, 0]
    content_uv = cv2.cvtColor(content, cv2.COLOR_BGR2YUV)[:, :, 1:2]
    combined_image = cv2.cvtColor(np.stack((generated_y, content_uv), 1), cv2.COLOR_YUV2BGR)
    return combined_image
