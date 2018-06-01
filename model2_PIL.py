import math
import copy

import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import torchvision.transforms as transforms

device = torch.device("cuda:0")

layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4']

content_layers_default = ['conv_4']
style_layers_default = ['conv_3', 'conv_4', 'conv_5', 'conv_6',' conv_7']

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


def gram_matrix(input):
    a, b, c, d = input.size()

    features = input.view(a * b, c * d)

    G = torch.mm(features, features.t())

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)


class ContentLoss(nn.Module):

    def __init__(self, target, mask, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.mask = mask.clone()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target) * self.weight
        return input

    def content_hook(self, module, grad_input, grad_output):
        mask = self.mask.clone().expand_as(grad_input[0])

        # print('Inside ' + module.__class__.__name__ + ' backward')
        #
        # print('grad_input size:', grad_input[0].size())
        # print('grad_output size:', grad_output[0].size())
        # assert grad_input[0].shape == self.mask.shape, \
        #     'grad_input:{} is not matchable with mask:{}'.format(grad_input[0].shape, self.mask.shape)

        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        # grad_input_1 = grad_input_1 * self.weight
        # grad_input_1 = grad_input_1 * mask
        # grad_input = tuple([grad_input_1])

        # plt.figure()
        # imshow(mask, title='Content hook Image')

        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        grad_input_1 = grad_input[0] * self.weight
        grad_input_1 = grad_input_1 * mask
        grad_input = tuple([grad_input_1, grad_input[1], grad_input[2]])
        return grad_input


class StyleLoss(nn.Module):

    def __init__(self, target, mask, weight):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target).detach()
        self.mask = mask.clone()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        mask = self.mask.clone().expand_as(input)
        # mask = self.mask.clone()[:, 0:1, :, :]

        # assert section
        # assert input.size()[:] == self.mask.size()[:], \
        #     'the input-size:{} is not matchable with mask-size:{}'.format(input.size()[2:], self.mask.size()[2:])

        G = gram_matrix(input * mask)
        # G.div(mask.sum())
        # self.target = self.target.div(mask.sum())
        self.loss = F.mse_loss(G, self.target) * self.weight
        #
        # G = gram_matrix(input)
        # self.loss = F.mse_loss(G, self.target) * self.weight

        return input

    def style_hook(self, module, grad_input, grad_output):
        mask = self.mask.clone().expand_as(grad_input[0])

        # print('Inside ' + module.__class__.__name__ + ' backward')
        #
        # print('grad_input size:', grad_input[0].size())
        # print('grad_output size:', grad_output[0].size())

        # assert grad_input[0].shape == self.mask.shape, \
        #     'grad_input:{} is not matchable with mask:{}'.format(grad_input[0].shape, self.mask.shape)

        # plt.figure()
        # imshow(mask, title='Style hook Image')

        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        grad_input_1 = grad_input[0] * self.weight
        grad_input_1 = grad_input_1 * mask
        grad_input = tuple([grad_input_1, grad_input[1], grad_input[2]])

        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        # grad_input_1 = grad_input_1 * self.weight
        # grad_input_1 = grad_input_1 * mask
        # grad_input = tuple([grad_input_1])
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


def get_feature_extractor(cnn, cnnmrf_image, style_image):

    feature_extractor = nn.Sequential()
    input_features = []
    target_features = []
    match_features = []
    match_masks = []
    layerIdx = 1

    i = 0
    for layer in cnn.children():
        if layerIdx < len(layers):
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = "conv_" + str(i)
            elif isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
            elif isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
            if name == layers[layerIdx]:
                print('Extracting feature layer')
                input = feature_extractor(cnnmrf_image).clone()
                target = feature_extractor(style_image).clone()
                input_features.append(input)
                target_features.append(target)
                layerIdx += 1

    return feature_extractor, input_features, target_features


def get_model_and_losses(cnn, normalization_mean, normalization_std,
                         style_img, content_img, mask_image, tmask_image,
                         style_weight=100, content_weight=5, tv_weight=1e-3,
                         content_layers=content_layers_default,
                         style_layers=style_layers_default):
    content_losses = []
    style_losses = []

    model = nn.Sequential()
    tv_loss = None

    if tv_weight > 0:
        tv_loss = TVLoss(tv_weight)
        model.add_module('tv_loss', tv_loss)

    i = 0
    for layer in cnn.children():

        if isinstance(layer, nn.Conv2d):
            sap = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
            if not isinstance(mask_image, torch.Tensor):
                mask_image = PIL_to_tensor(mask_image).to(device)
            mask_image = sap(mask_image).clone()

            # plt.figure()
            # imshow(mask_image, title='sap Image')

            i += 1
            name = "conv_" + str(i)
            model.add_module(name, layer)

        # why every time we resize the mask image to a smaller image,
        # because later we need mask image to fit input image in deep layers
        # vgg19 only shrink image size in pooling layer and the rate is 1/2!
        elif isinstance(layer, nn.MaxPool2d):
            if isinstance(mask_image, torch.Tensor):
                mask_image = tensor_to_PIL(mask_image)
            resize = transforms.Resize((math.floor(mask_image.height / 2), math.floor(mask_image.width / 2)))
            mask_image = resize(mask_image)
            mask_image = PIL_to_tensor(mask_image).to(device)

            # plt.figure()
            # imshow(mask_image, title='resize Image')

            name = "pool_" + str(i)
            model.add_module(name, layer)

        elif isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, nn.ReLU(inplace=False))

        if name in content_layers:
            print('-----Setting up content layer-----')
            target = model(content_img).clone()

            content_loss = ContentLoss(target, mask_image, content_weight)
            content_loss.register_backward_hook(content_loss.content_hook)
            model.add_module("content_loss_" + str(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            print('-----Setting up style layer-----')
            # content_target = model(content_img).detach()

            target_feature = model(style_img).clone()

            mask = mask_image.clone()
            mask = mask.expand_as(target_feature)
            target_feature = target_feature * mask

            # add a histogram match here
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


def normalize_features(tensor):
    _, c, h, w = tensor.size()
    print('Normalizing feature map with dim3[tensor] = ', c, h, w)
    x2 = torch.pow(tensor, 2)
    sum_x2 = torch.sum(x2, 0)
    dis_x2 = torch.sqrt(sum_x2.expand_as(tensor))
    Nx = tensor.div(dis_x2 + 1e-8)
    return Nx


def compute_weightMap(tensor):
    _, c, h, w = tensor.size()
    print('Computing weight map with dim3[tensor] = ', c, h, w)
    x2 = torch.pow(tensor, 2)
    sum_x2 = torch.sum(x2, 0)[0]
    sum_min, sum_max = sum_x2.min(), sum_x2.max()
    wMap = (sum_x2 - sum_min) / (sum_max - sum_min + 1e-8)
    return wMap


def noise_estimate(input):
    x_diff = input[:, :, 1:, :] - input[:, :, :-1, :]
    y_diff = input[:, :, :, 1:] - input[:, :, :, :-1]
    x_diff_sqr = torch.pow(x_diff, 2)
    y_diff_sqr = torch.pow(y_diff, 2)
    diff_sqr = (x_diff_sqr - y_diff_sqr) / 2
    noise = torch.median(diff_sqr)      # need modification
    return noise


def params_wikiart_genre(style_image, index, ouput_dir):

    tv_noise = noise_estimate(style_image)
    tv_weight = 10.0 / (1.0 + torch.exp(1e4 * tv_noise - 25.0))
    his_weight = 1.0
    content_weight = 1.0
    style_weight = 1.0

    # ...

