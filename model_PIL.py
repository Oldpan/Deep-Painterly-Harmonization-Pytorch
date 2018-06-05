import math
import copy

import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

import cuda_utils._ext.cuda_util as cu
import torchvision.transforms as transforms

import matplotlib.cm as cm
import matplotlib.pyplot as plt


device = torch.device("cuda:0")

content_layers_default = ['conv_4']  # conv_4
style_layers_default = ['conv_3', 'conv_4', 'conv_5', 'conv_6']


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
            print('-----Setting up content {} layer-----'.format(name))
            target = model(content_img).clone()

            content_loss = ContentLoss(target, mask_image, content_weight)
            content_loss.register_backward_hook(content_loss.content_hook)
            model.add_module("content_loss_" + str(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            print('-----Setting up style {} layer-----'.format(name))

            if name in []:

                input_feature = model(content_img).clone()
                input_feature = input_feature.squeeze(0)

                target_feature = model(style_img).clone()

                # imshow(target_feature[:,4:7,:,:], title='target_feature')
                # test_show(target_feature[:,4:7,:,:])

                mask = mask_image.clone()
                mask = mask.expand_as(target_feature)

                target_feature = target_feature.squeeze(0)
                match = input_feature.clone()
                #
                # plt.figure()
                # match = match.unsqueeze(0)
                # imshow(match[:,4:7,:,:], title='match before')
                # test_show(match[:, 4:7, :, :])
                # match = match.squeeze(0)

                cu.patchmatch_r(input_feature, target_feature, match, 3, 1)

                plt.figure()
                match = match.unsqueeze(0)
                # imshow(match[:,4:7,:,:], title='match after')
                # test_show(match[:, 4:7, :, :])

                match = match * mask
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


def histogram_match(input, target, patch, stride):
    n1, c1, h1, w1 = input.size()
    n2, c2, h2, w2 = target.size()
    input.resize_(h1 * w1 * h2 * w2)
    target.resize_(h2 * w2 * h2 * w2)
    conv = torch.tensor((), dtype=torch.float32)
    conv = conv.new_zeros((h1 * w1, h2 * w2))
    conv.resize_(h1 * w1 * h2 * w2)
    assert c1 == c2, 'input:c{} is not equal to target:c{}'.format(c1, c2)

    size1 = h1 * w1
    size2 = h2 * w2
    N = h1 * w1 * h2 * w2
    print('N is', N)

    for i in range(0, N):
        i1 = i / size2
        i2 = i % size2
        x1 = i1 % w1
        y1 = i1 / w1
        x2 = i2 % w2
        y2 = i2 / w2
        kernal_radius = int((patch - 1) / 2)

        conv_result = 0
        norm1 = 0
        norm2 = 0
        dy = -kernal_radius
        dx = -kernal_radius
        while dy <= kernal_radius:
            while dx <= kernal_radius:
                xx1 = x1 + dx
                yy1 = y1 + dy
                xx2 = x2 + dx
                yy2 = y2 + dy
                if 0 <= xx1 < w1 and 0 <= yy1 < h1 and 0 <= xx2 < w2 and 0 <= yy2 < h2:
                    _i1 = yy1 * w1 + xx1
                    _i2 = yy2 * w2 + xx2
                    for c in range(0, c1):
                        term1 = input[int(c * size1 + _i1)]
                        term2 = target[int(c * size2 + _i2)]
                        conv_result += term1 * term2
                        norm1 += term1 * term1
                        norm2 += term2 * term2
                dx += stride
            dy += stride
        norm1 = math.sqrt(norm1)
        norm2 = math.sqrt(norm2)
        conv[i] = conv_result / (norm1 * norm2 + 1e-9)

    match = torch.tensor((), dtype=torch.float32)
    match = match.new_zeros(input.size())

    correspondence = torch.tensor((), dtype=torch.int16)
    correspondence.new_zeros((h1, w1, 2))
    correspondence.resize_(h1 * w1 * 2)

    for id1 in range(0, size1):
        conv_max = -1e20
        for y2 in range(0, h2):
            for x2 in range(0, w2):
                id2 = y2 * w2 + x2
                id = id1 * size2 + id2
                conv_result = conv[id1]

                if conv_result > conv_max:
                    conv_max = conv_result
                    correspondence[id1 * 2 + 0] = x2
                    correspondence[id1 * 2 + 1] = y2

                    for c in range(0, c1):
                        match[c * size1 + id1] = target[c * size2 + id2]

    match.resize_((n1, c1, h1, w1))

    return match, correspondence
