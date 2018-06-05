import math
import copy
import argparse
import os
import time
import os.path as osp

import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import cuda_utils._ext.cuda_util as cu
from PIL import Image, ImageFilter
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='DeepFake-Pytorch')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10000)')
parser.add_argument('--no-cuda', action='store_true',
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=222, metavar='S',
                    help='random seed (default: 222)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.cuda is True:
    print('===> Using GPU to train')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    print('===> Using CPU to train')

device = torch.device("cuda:0" if args.cuda else "cpu")

content_layers_default = ['conv_4']
style_layers_default = ['conv_3', 'conv_4', 'conv_5', 'conv_6']
layers = ['conv_3', 'conv_4', 'conv_5', 'conv_6']
hist_layers = ['conv_3', 'conv_6']

style_weight = 200
content_weight = 8
tv_weight = 1e-3
hist_weight = 1

loader = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor

unloader = transforms.ToPILImage()  # reconvert into PIL image


def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


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


def save_image(tensor, **para):
    num = 12
    dir = 'results_all/results_{}'.format(num)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    if not osp.exists(dir):
        os.makedirs(dir)
    image.save('results_all/results_{}/s{}-c{}-l{}-e{}-sl{:4f}-cl{:4f}.jpg'
               .format(num, para['style_weight'], para['content_weight'], para['lr'], para['epoch'],
                       para['style_loss'], para['content_loss']))


print('===> Loaing datasets')
style_image = image_loader("datasets/0_target.jpg")
content_image = image_loader("datasets/0_cmn.jpg")
mask_image = image_loader('datasets/0_c_mask_dilated.jpg')[:, 0:1, :, :]
mask_image_ori = mask_image.clone()
tmask_image = Image.open('datasets/0_c_mask.jpg').convert('RGB')
tmask_image = tmask_image.filter(ImageFilter.GaussianBlur())
tmask_image = PIL_to_tensor(tmask_image)
tmask_image_ori = tmask_image.clone()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

cnn = models.vgg19(pretrained=True).features.to(device).eval()

print('===> Initialize the image...')
input_img = content_image.clone()


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())

    return G.div(a * b * c * d)


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
    noise = torch.median(diff_sqr)  # need modification
    return noise


class ContentLoss(nn.Module):

    def __init__(self, target, msk, weight):
        super(ContentLoss, self).__init__()
        self.target = target * msk
        self.mask = mask.clone()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        return input

    def content_hook(self, module, grad_input, grad_output):
        self.loss = F.mse_loss((self.target * self.msk), self.target) * self.weight

        grad_input_1 = grad_input[0] * self.msk
        grad_input_1 = grad_input_1.div(torch.norm(grad_input[0], 2) + 1e-8)
        grad_input_1 = grad_input_1 * self.weight
        grad_input = tuple([grad_input_1, grad_input[1], grad_input[2]])
        return grad_input


class StyleLoss(nn.Module):

    def __init__(self, target_gram, msk, weight):
        super(StyleLoss, self).__init__()
        self.target_gram = target_gram
        self.msk = msk
        self.weight = weight
        self.gram = gram_matrix
        self.G = None
        self.loss = 0
        self.msk_mean = msk.mean()

    def forward(self, input):
        self.G = self.gram((input * self.msk))
        self.loss = F.mse_loss(self.G, self.target_gram) * self.weight
        return input

    def style_hook(self, module, grad_input, grad_output):
        mask = self.mask.clone().expand_as(grad_input[0])

        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        grad_input_1 = grad_input[0] * self.weight
        grad_input_1 = grad_input_1 * mask
        grad_input = tuple([grad_input_1, grad_input[1], grad_input[2]])

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


class HistLoss(nn.Module):
    def __init__(self, strength, input, target, nbins, maskI, maskJ, mask):
        super(TVLoss, self).__init__()
        self.strength = strength
        self.loss = 0
        self.nbins = nbins
        self.maskI = maskI
        self.nI = maskI.sum()

        _, c, h1, w1 = input.size()
        self.msk = self.maskI.float().expand_as(input)
        self.msk_sub = torch.ones((1, c, h1, w1)) * (1 - self.msk.float())
        self.mask = mask.float().expand_as(input)

        self.nJ = maskJ.sum()
        _, c, h2, w2 = target.size()
        mJ = maskJ.expand_as(target)
        J = target.float()
        _J = J[mJ].view(-1, c, self.nJ)
        self.minJ = _J.min(2)
        self.maxJ = _J.max(2)

        self.hisJ = torch.tensor([1])
        cu.histogram(target, self.nbins, self.minJ, self.maxJ, maskJ, self.hisJ)

        self.hisJ = self.hisJ * (self.nI / self.nJ)
        self.cumJ = torch.cumsum(self.hisJ, 2)

    def forward(self, input):
        self.output = input
        return self.output

    def hist_hook(self, module, grad_input, grad_output):
        grad_input_1 = grad_input[0]
        grad_input_1 = grad_input_1.expand_as(self.output)
        I = self.output
        _, c, h1, w1 = I.size()
        _I = (I * self.msk) - self.msk_sub
        sortI, idxI = torch.sort(_I.view_(1, c, h1 * w1), 2)

        R = I.clone()
        cu.hist_remap2(I, self.nI, self.maskI, self.hisJ, self.cumJ, self.minJ, self.maxJ,
                       self.nbins, sortI, idxI, R)
        grad_input_1 = grad_input_1 + I
        grad_input_1 = grad_input_1 - R

        err = grad_input_1.clone()
        err = err.pow(2.0)
        self.loss = err.sum() * self.strength / self.output.nelement()

        magnitude = torch.norm(grad_input_1, 2)
        grad_input_1 = grad_input_1.div(magnitude + 1e-8) * self.strength
        grad_input_1 = grad_input_1 * mask
        grad_input = tuple([grad_input_1, grad_input[1], grad_input[2]])

        return grad_input


def get_input_optimizer(input_img, lr):
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=lr)
    return optimizer


def original_color(content, generated):
    generated_y = cv2.cvtColor(generated, cv2.COLOR_BGR2YUV)[:, :, 0]
    content_uv = cv2.cvtColor(content, cv2.COLOR_BGR2YUV)[:, :, 1:2]
    combined_image = cv2.cvtColor(np.stack((generated_y, content_uv), 1), cv2.COLOR_YUV2BGR)
    return combined_image


def params_wikiart_genre(style_image, index, ouput_dir):
    tv_noise = noise_estimate(style_image)
    tv_weight = 10.0 / (1.0 + torch.exp(1e4 * tv_noise - 25.0))
    his_weight = 1.0
    content_weight = 1.0
    style_weight = 1.0
    # ....


input_features = []
target_features = []
match_features = []
match_masks = []

feature_extractor = nn.Sequential()
layerIdx = 0

i = 0
for layer in cnn.children():
    if layerIdx < len(layers):
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = "conv_" + str(i)
            feature_extractor.add_module(name, layer)
        elif isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            feature_extractor.add_module(name, layer)
        elif isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            feature_extractor.add_module(name, nn.ReLU(inplace=False))
        if name == layers[layerIdx]:
            print('Extracting feature layer')
            input = feature_extractor(content_image).clone()
            target = feature_extractor(style_image).clone()
            input_features.append(input)
            target_features.append(target)
            layerIdx += 1

del [feature_extractor]

curr_corr, corr = None, None
curr_mask, mask = None, None

i = len(layers) - 1  # 3 2 1 0
while i > 0:
    name = layers[i]
    print('Working on patchmatch layer', i, ":", name)
    A = input_features[i].clone()
    BP = target_features[i].clone()
    N_A = normalize_features(A)
    N_BP = normalize_features(BP)

    _, c, h, w = A.size()
    _, __, h2, w2 = BP.size()

    if not h == h2 or not w == w2:
        print("Input and target should have the same dimension! h, h2, w, w2 = ", h, h2, w, w2)

    resize = transforms.Resize((h, w))
    tmask = resize(tensor_to_PIL(torch.gt(tmask_image_ori[:, 0:1, :, :], 0.1)))

    tmask = PIL_to_tensor(tmask).int()

    if name in ['conv_6']:

        print("Initializing NNF in layer ", i, ":", name, "with patch", 3)
        print("Brute-force patch matching...")
        init_corr = torch.tensor(1)
        N_A.squeeze(0)
        N_BP.squeeze(0)
        cu.patchmatch(N_A, N_BP, init_corr, 3)

        resize = transforms.Resize((h, w))
        guide = resize(tensor_to_PIL(style_image))
        guide = PIL_to_tensor(tmask)

        print("  Refining NNF...")
        cu.refineNNF(N_A, N_BP, init_corr, guide, tmask, corr, 5, 1)
        cu.Ring2(N_A, N_BP, corr, mask, 1, tmask)

        curr_corr = corr
        curr_mask = mask

    else:
        print('Upsampling NNF in layer', i, ':', name)
        cu.upsample_corr(corr, h, w, curr_corr)
        if isinstance(mask, torch.Tensor):
            curr_mask = transforms.Resize(tensor_to_PIL(mask), (h, w)).int()

    i -= 1
    match_features.append(BP)
    match_masks.append(curr_mask)

gram_features, hist_features = [], []
gram_match_masks, hist_match_masks = [], []
gramIdx, hisIdx = 1, 1
for i in range(len(layers)):
    name = layers[i]
    features = match_features[-(i - 1)]
    mask = match_masks[-(i - 1)]
    if gramIdx < len(style_layers_default) or hisIdx < len(hist_layers):
        if name == style_layers_default[gramIdx]:
            gram_features.append(features)
            gram_match_masks.append(mask)
            gramIdx += 1
        if name == hist_layers[hisIdx]:
            hist_features.append(features)
            hist_match_masks.append(mask)
            hisIdx += 1

input_features = None
target_features = None

print('Building model ...')
content_losses, style_losses, hist_losses = [], [], []
next_cont_idx, next_style_idx, next_hist_idx = 0, 0, 0

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

        i += 1
        name = "conv_" + str(i)
        model.add_module(name, layer)

    elif isinstance(layer, nn.MaxPool2d):
        if isinstance(mask_image, torch.Tensor):
            mask_image = tensor_to_PIL(mask_image)
        resize = transforms.Resize((math.floor(mask_image.height / 2), math.floor(mask_image.width / 2)))
        mask_image = resize(mask_image)
        mask_image = PIL_to_tensor(mask_image).to(device)

        name = "pool_" + str(i)
        model.add_module(name, layer)

    elif isinstance(layer, nn.ReLU):
        name = "relu_" + str(i)
        model.add_module(name, nn.ReLU(inplace=False))

    if name in content_layers_default:
        print('-----Setting up content layer-----')
        target = model(content_image).clone()

        content_loss = ContentLoss(target, mask_image, content_weight)
        content_loss.register_backward_hook(content_loss.content_hook)
        model.add_module("content_loss_" + str(i), content_loss)
        content_losses.append(content_loss)

    if name in style_layers_default:
        print('-----Setting up style layer-----')

        content_target = model(content_image).detach()
        target_feature = model(style_image).clone()
        mask = mask_image.clone()
        mask = mask.expand_as(target_feature)

        _, c, h1, w1 = content_target.size()
        _, __, h2, w2 = target_feature.size()

        gram_feature = gram_features[next_style_idx]
        gram_mask = gram_match_masks[next_style_idx]
        gram_msk = gram_mask.float().expand_as(content_target)
        target_gram = gram_matrix((gram_feature * gram_msk)).clone()

        style_loss = StyleLoss(target_gram, mask, style_weight)
        style_loss.register_backward_hook(style_loss.style_hook)
        model.add_module("style_loss" + str(i), style_loss)
        style_losses.append(style_loss)

        next_style_idx += 1

        if name in hist_layers:
            print('Setting up histogram layer', i, ':', name)
            maskI = torch.gt(mask_image, 0.1)
            maskJ = hist_match_masks[next_hist_idx].byte()
            hist_feature = hist_features[next_hist_idx]
            loss_model = HistLoss(hist_weight, content_target, hist_feature, 256, maskI, maskJ, mask_image)
            model.add_module('hist_loss' + str(next_hist_idx), loss_model)
            hist_losses.append(loss_model)

for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss) or isinstance(model[i], HistLoss):
        break

    model = model[:i]


def run_painterly_transfer(cnn, normalization_mean, normalization_std,
                           style_img, content_img, mask_img, tmask_img, num_steps=1000,
                           style_weight=0.01, content_weight=1, tv_weight=0, lr=1):
    print('===> Building the painterly model...')
    model, style_loss, content_loss, tv_loss = get_model_and_losses(cnn, normalization_mean, normalization_std,
                                                                    style_img, content_img, mask_img, tmask_img,
                                                                    style_weight, content_weight, tv_weight)
    # model.register_backward_hook(model_hook)
    optimizer = get_input_optimizer(input_img, lr=lr)

    print('===> Optimizer running...')
    run = [0]
    while run[0] <= num_steps:

        def closure():

            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            content_score = 0
            style_score = 0

            for sl in content_loss:
                content_score += sl.loss
            for sl in style_loss:
                style_score += sl.loss

            if tv_loss is not None:
                tv_score = tv_loss.loss
                loss = style_score + content_score + tv_score
            else:
                loss = style_score + content_score

            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("epoch:{}".format(run))
                if tv_loss is not None:
                    tv_score = tv_loss.loss
                    print('Content loss : {:4f} Style loss : {:4f} TV loss : {:4f}'.format(
                        content_score.item(), style_score.item(), tv_score.item()))
                else:
                    print('Content loss : {:4f} Style loss : {:4f}'.format(
                        content_score.item(), style_score.item()))

                new_image = input_img * tmask_image
                new_image += (style_img * (1.0 - tmask_img))

                para = {'style_weight': style_weight, 'content_weight': content_weight,
                        'epoch': run[0], 'lr': lr, 'content_loss': content_score.item(),
                        'style_loss': style_score.item()}

                save_image(new_image, **para)
                # plt.figure()
                # imshow(new_image, title='new Image')

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img


# 第一张图 s_w: 200 c_w:8
# s15 c0.5 s20 c0.5 s20 c1
if __name__ == '__main__':
    style_weights = [0.1, 0.5, 1, 1.5, 2.5, 5, 10, 15, 20, 50, 100, 150, 200, 500, 1000,
                     5000, 10000, 50000, 100000, 500000, 1000000]
    content_weights = [1, 5, 10, 100]

    style_weights_rd = list(np.random.randint(100, 200, size=20))
    content_weights_rd = list(np.random.randint(4, 9, size=5))

    # for i in range(len(content_weights_rd)):
    #     for j in range(len(style_weights_rd)):
    #         output = run_painterly_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, style_img=style_image,
    #                                         content_img=content_image, mask_img=mask_image, tmask_img=tmask_image,
    #                                         style_weight=int(style_weights_rd[j]), content_weight=int(content_weights_rd[i]), lr=1)

    since = time.time()
    output = run_painterly_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, style_img=style_image,
                                    content_img=content_image, mask_img=mask_image, tmask_img=tmask_image,
                                    num_steps=1100,
                                    style_weight=220, content_weight=8, tv_weight=0, lr=0.5)
    time_elapsed = time.time() - since
    print('The time used is {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    pass
