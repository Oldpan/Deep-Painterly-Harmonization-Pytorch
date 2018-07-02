import math
import argparse
import os
import gc
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

content_layers = ['relu_9']
style_layers = ['relu_1', 'relu_3', 'relu_5', 'relu_9']
layers = ['relu_1', 'relu_3', 'relu_5', 'relu_9']
hist_layers = ['relu_1', 'relu_9']

style_weight = 10
content_weight = 1
tv_weight = 0
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
    num = 14
    dir = 'results_all/results_{}'.format(num)
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    if not osp.exists(dir):
        os.makedirs(dir)
    image.save('results_all/results_{}/s{}-c{}-l{}-e{}-sl{:4f}-cl{:4f}-hl{:4f}.jpg'
               .format(num, para['style_weight'], para['content_weight'], para['lr'], para['epoch'],
                       para['style_loss'], para['content_loss'], para['his_loss']))


print('===> Loaing datasets')
style_image = image_loader("datasets/16_target.jpg")
content_image = image_loader("datasets/16_naive.jpg")
cnnmrf_image = image_loader("datasets/16_cnnmrf.jpg")
# 这里注意 [:,0:1,:,:] 和 [:,1:,:,:] 的区别
mask_image = image_loader('datasets/16_c_mask_dilated.jpg')[:, 0:1, :, :]
# mask_image[mask_image > 0] = 1
mask_image_ori = mask_image.clone()
tmask_image = Image.open('datasets/16_c_mask.jpg').convert('RGB')

tmask_image = tmask_image.filter(ImageFilter.GaussianBlur())
tmask_image = PIL_to_tensor(tmask_image)
# tmask_image[tmask_image > 0] = 1
tmask_image_ori = tmask_image.clone()

print('content image size', content_image.size())
print('cnnmrf image size', cnnmrf_image.size())
print('styke image size', style_image.size())
print('mask image size', mask_image.size())
print('tmask image size', tmask_image.size())

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

cnn = models.vgg19(pretrained=True).features.to(device).eval()

print('===> Initialize the image...')
input_img = content_image.clone()


def gram_matrix(input):
    a, b, c, d = input.size()
    features = input.view(a * b, c * d)
    G = torch.mm(features, features.t())

    # return G.div(a * b * c * d)
    return G


def normalize_features(tensor):
    _, c, h, w = tensor.size()
    print('Normalizing feature map with dim3[tensor] = ', c, h, w)
    x2 = torch.pow(tensor, 2)
    sum_x2 = torch.sum(x2, 1)
    dis_x2 = torch.sqrt(sum_x2)
    dis_x2 = dis_x2.expand_as(tensor)
    Nx = tensor.div(dis_x2 + 1e-8)
    return Nx


def compute_weightMap(tensor):
    _, c, h, w = tensor.size()
    print('Computing weight map with dim3[tensor] = ', c, h, w)
    x2 = torch.pow(tensor, 2)
    sum_x2 = torch.sum(x2, 1)[0]
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

    def __init__(self, target, mask, weight):
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.mask = mask.clone()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        mask = self.mask.clone().expand_as(input)
        self.loss = F.mse_loss(input * mask, self.target) * self.weight
        # self.loss = self.loss / (input.size(1) * mask.sum())

        return input

    def content_hook(self, module, grad_input, grad_output):
        mask = self.mask.clone().expand_as(grad_input[0])

        grad_input_1 = grad_input[0]
        grad_input_1 = grad_input_1 * mask
        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        # grad_input_1 = grad_input_1 * self.weight
        grad_input = tuple([grad_input_1])
        return grad_input


class StyleLoss(nn.Module):

    def __init__(self, target_gram, mask, weight):
        super(StyleLoss, self).__init__()
        self.target_gram = target_gram
        self.mask = mask
        self.weight = weight
        self.G = None
        self.loss = 0
        self.msk_mean = mask.mean()

    def forward(self, input):
        self.G = gram_matrix((input * self.mask))
        self.loss = F.mse_loss(self.G, self.target_gram) * self.weight
        return input

    def style_hook(self, module, grad_input, grad_output):
        mask = self.mask.clone().expand_as(grad_input[0])

        # grad_input_1 = grad_input[0].div(torch.norm(grad_input[0], 1) + 1e-8)
        grad_input_1 = grad_input[0]
        grad_input_1 = grad_input_1 * mask
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


# def select_idx(tensor, idx):
#     ch = tensor.size(0)
#     return tensor.view(-1)[idx.view(-1)].view(ch,-1)
#
#
# def remap_hist(x,hist_ref):
#     ch, n = x.size()
#     sorted_x, sort_idx = x.data.sort(1)
#     ymin, ymax = x.data.min(1)[0].unsqueeze(1), x.data.max(1)[0].unsqueeze(1)
#     hist = hist_ref * n/hist_ref.sum(1).unsqueeze(1)#Normalization between the different lengths of masks.
#     cum_ref = hist.cumsum(1)
#     cum_prev = torch.cat([torch.zeros(ch,1).cuda(), cum_ref[:,:-1]],1)
#     step = (ymax-ymin)/256
#     rng = torch.arange(1,n+1).unsqueeze(0).cuda()
#     idx = (cum_ref.unsqueeze(1) - rng.unsqueeze(2) < 0).sum(2).long()
#     ratio = (rng - select_idx(cum_prev,idx)) / (1e-8 + select_idx(hist,idx))
#     ratio = ratio.squeeze().clamp(0,1)
#     new_x = ymin + (ratio + idx.float()) * step
# #     print(new_x[: , -2:-1].size())
#     new_x[: , -2:-1] = ymax
#     _, remap = sort_idx.sort()
#     new_x = select_idx(new_x,idx)
#     return new_x


class HistLoss(nn.Module):
    def __init__(self, strength, input, target, nbins, maskI, maskJ, mask):
        super(HistLoss, self).__init__()
        self.strength = strength
        self.loss = 0
        self.nbins = nbins
        self.maskI = maskI
        self.nI = maskI.sum()

        _, c, h1, w1 = input.size()
        self.msk = self.maskI.float().expand_as(input)
        self.msk_sub = torch.ones((1, c, h1, w1)).to(device) * (1 - self.msk.float())
        self.mask = mask.float().expand_as(input)

        self.nJ = maskJ.sum()
        _, c, h2, w2 = target.size()

        # print('maskJ', maskJ.size())
        mJ = maskJ[:, 0:1, :, :, ].expand_as(target)
        J = target.float()
        _J = J[mJ.byte()].view(c, self.nJ)
        # print('debug-shape', _J.size())
        self.minJ, _ = _J.min(1)
        self.maxJ, _ = _J.max(1)

        self.hisJ = J.clone()

        # print('before maskJ size', maskJ.size())
        cu.histogram(target, self.nbins, self.minJ, self.maxJ, maskJ, self.hisJ)  # 返回self.hisJ
        # print('after maskJ size', maskJ.size())

        # print('hisJ-size',self.hisJ.size())
        # print('hisJ-type', type(self.hisJ))
        self.hisJ = self.hisJ * (self.nI / self.nJ).float()
        self.cumJ = torch.cumsum(self.hisJ, 1)

    def forward(self, input):

        I = input.clone()

        _, c, h1, w1 = I.size()
        _I = (I * self.msk) - self.msk_sub
        sortI, idxI = torch.sort(_I.view(1, c, h1 * w1), 2)

        idxI = idxI.int()
        R = I.clone()

        cu.hist_remap2(I, int(self.nI), self.maskI, self.hisJ, self.cumJ, self.minJ, self.maxJ,
                       self.nbins, sortI, idxI, R)

        self.loss = F.mse_loss(I, R) * self.strength

        return input

    def hist_hook(self, module, grad_input, grad_output):

        grad_input_1 = grad_input[0]
        grad_input_1 = grad_input_1.expand_as(self.output)

        I = self.output.clone()

        _, c, h1, w1 = I.size()
        _I = (I * self.msk) - self.msk_sub
        sortI, idxI = torch.sort(_I.view(1, c, h1 * w1), 2)

        idxI = idxI.int()
        R = I.clone()

        cu.hist_remap2(I, int(self.nI), self.maskI, self.hisJ, self.cumJ, self.minJ, self.maxJ,
                       self.nbins, sortI, idxI, R)

        # 下面这两条语句会引发 不正确的内存访问
        grad_input_1.add_(I)
        grad_input_1.add_(-1, R)
        # grad_input_1 = grad_input_1 + I - R

        err = grad_input_1.clone()
        err = err.pow(2.0)
        self.loss = err.sum() * self.strength / self.output.nelement()

        # magnitude = torch.norm(grad_input_1, 2)
        # grad_input_1 = grad_input_1.div(magnitude + 1e-8) * self.strength
        grad_input_1 = grad_input_1 * self.mask
        grad_input = tuple([grad_input_1])

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

feature_extractor = nn.Sequential().to(device)
layerIdx = 0

"""
提取特征
"""
print('Extracting feature layer')
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
            feature_extractor.add_module(name, nn.ReLU(inplace=True))
        if name == layers[layerIdx]:
            input = feature_extractor(content_image).clone()
            target = feature_extractor(style_image).clone()

            input_features.append(input)
            target_features.append(target)

            del input
            del target

            layerIdx += 1

print('input_features:', [layer.size() for layer in input_features])
print('target_features:', [layer.size() for layer in target_features])

# for m in feature_extractor:
#     if isinstance(m, nn.Conv2d):
#         m.weight = None

del feature_extractor


curr_corr, corr = None, None
curr_mask, mask = None, None

i = len(layers) - 1  # 3 2 1 0
while i >= 0:
    name = layers[i]
    print('Working on patchmatch layer', i, ":", name)
    A = input_features[i].clone()
    BP = target_features[i].clone()
    N_A = normalize_features(A)
    N_BP = normalize_features(BP)

    print('Normalized A size', N_A.size())
    print('Normalized BP size', N_BP.size())

    _, c, h, w = A.size()
    _, __, h2, w2 = BP.size()

    if not h == h2 or not w == w2:
        print("Input and target should have the same dimension! h, h2, w, w2 = ", h, h2, w, w2)

    resize = transforms.Resize((h, w))
    # 需要修改以防数据格式的错误  原文中gt的目的可能是消除一些精度误差
    # print(tmask_image_ori[:, 0:1, :, :])
    # a = torch.gt(tmask_image_ori[:, 0:1, :, :], 0.01)
    # print('test',a)
    # 这里不适用精度误差了 因为在读取mask image的时候已经进行了 非零值挑选

    # 注意这里使用的是tmask而不是之前使用的mask
    tmask = resize(tensor_to_PIL(tmask_image_ori[:, 0:1, :, :]))
    tmask = torch.gt(PIL_to_tensor(tmask), 0.01).int()  # int32
    assert tmask[tmask > 0] is not None
    # print('tmask > 0', tmask[tmask > 0])

    if i == len(layers) - 1:
        print("Initializing NNF in layer ", i, ":", name, "with patch", 3)
        print("Brute-force patch matching...")

        init_corr = N_A.clone().int()
        print('N_A', N_A.size())
        print('N_BP', N_BP.size())

        result = cu.patchmatch(N_A, N_BP, init_corr, 3)

        print('init_corr size', init_corr.size())

        guide = resize(tensor_to_PIL(style_image))
        guide = PIL_to_tensor(guide)
        print('guide size', guide.size())

        print("  Refining NNF...")

        corr = torch.ones(h, w, 2).int().to(device)  # int32
        mask = torch.ones(h, w).int().to(device)     # int32
        # 因为tmask为 (1,1,xxx,xxx) 要变成 (xxx,xxx)

        # 需要修改  ** mask 和 corr 初始化问题
        cu.refineNNF(N_A, N_BP, init_corr, guide, tmask, corr, 5, 1)
        cu.Ring2(N_A, N_BP, corr, mask, 1, tmask)

        print('corr', corr.size())
        print('mask', mask.size())

        curr_corr = corr
        # 进行clone避免对curr-mask的操作会对mask造成影响
        curr_mask = mask.clone()
        curr_mask.unsqueeze_(0).unsqueeze_(0)

    else:
        print('Upsampling NNF in layer', i, ':', name)
        cu.upsample_corr(corr, h, w, curr_corr)

        curr_mask = resize(tensor_to_PIL(mask.unsqueeze(0).unsqueeze(0).float()))
        curr_mask = PIL_to_tensor(curr_mask).int()

    i -= 1
    match_features.append(BP)
    match_masks.append(curr_mask)

print('match_features:', [layer.size() for layer in match_features])
print('match_masks:', [layer.size() for layer in match_masks])

gram_features, hist_features = [], []
gram_match_masks, hist_match_masks = [], []
gramIdx, hisIdx = 0, 0
for i in range(len(layers)):
    name = layers[i]
    features = match_features[len(layers) - i - 1]
    mask = match_masks[len(layers) - i - 1]
    if gramIdx < len(style_layers) or hisIdx < len(hist_layers):
        if name == style_layers[gramIdx]:
            gram_features.append(features)
            gram_match_masks.append(mask)
            gramIdx += 1
        if name == hist_layers[hisIdx]:
            hist_features.append(features)
            hist_match_masks.append(mask)
            hisIdx += 1

print('hist_features:', [layer.size() for layer in hist_features])
print('hist_match_masks:', [layer.size() for layer in hist_match_masks])

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

    if name in content_layers:
        print('-----Setting up content {} layer-----'.format(name))
        target = model(content_image).clone()

        mask = mask_image.clone()
        mask = mask.expand_as(target)
        target = target * mask

        content_loss = ContentLoss(target, mask_image, content_weight)
        content_loss.register_backward_hook(content_loss.content_hook)
        model.add_module("content_loss_" + str(i), content_loss)
        content_losses.append(content_loss)

    if name in style_layers:
        print('-----Setting up style {} layer-----'.format(name))

        content_target = model(cnnmrf_image).detach()
        target_feature = model(style_image).clone()
        mask = mask_image.clone()
        mask = mask.expand_as(target_feature)

        _, c, h1, w1 = content_target.size()
        _, __, h2, w2 = target_feature.size()

        gram_feature = gram_features[next_style_idx]
        gram_mask = gram_match_masks[next_style_idx]
        gram_msk = gram_mask.float().expand_as(content_target)
        gram_feature = gram_feature * gram_msk
        gram_feature = gram_feature * torch.sqrt(mask.sum()/gram_msk.sum())
        target_gram = gram_matrix(gram_feature).clone()

        # target_gram.div_(gram_mask.sum().float() * c)

        style_loss = StyleLoss(target_gram, mask, style_weight)
        style_loss.register_backward_hook(style_loss.style_hook)
        model.add_module("style_loss" + str(i), style_loss)
        style_losses.append(style_loss)

        next_style_idx += 1

        if name in hist_layers:
            print('Setting up histogram layer', next_hist_idx, ':', name)
            # print('mask_image', mask_image[mask_image>0])
            maskI = torch.gt(mask_image, 0.01)
            maskJ = hist_match_masks[next_hist_idx].byte()
            hist_feature = hist_features[next_hist_idx]

            loss_model = HistLoss(hist_weight, content_target, hist_feature, 256, maskI, maskJ, mask_image)
            # loss_model.register_backward_hook(loss_model.hist_hook)
            model.add_module('hist_loss' + str(next_hist_idx), loss_model)
            hist_losses.append(loss_model)

            next_hist_idx += 1

for i in range(len(model) - 1, -1, -1):
    if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss) or isinstance(model[i], HistLoss):
        break

    model = model[:i]

del cnn

gc.collect()


print(model)
lr = 1

optimizer = get_input_optimizer(input_img, lr=lr)

print('===> Optimizer running...')
run = [0]
while run[0] <= 1000:

    def closure():

        input_img.data.clamp_(0, 1)

        optimizer.zero_grad()
        model(input_img)

        content_score = 0
        style_score = 0
        his_score = 0

        for sl in content_losses:
            content_score += sl.loss
        for sl in style_losses:
            style_score += sl.loss
        for hl in hist_losses:
            his_score += hl.loss

        if tv_loss is not None:
            tv_score = tv_loss.loss
            loss = style_score + content_score + tv_score + his_score
        else:
            loss = style_score + content_score + his_score

        loss.backward(retain_graph=True)

        run[0] += 1
        if run[0] % 50 == 0:
            print("epoch:{}".format(run))
            if tv_loss is not None:
                tv_score = tv_loss.loss
                print('Content loss : {:4f} Style loss : {:4f} His loss : {:4f} TV loss : {:4f}'.format(
                    content_score.item(), style_score.item(), his_score, tv_score.item()))
            else:
                print('Content loss : {:4f} Style loss : {:4f} His loss : {:4f}'.format(
                    content_score.item(), style_score.item(), his_score))

            new_image = input_img * tmask_image
            new_image += (style_image * (1.0 - tmask_image))

            para = {'style_weight': style_weight, 'content_weight': content_weight,
                    'epoch': run[0], 'lr': lr, 'content_loss': content_score.item(),
                    'style_loss': style_score.item(), 'his_loss': his_score}

            save_image(new_image, **para)

        return loss

    optimizer.step(closure)

input_img.data.clamp_(0, 1)

pass


