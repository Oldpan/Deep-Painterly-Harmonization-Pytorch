from __future__ import print_function
import argparse
import copy

import os.path as osp
import os
import time
import math
import numpy as np
import torch
import torch.optim as optim
from torchvision import models

from PIL import Image, ImageFilter
import matplotlib.pyplot as plt

import torchvision.transforms as transforms

from utils import show_from_cv, toTensor, tensor_to_np, show_from_tensor
from model_PIL import get_model_and_losses

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

device = torch.device("cuda:0" if args.cuda else "cpu")

if args.cuda is True:
    print('===> Using GPU to train')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    print('===> Using CPU to train')

# -------------------------------------------------------------------------
loader = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor


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
    num = 13
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
content_image = image_loader("datasets/0_naive.jpg")
mask_image = image_loader('datasets/0_c_mask_dilated.jpg')[:, 0:1, :, :]
mask_image_ori = mask_image.clone()
tmask_image = Image.open('datasets/0_c_mask.jpg').convert('RGB')
tmask_image = tmask_image.filter(ImageFilter.GaussianBlur())
tmask_image = PIL_to_tensor(tmask_image)
tmask_image_ori = tmask_image.clone()

unloader = transforms.ToPILImage()  # reconvert into PIL image

plt.ion()

# plt.figure()
# imshow(style_image, title='Style Image')
#
# plt.figure()
# imshow(content_image, title='Content Image')
#
# plt.figure()
# imshow(mask_image, title='Mask Image')
#
# plt.figure()
# imshow(tmask_image, title='t-Mask Image')
# -------------------------------------------------------------------------

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

cnn = models.vgg19(pretrained=True).features.to(device).eval()

print('===> Initialize the image...')
input_img = content_image.clone()


# plt.figure()
# imshow(input_img, title='Input Image')


def get_input_optimizer(input_img, lr):
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=lr)
    return optimizer


def model_hook(module, grad_input, grad_output):
    mask = mask_image_ori.clone()

    print('grad_input:{} is matchable with mask:{}'.format(grad_input[0].shape, mask.shape))
    if isinstance(mask, torch.Tensor):
        mask = tensor_to_PIL(mask)
    resize = transforms.Resize((math.floor(grad_input[0].shape[2] / 2), math.floor(grad_input[0].shape[3] / 2)))
    mask = resize(mask)
    mask = PIL_to_tensor(mask).to(device)

    if grad_input[0].shape == mask.shape:
        print('grad_input:{} is matchable with mask:{}'.format(grad_input[0].shape, mask.shape))
        mask = mask.expand_as(grad_input[0])
        grad_input_1 = grad_input[0] * mask
        grad_input = tuple([grad_input_1, grad_input[1], grad_input[2]])

    return grad_input


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
                                    num_steps=1000,
                                    style_weight=1000000, content_weight=50, tv_weight=0, lr=0.1)
    time_elapsed = time.time() - since
    print('The time used is {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    pass
