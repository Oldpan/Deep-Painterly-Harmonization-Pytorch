from __future__ import print_function
import argparse
import copy

import cv2
import torch
import torch.optim as optim
from torchvision import models

# -----------------------------------------
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
# -----------------------------------------

from utils import show_from_cv, toTensor, tensor_to_np, show_from_tensor
from model import get_model_and_losses

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

style_image = 'datasets/0_target.jpg'
content_image = 'datasets/0_naive.jpg'
mask_image = 'datasets/0_c_mask_dilated.jpg'
tmask_image = 'datasets/0_c_mask.jpg'

# -------------------------------------------------------------------------
# loader = transforms.Compose([
#     transforms.ToTensor()])  # transform it into a torch tensor
#
#
# def image_loader(image_name):
#     image = Image.open(image_name)
#     # fake batch dimension required to fit network's input dimensions
#     image = loader(image).unsqueeze(0)
#     return image.to(device, torch.float)
#
#
# style_image = image_loader("datasets/0_target.jpg")
# content_image = image_loader("datasets/0_naive.jpg")
#
# unloader = transforms.ToPILImage()  # reconvert into PIL image
#
# plt.ion()
#
#
# def imshow(tensor, title=None):
#     image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
#     image = image.squeeze(0)  # remove the fake batch dimension
#     image = unloader(image)
#     plt.imshow(image)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated
#
# plt.figure()
# imshow(style_image, title='Style Image')
#
# plt.figure()
# imshow(content_image, title='Content Image')
# -------------------------------------------------------------------------

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

print('===> Loaing datasets')

# use opencv to read image to get 'BGR' image, while we need 'RGB' to sit model
content_image = cv2.imread(content_image)
print('the image ndarray size is {}'.format(content_image.shape))
show_from_cv(content_image)
# content_image = transforms.Resize(toTensor(content_image)).type(dtype)

style_image = cv2.imread(style_image)
show_from_cv(style_image)

mask_image = cv2.imread(mask_image)
mask_image_ori = copy.deepcopy(mask_image)
show_from_cv(mask_image)

tmask_image = cv2.imread(tmask_image)
tmask_image_ori = copy.deepcopy(tmask_image)
show_from_cv(tmask_image)

tr = 3
tmask_image = cv2.GaussianBlur(tmask_image, (2 * tr + 1, 2 * tr + 1), tr)
show_from_cv(tmask_image)

cnn = models.vgg19(pretrained=True).features.to(device).eval()
#
content_image = toTensor(content_image).to(device, torch.float)
style_image = toTensor(style_image).to(device, torch.float)
mask_image = toTensor(mask_image).to(device, torch.float)
tmask_image = toTensor(tmask_image).to(device, torch.float)

# a = mask_image.reshape(1,-1)
#
# for i in range(0, 1432200):
#     print(a[0][i])

print('===> Initialize the image...')
# input_img = torch.randn(content_image.data.size(), device=device)
input_img = content_image.clone()
print('the image tensor size is {}'.format(input_img.size()))
show_from_tensor(input_img)


def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def run_painterly_transfer(cnn, normalization_mean, normalization_std,
                           style_img, content_img, mask_img, tmask_img, num_steps=1000,
                           style_weight=1000, content_weight=1000000, tv_weight=0):
    print('===> Building the painterly model...')
    model, style_loss, content_loss, tv_loss = get_model_and_losses(cnn, normalization_mean, normalization_std,
                                                                    style_img, content_img, mask_img, tmask_img,
                                                                    style_weight, content_weight, tv_weight)

    optimizer = get_input_optimizer(input_img)

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
                show_from_tensor(new_image)
                # show_from_tensor(input_img)
                # plt.figure()
                # imshow(input_img, title='Output Image')

            return style_score + content_score

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)

    return input_img


if __name__ == '__main__':
    output = run_painterly_transfer(cnn, cnn_normalization_mean, cnn_normalization_std, style_image,
                                    content_image, mask_image, tmask_image)

    pass
