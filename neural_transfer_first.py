# This .py will be finished when the entire function is working nice
# So just comment the codes

# from __future__ import print_function
# import math
# import copy
#
# import argparse
# import cv2
# import torch
# import torch.optim as optim
# from torchvision import models
# import numpy as np
# import torch.nn as nn
#
# from utils import show_from_cv, toTensor, tensor_to_np, show_from_tensor
# from model import get_model_and_losses
#
#
# parser = argparse.ArgumentParser(description='DeepFake-Pytorch')
# parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                     help='input batch size for training (default: 64)')
# parser.add_argument('--epochs', type=int, default=100000, metavar='N',
#                     help='number of epochs to train (default: 10000)')
# parser.add_argument('--no-cuda', action='store_true',
#                     help='enables CUDA training')
# parser.add_argument('--seed', type=int, default=222, metavar='S',
#                     help='random seed (default: 222)')
# parser.add_argument('--log-interval', type=int, default=100, metavar='N',
#                     help='how many batches to wait before logging training status')
#
#
# style_image = 'datasets/0_target.jpg'
# content_image = 'datasets/0_naive.jpg'
# tmask_image = 'datasets/0_c_mask.jpg'
# mask_image = 'datasets/0_c_mask_dilated.jpg'
#
# content_weight = 5
# style_weight = 100
# tv_weight = 1e-3
# num_iterations = 1000
# normalize_gradients = False
#
# content_layers_default = ['relu_7']
# style_layers_default = ['relu_5', 'relu_7', 'relu_11']
#
#
# def main(content_image, style_weight, ):
#
#     args = parser.parse_args()
#     args.cuda = not args.no_cuda and torch.cuda.is_available()
#
#     torch.manual_seed(args.seed)
#     if args.cuda:
#         torch.cuda.manual_seed(args.seed)
#
#     device = torch.device("cuda:0" if args.cuda else "cpu")
#
#     if args.cuda is True:
#         print('===> Using GPU to train')
#         torch.backends.cudnn.enabled = True
#         torch.backends.cudnn.benchmark = True
#     else:
#         print('===> Using CPU to train')
#
#     cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
#     cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
#
#     print('===> Loaing datasets')
#
#     # use opencv to read image to get 'BGR' image, while we need 'RBG' to sit model
#     content_image = cv2.imread(content_image)
#     print('the image ndarray size is {}'.format(content_image.shape))
#     show_from_cv(content_image)
#     # content_image = transforms.Resize(toTensor(content_image)).type(dtype)
#
#     style_image = cv2.imread(style_image)
#     show_from_cv(style_image)
#
#     mask_image = cv2.imread(mask_image)
#     mask_image_ori = copy.deepcopy(mask_image)
#     show_from_cv(mask_image)
#
#     tmask_image = cv2.imread(tmask_image)  # I'm not sure which img is mask and which is tmask
#     tmask_image_ori = copy.deepcopy(tmask_image)
#     show_from_cv(tmask_image)
#
#     # maybe the gaussianblur needs a bit modification
#     tr = 3
#     tmask_image = cv2.GaussianBlur(tmask_image, (2 * tr + 1, 2 * tr + 1), tr)
#     show_from_cv(tmask_image)
#
#     cnn = models.vgg19(pretrained=True).features.to(device).eval()
#
#     content_image = toTensor(content_image).to(device)
#     style_image = toTensor(style_image).to(device)
#     mask_image = toTensor(mask_image).to(device)
#     tmask_image = toTensor(tmask_image).to(device)
#
#     print('===> Initialize the image...')
#     input_img = content_image.clone()
#     print('the image tensor size is {}'.format(input_img.size()))
#     show_from_tensor(input_img)
#
#
# class Normalization(nn.Module):
#     def __init__(self, mean, std):
#         super(Normalization, self).__init__()
#         # .view the mean and std to make them [C x 1 x 1] so that they can
#         # directly work with image Tensor of shape [B x C x H x W].
#         # B is batch size. C is number of channels. H is height and W is width.
#         self.mean = torch.tensor(mean).view(-1, 1, 1)
#         self.std = torch.tensor(std).view(-1, 1, 1)
#
#     def forward(self, img):
#         # normalize img
#         return (img - self.mean) / self.std
#
#
#
# if __name__ == '__main__':
#     main()
