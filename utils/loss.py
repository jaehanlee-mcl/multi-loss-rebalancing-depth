import torch
from math import exp
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, val_range, window_size=11, window=None, size_average=False, full=False):
    L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs

    return ret

def gradient_dx(img):
    (_, channel, height, width) = img.size()

    kernel_dx = np.zeros((channel, 1, 1, 2))
    kernel_dx[:, :, :, 0] =  1
    kernel_dx[:, :, :, 1] = -1
    kernel_dx = torch.tensor(kernel_dx, device=img.device).float()
    img2kernel = F.conv2d(F.pad(img, pad=(0,1,0,0), mode='replicate'), kernel_dx, padding=0, groups=channel)

    return img2kernel

def gradient_dy(img):
    (_, channel, height, width) = img.size()

    kernel_dy = np.zeros((channel, 1, 2, 1))
    kernel_dy[:, :, 0, :] =  1
    kernel_dy[:, :, 1, :] = -1
    kernel_dy = torch.tensor(kernel_dy, device=img.device).float()

    img2kernel = F.conv2d(F.pad(img, pad=(0,0,0,1), mode='replicate'), kernel_dy, padding=0, groups=channel)

    return img2kernel

def loss_L1_by_channel(pred, gt):

    loss_l1 = torch.mean(torch.abs(pred-gt), dim=[1, 2, 3])

    return loss_l1

def norm_loss(pred_dx, pred_dy, gt_dx, gt_dy):
    (_, channel, height_dx, width_dx) = gt_dx.size()
    (_, channel, height_dy, width_dy) = gt_dy.size()
    height = min(height_dx, height_dy)
    width = min(width_dx, width_dy)

    inner_pred = pred_dx[:, :, 0:height, 0:width] * pred_dx[:, :, 0:height, 0:width] + pred_dy[:, :, 0:height, 0:width] * pred_dy[:, :, 0:height, 0:width] + 1
    inner_gt = gt_dx[:, :, 0:height, 0:width] * gt_dx[:, :, 0:height, 0:width] + gt_dy[:, :, 0:height, 0:width] * gt_dy[:, :, 0:height, 0:width] + 1
    inner_gt_pred = pred_dx[:, :, 0:height, 0:width] * gt_dx[:, :, 0:height, 0:width] + pred_dy[:, :, 0:height, 0:width] * gt_dy[:, :, 0:height, 0:width] + 1

    norm_loss_map = 1 - inner_gt_pred / (torch.pow(inner_pred, 0.5) * torch.pow(inner_gt, 0.5) + 0.00000001)
    norm_loss = torch.mean(norm_loss_map, dim=[1, 2, 3])

    return norm_loss

def loss_for_metric8(pred, gt):
    # Estimate loss of 8 metrics + 2 mean
    #   rmse, rmse_log, abs_rel, sqr_rel, log10, 1-delta1, 1-delta2, 1-delta3
    #   geometric mean of 3 metrics (rmse, abs_rel, 1-delta1)
    #   geometric mean of 8 metrics (all)

    relu = nn.ReLU()

    pred = relu(pred - 0.0000000001) + 0.0000000001

    l_rmse = torch.pow(torch.mean(torch.pow((pred - gt), 2), dim=[1, 2, 3]), 1 / 2)
    l_rmse_log = torch.pow(torch.mean(torch.pow((torch.log(pred) - torch.log(gt)), 2), dim=[1, 2, 3]), 1 / 2)
    l_abs_rel = torch.mean(torch.abs(pred - gt) / gt, dim=[1, 2, 3])
    l_sqr_rel = torch.mean(torch.pow(torch.abs(pred - gt), 2) / gt, dim=[1, 2, 3])
    l_log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(gt)), dim=[1, 2, 3])

    error_for_delta = torch.exp(torch.abs(torch.log(pred / gt)))
    l_delta1 = 1 - torch.mean(torch.le(error_for_delta, 1.25 ** 1).float(), dim=[1, 2, 3])
    l_delta2 = 1 - torch.mean(torch.le(error_for_delta, 1.25 ** 2).float(), dim=[1, 2, 3])
    l_delta3 = 1 - torch.mean(torch.le(error_for_delta, 1.25 ** 3).float(), dim=[1, 2, 3])

    l_metric3 = torch.pow(l_rmse*l_abs_rel*l_delta1, 1/3)
    l_metric8 = torch.pow(l_rmse*l_rmse_log*l_abs_rel*l_sqr_rel*l_log10*l_delta1*l_delta2*l_delta3, 1/8)

    return l_rmse, l_rmse_log, l_abs_rel, l_sqr_rel, l_log10, l_delta1, l_delta2, l_delta3, l_metric3, l_metric8

def loss_for_derivative(pred, gt):
    # estimate loss of derivative
    #   d, dx, dy, d_norm, dx2, dxy, dy2, dx_norm, dy_norm

    pred_dx = gradient_dx(pred)
    pred_dy = gradient_dy(pred)
    pred_dx2 = gradient_dx(pred_dx)
    pred_dxy = gradient_dx(pred_dy)
    pred_dy2 = gradient_dy(pred_dy)

    gt_dx = gradient_dx(gt)
    gt_dy = gradient_dy(gt)
    gt_dx2 = gradient_dx(gt_dx)
    gt_dxy = gradient_dx(gt_dy)
    gt_dy2 = gradient_dy(gt_dy)

    l_depth = loss_L1_by_channel(pred, gt)
    l_depth_dx = loss_L1_by_channel(pred_dx, gt_dx)
    l_depth_dy = loss_L1_by_channel(pred_dy, gt_dy)
    l_depth_dx2 = loss_L1_by_channel(pred_dx2, gt_dx2)
    l_depth_dxy = loss_L1_by_channel(pred_dxy, gt_dxy)
    l_depth_dy2 = loss_L1_by_channel(pred_dy2, gt_dy2)

    l_depth_norm = norm_loss(pred_dx, pred_dy, gt_dx, gt_dy)
    l_depth_dx_norm = norm_loss(pred_dx2, pred_dxy, gt_dx2, gt_dxy)
    l_depth_dy_norm = norm_loss(pred_dxy, pred_dy2, gt_dxy, gt_dy2)

    return l_depth, l_depth_dx, l_depth_dy, l_depth_norm, l_depth_dx2, l_depth_dxy, l_depth_dy2, l_depth_dx_norm, l_depth_dy_norm

def normalized_depth(img, range=0):
    # make input average to zero (in window size (2*range+1) x (2*range+1)

    if range == 0:
        img_avg = torch.mean(img, dim=[1, 2, 3], keepdim=True)
        img_normalized = img - img_avg
    else:
        kernel_size = 2 * range + 1
        padding = (range, range, range, range)
        img_padded = F.pad(img, padding, mode='replicate')
        img_avg = F.avg_pool2d(img_padded, kernel_size, stride=(1, 1))
        img_normalized = img - img_avg

    return img_normalized

def loss_for_normalized_depth(pred, gt, window_size=0):
    # estimate loss for normalized depth

    pred_normalized = normalized_depth(pred, window_size)
    gt_normalized = normalized_depth(gt, window_size)
    l_ndepth = loss_L1_by_channel(pred_normalized, gt_normalized)

    return l_ndepth


