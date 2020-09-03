from utils.loss import loss_for_metric8, loss_for_derivative, loss_for_normalized_depth
import torch
import torch.nn as nn
import numpy as np

def compute_multi_metric_with_record(depth_pred_for_metric, depth_gt_for_metric, metric_valid, batch_size, current_batch_size, i, num_test_data, test_metrics):
    rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3\
        = compute_multi_metric(depth_pred_for_metric, depth_gt_for_metric, metric_valid)

    test_metrics = get_metric_1batch(batch_size, current_batch_size, i, num_test_data, test_metrics,
                                     rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3
                                     )
    return test_metrics

def compute_multi_metric(depth_pred_for_metric, depth_gt_for_metric):
    ## METRIC LIST
    rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3, metric3, metric8 \
        = loss_for_metric8(depth_pred_for_metric, depth_gt_for_metric)
    delta1 = 1 - delta1
    delta2 = 1 - delta2
    delta3 = 1 - delta3
    metrics = [rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3]

    return metrics

def get_metric_1batch(batch_size, current_batch_size, index_iter, num_data, metrics_recoder, metrics):
    for index_batch in range(current_batch_size):
        index_record = batch_size * index_iter + index_batch
        if index_record < num_data:
            metrics_recoder[index_record, :] = [
                metrics[0][index_batch], metrics[1][index_batch], metrics[2][index_batch], metrics[3][index_batch], metrics[4][index_batch],
                metrics[5][index_batch], metrics[6][index_batch], metrics[7][index_batch],
            ]
    return metrics_recoder

def get_loss_weights():
    loss_weights = [
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d0_depth,        l_d0_depth_dx,      l_d0_depth_dy,          l_d0_depth_norm,        l_d0_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d0_depth_dxy,    l_d0_depth_dy2,     l_d0_depth_dx_norm,     l_d0_depth_dy_norm,     l_d0_ndepth,
            0.01282051,         0.01282051,         0.01282051,                                         # l_d0_ndepth_win5,  l_d0_ndepth_win17,  l_d0_ndepth_win65,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ndepth,
            0.01282051,         0.01282051,         0.01282051,                                         # l_d1_ndepth_win5,  l_d1_ndepth_win17,  l_d1_ndepth_win65,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ndepth,
            0.01282051,         0.01282051,         0.01282051,                                         # l_d2_ndepth_win5,  l_d2_ndepth_win17,  l_d2_ndepth_win65,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ndepth,
            0.01282051,         0.01282051,         0.01282051,                                         # l_d3_ndepth_win5,  l_d3_ndepth_win17,  l_d3_ndepth_win65,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ndepth,
            0.01282051,         0.01282051,         0.01282051,                                         # l_d4_ndepth_win5,  l_d4_ndepth_win17,  l_d4_ndepth_win65,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ndepth,
            0.01282051,         0.01282051,         0.01282051,                                         # l_d5_ndepth_win5,  l_d5_ndepth_win17,  l_d5_ndepth_win65,
        ]
    for index in range(0, len(loss_weights)):
        loss_weights[index] = loss_weights[index] * 1.0000

    return loss_weights

def get_loss_initialize_scale():
    loss_initialize_scale = [
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d0_depth,        l_d0_depth_dx,      l_d0_depth_dy,          l_d0_depth_norm,        l_d0_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d0_depth_dxy,    l_d0_depth_dy2,     l_d0_depth_dx_norm,     l_d0_depth_dy_norm,     l_d0_ndepth,
            0.01282051,         0.01282051,         0.01282051,                                         # l_d0_ndepth_win5,  l_d0_ndepth_win17,  l_d0_ndepth_win65,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d1_depth,        l_d1_depth_dx,      l_d1_depth_dy,          l_d1_depth_norm,        l_d1_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d1_depth_dxy,    l_d1_depth_dy2,     l_d1_depth_dx_norm,     l_d1_depth_dy_norm,     l_d1_ndepth,
            0.01282051,         0.01282051,         0.01282051,                                         # l_d1_ndepth_win5,  l_d1_ndepth_win17,  l_d1_ndepth_win65,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d2_depth,        l_d2_depth_dx,      l_d2_depth_dy,          l_d2_depth_norm,        l_d2_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d2_depth_dxy,    l_d2_depth_dy2,     l_d2_depth_dx_norm,     l_d2_depth_dy_norm,     l_d2_ndepth,
            0.01282051,         0.01282051,         0.01282051,                                         # l_d2_ndepth_win5,  l_d2_ndepth_win17,  l_d2_ndepth_win65,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d3_depth,        l_d3_depth_dx,      l_d3_depth_dy,          l_d3_depth_norm,        l_d3_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d3_depth_dxy,    l_d3_depth_dy2,     l_d3_depth_dx_norm,     l_d3_depth_dy_norm,     l_d3_ndepth,
            0.01282051,         0.01282051,         0.01282051,                                         # l_d3_ndepth_win5,  l_d3_ndepth_win17,  l_d3_ndepth_win65,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d4_depth,        l_d4_depth_dx,      l_d4_depth_dy,          l_d4_depth_norm,        l_d4_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d4_depth_dxy,    l_d4_depth_dy2,     l_d4_depth_dx_norm,     l_d4_depth_dy_norm,     l_d4_ndepth,
            0.01282051,         0.01282051,         0.01282051,                                         # l_d4_ndepth_win5,  l_d4_ndepth_win17,  l_d4_ndepth_win65,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d5_depth,        l_d5_depth_dx,      l_d5_depth_dy,          l_d5_depth_norm,        l_d5_depth_dx2,
            0.01282051,         0.01282051,         0.01282051,         0.01282051,         0.01282051, # l_d5_depth_dxy,    l_d5_depth_dy2,     l_d5_depth_dx_norm,     l_d5_depth_dy_norm,     l_d5_ndepth,
            0.01282051,         0.01282051,         0.01282051,                                         # l_d5_ndepth_win5,  l_d5_ndepth_win17,  l_d5_ndepth_win65,
        ]
    return loss_initialize_scale

def compute_multi_loss(depth_pred_for_loss, depth_gt_for_loss, loss_valid):
    _, channel, height, width = depth_pred_for_loss.size()

    # Loss
    # interpolation function
    interpolate_bicubic_div02 = nn.Upsample(scale_factor=1 / 2, mode='bicubic')
    interpolate_bicubic_div04 = nn.Upsample(scale_factor=1 / 4, mode='bicubic')
    interpolate_bicubic_div08 = nn.Upsample(scale_factor=1 / 8, mode='bicubic')
    interpolate_bicubic_div16 = nn.Upsample(scale_factor=1 / 16, mode='bicubic')
    interpolate_bicubic_div32 = nn.Upsample(scale_factor=1 / 32, mode='bicubic')
    interpolate_bicubic_1by1 = nn.Upsample(size=[1, 1], mode='bicubic')

    # resize map
    # size 288x384
    depth_pred_for_loss_down0 = depth_pred_for_loss
    depth_gt_for_loss_down0 = depth_gt_for_loss
    # size 144x192
    if min(height, width) >= pow(2,1):
        depth_pred_for_loss_down1 = interpolate_bicubic_div02(depth_pred_for_loss)
        depth_gt_for_loss_down1 = interpolate_bicubic_div02(depth_gt_for_loss)
    else:
        depth_pred_for_loss_down1 = interpolate_bicubic_1by1(depth_pred_for_loss)
        depth_gt_for_loss_down1 = interpolate_bicubic_1by1(depth_gt_for_loss)
    # size 72x96
    if min(height, width) >= pow(2, 2):
        depth_pred_for_loss_down2 = interpolate_bicubic_div04(depth_pred_for_loss)
        depth_gt_for_loss_down2 = interpolate_bicubic_div04(depth_gt_for_loss)
    else:
        depth_pred_for_loss_down2 = interpolate_bicubic_1by1(depth_pred_for_loss)
        depth_gt_for_loss_down2 = interpolate_bicubic_1by1(depth_gt_for_loss)
    # size 36x48
    if min(height, width) >= pow(2, 3):
        depth_pred_for_loss_down3 = interpolate_bicubic_div08(depth_pred_for_loss)
        depth_gt_for_loss_down3 = interpolate_bicubic_div08(depth_gt_for_loss)
    else:
        depth_pred_for_loss_down3 = interpolate_bicubic_1by1(depth_pred_for_loss)
        depth_gt_for_loss_down3 = interpolate_bicubic_1by1(depth_gt_for_loss)
    # size 18x24
    if min(height, width) >= pow(2, 4):
        depth_pred_for_loss_down4 = interpolate_bicubic_div16(depth_pred_for_loss)
        depth_gt_for_loss_down4 = interpolate_bicubic_div16(depth_gt_for_loss)
    else:
        depth_pred_for_loss_down4 = interpolate_bicubic_1by1(depth_pred_for_loss)
        depth_gt_for_loss_down4 = interpolate_bicubic_1by1(depth_gt_for_loss)
    # size 9x12
    if min(height, width) >= pow(2, 5):
        depth_pred_for_loss_down5 = interpolate_bicubic_div32(depth_pred_for_loss)
        depth_gt_for_loss_down5 = interpolate_bicubic_div32(depth_gt_for_loss)
    else:
        depth_pred_for_loss_down5 = interpolate_bicubic_1by1(depth_pred_for_loss)
        depth_gt_for_loss_down5 = interpolate_bicubic_1by1(depth_gt_for_loss)

    current_batch_size = depth_gt_for_loss.size(0)
    invalid_input = torch.zeros(current_batch_size).cuda(torch.device("cuda:0"))

    ## LOSS LIST
    if sum(loss_valid[0:9]) > 0:
        l_down0_depth, l_down0_depth_dx, l_down0_depth_dy, l_down0_depth_norm, l_down0_depth_dx2, l_down0_depth_dxy, l_down0_depth_dy2, l_down0_depth_dx_norm, l_down0_depth_dy_norm \
            = loss_for_derivative(depth_pred_for_loss_down0, depth_gt_for_loss_down0)
    else:
        l_down0_depth, l_down0_depth_dx, l_down0_depth_dy, l_down0_depth_norm, l_down0_depth_dx2, l_down0_depth_dxy, l_down0_depth_dy2, l_down0_depth_dx_norm, l_down0_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if sum(loss_valid[9:13]) > 0:
        l_down0_ndepth = loss_for_normalized_depth(depth_pred_for_loss_down0, depth_gt_for_loss_down0, window_size=0)
        l_down0_ndepth_win5 = loss_for_normalized_depth(depth_pred_for_loss_down0, depth_gt_for_loss_down0, window_size=2)
        l_down0_ndepth_win17 = loss_for_normalized_depth(depth_pred_for_loss_down0, depth_gt_for_loss_down0, window_size=8)
        l_down0_ndepth_win65 = loss_for_normalized_depth(depth_pred_for_loss_down0, depth_gt_for_loss_down0, window_size=32)
    else:
        l_down0_ndepth, l_down0_ndepth_win5, l_down0_ndepth_win17, l_down0_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    if sum(loss_valid[13:22]) > 0:
        l_down1_depth, l_down1_depth_dx, l_down1_depth_dy, l_down1_depth_norm, l_down1_depth_dx2, l_down1_depth_dxy, l_down1_depth_dy2, l_down1_depth_dx_norm, l_down1_depth_dy_norm \
            = loss_for_derivative(depth_pred_for_loss_down1, depth_gt_for_loss_down1)
    else:
        l_down1_depth, l_down1_depth_dx, l_down1_depth_dy, l_down1_depth_norm, l_down1_depth_dx2, l_down1_depth_dxy, l_down1_depth_dy2, l_down1_depth_dx_norm, l_down1_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if sum(loss_valid[22:26]) > 0:
        l_down1_ndepth = loss_for_normalized_depth(depth_pred_for_loss_down1, depth_gt_for_loss_down1, window_size=0)
        l_down1_ndepth_win5 = loss_for_normalized_depth(depth_pred_for_loss_down1, depth_gt_for_loss_down1, window_size=2)
        l_down1_ndepth_win17 = loss_for_normalized_depth(depth_pred_for_loss_down1, depth_gt_for_loss_down1, window_size=8)
        l_down1_ndepth_win65 = loss_for_normalized_depth(depth_pred_for_loss_down1, depth_gt_for_loss_down1, window_size=32)
    else:
        l_down1_ndepth, l_down1_ndepth_win5, l_down1_ndepth_win17, l_down1_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    if sum(loss_valid[26:35]) > 0:
        l_down2_depth, l_down2_depth_dx, l_down2_depth_dy, l_down2_depth_norm, l_down2_depth_dx2, l_down2_depth_dxy, l_down2_depth_dy2, l_down2_depth_dx_norm, l_down2_depth_dy_norm \
            = loss_for_derivative(depth_pred_for_loss_down2, depth_gt_for_loss_down2)
    else:
        l_down2_depth, l_down2_depth_dx, l_down2_depth_dy, l_down2_depth_norm, l_down2_depth_dx2, l_down2_depth_dxy, l_down2_depth_dy2, l_down2_depth_dx_norm, l_down2_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if sum(loss_valid[35:39]) > 0:
        l_down2_ndepth = loss_for_normalized_depth(depth_pred_for_loss_down2, depth_gt_for_loss_down2, window_size=0)
        l_down2_ndepth_win5 = loss_for_normalized_depth(depth_pred_for_loss_down2, depth_gt_for_loss_down2, window_size=2)
        l_down2_ndepth_win17 = loss_for_normalized_depth(depth_pred_for_loss_down2, depth_gt_for_loss_down2, window_size=8)
        l_down2_ndepth_win65 = loss_for_normalized_depth(depth_pred_for_loss_down2, depth_gt_for_loss_down2, window_size=32)
    else:
        l_down2_ndepth, l_down2_ndepth_win5, l_down2_ndepth_win17, l_down2_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    if sum(loss_valid[39:48]) > 0:
        l_down3_depth, l_down3_depth_dx, l_down3_depth_dy, l_down3_depth_norm, l_down3_depth_dx2, l_down3_depth_dxy, l_down3_depth_dy2, l_down3_depth_dx_norm, l_down3_depth_dy_norm \
            = loss_for_derivative(depth_pred_for_loss_down3, depth_gt_for_loss_down3)
    else:
        l_down3_depth, l_down3_depth_dx, l_down3_depth_dy, l_down3_depth_norm, l_down3_depth_dx2, l_down3_depth_dxy, l_down3_depth_dy2, l_down3_depth_dx_norm, l_down3_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if sum(loss_valid[48:52]) > 0:
        l_down3_ndepth = loss_for_normalized_depth(depth_pred_for_loss_down3, depth_gt_for_loss_down3, window_size=0)
        l_down3_ndepth_win5 = loss_for_normalized_depth(depth_pred_for_loss_down3, depth_gt_for_loss_down3, window_size=2)
        l_down3_ndepth_win17 = loss_for_normalized_depth(depth_pred_for_loss_down3, depth_gt_for_loss_down3, window_size=8)
        l_down3_ndepth_win65 = loss_for_normalized_depth(depth_pred_for_loss_down3, depth_gt_for_loss_down3, window_size=32)
    else:
        l_down3_ndepth, l_down3_ndepth_win5, l_down3_ndepth_win17, l_down3_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    if sum(loss_valid[52:61]) > 0:
        l_down4_depth, l_down4_depth_dx, l_down4_depth_dy, l_down4_depth_norm, l_down4_depth_dx2, l_down4_depth_dxy, l_down4_depth_dy2, l_down4_depth_dx_norm, l_down4_depth_dy_norm \
            = loss_for_derivative(depth_pred_for_loss_down4, depth_gt_for_loss_down4)
    else:
        l_down4_depth, l_down4_depth_dx, l_down4_depth_dy, l_down4_depth_norm, l_down4_depth_dx2, l_down4_depth_dxy, l_down4_depth_dy2, l_down4_depth_dx_norm, l_down4_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if sum(loss_valid[61:65]) > 0:
        l_down4_ndepth = loss_for_normalized_depth(depth_pred_for_loss_down4, depth_gt_for_loss_down4, window_size=0)
        l_down4_ndepth_win5 = loss_for_normalized_depth(depth_pred_for_loss_down4, depth_gt_for_loss_down4, window_size=2)
        l_down4_ndepth_win17 = loss_for_normalized_depth(depth_pred_for_loss_down4, depth_gt_for_loss_down4, window_size=8)
        l_down4_ndepth_win65 = loss_for_normalized_depth(depth_pred_for_loss_down4, depth_gt_for_loss_down4, window_size=32)
    else:
        l_down4_ndepth, l_down4_ndepth_win5, l_down4_ndepth_win17, l_down4_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    if sum(loss_valid[65:74]) > 0:
        l_down5_depth, l_down5_depth_dx, l_down5_depth_dy, l_down5_depth_norm, l_down5_depth_dx2, l_down5_depth_dxy, l_down5_depth_dy2, l_down5_depth_dx_norm, l_down5_depth_dy_norm \
            = loss_for_derivative(depth_pred_for_loss_down5, depth_gt_for_loss_down5)
    else:
        l_down5_depth, l_down5_depth_dx, l_down5_depth_dy, l_down5_depth_norm, l_down5_depth_dx2, l_down5_depth_dxy, l_down5_depth_dy2, l_down5_depth_dx_norm, l_down5_depth_dy_norm \
            = invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input, invalid_input

    if sum(loss_valid[74:78]) > 0:
        l_down5_ndepth = loss_for_normalized_depth(depth_pred_for_loss_down5, depth_gt_for_loss_down5, window_size=0)
        l_down5_ndepth_win5 = loss_for_normalized_depth(depth_pred_for_loss_down5, depth_gt_for_loss_down5, window_size=2)
        l_down5_ndepth_win17 = loss_for_normalized_depth(depth_pred_for_loss_down5, depth_gt_for_loss_down5, window_size=8)
        l_down5_ndepth_win65 = loss_for_normalized_depth(depth_pred_for_loss_down5, depth_gt_for_loss_down5, window_size=32)
    else:
        l_down5_ndepth, l_down5_ndepth_win5, l_down5_ndepth_win17, l_down5_ndepth_win65 \
            = invalid_input, invalid_input, invalid_input, invalid_input

    losses = \
        [  l_down0_depth, l_down0_depth_dx, l_down0_depth_dy, l_down0_depth_norm, l_down0_depth_dx2, l_down0_depth_dxy, l_down0_depth_dy2, l_down0_depth_dx_norm, l_down0_depth_dy_norm, l_down0_ndepth, l_down0_ndepth_win5, l_down0_ndepth_win17, l_down0_ndepth_win65, \
           l_down1_depth, l_down1_depth_dx, l_down1_depth_dy, l_down1_depth_norm, l_down1_depth_dx2, l_down1_depth_dxy, l_down1_depth_dy2, l_down1_depth_dx_norm, l_down1_depth_dy_norm, l_down1_ndepth, l_down1_ndepth_win5, l_down1_ndepth_win17, l_down1_ndepth_win65, \
           l_down2_depth, l_down2_depth_dx, l_down2_depth_dy, l_down2_depth_norm, l_down2_depth_dx2, l_down2_depth_dxy, l_down2_depth_dy2, l_down2_depth_dx_norm, l_down2_depth_dy_norm, l_down2_ndepth, l_down2_ndepth_win5, l_down2_ndepth_win17, l_down2_ndepth_win65, \
           l_down3_depth, l_down3_depth_dx, l_down3_depth_dy, l_down3_depth_norm, l_down3_depth_dx2, l_down3_depth_dxy, l_down3_depth_dy2, l_down3_depth_dx_norm, l_down3_depth_dy_norm, l_down3_ndepth, l_down3_ndepth_win5, l_down3_ndepth_win17, l_down3_ndepth_win65, \
           l_down4_depth, l_down4_depth_dx, l_down4_depth_dy, l_down4_depth_norm, l_down4_depth_dx2, l_down4_depth_dxy, l_down4_depth_dy2, l_down4_depth_dx_norm, l_down4_depth_dy_norm, l_down4_ndepth, l_down4_ndepth_win5, l_down4_ndepth_win17, l_down4_ndepth_win65, \
           l_down5_depth, l_down5_depth_dx, l_down5_depth_dy, l_down5_depth_norm, l_down5_depth_dx2, l_down5_depth_dxy, l_down5_depth_dy2, l_down5_depth_dx_norm, l_down5_depth_dy_norm, l_down5_ndepth, l_down5_ndepth_win5, l_down5_ndepth_win17, l_down5_ndepth_win65
         ]

    return losses

def get_loss_1batch(batch_size, current_batch_size, index_iter, num_data, loss_weights, scores, losses):

    num_losses = len(losses)
    l_custom = torch.zeros(current_batch_size).cuda(torch.device("cuda:0"))
    for index_batch in range(current_batch_size):
        index_record = batch_size * index_iter + index_batch

        if index_batch == 0:
            loss = 0

        if index_record < num_data:
            loss_1batch = 0
            for index_loss in range(num_losses):
                if loss_weights[index_loss] != 0:
                    loss_1batch = loss_1batch + loss_weights[index_loss] * losses[index_loss][index_batch]

            l_custom[index_batch] = loss_1batch
            loss = loss + loss_1batch / current_batch_size

            for index_loss in range(num_losses):
                scores[index_record, index_loss] = losses[index_loss][index_batch]

    return loss, l_custom, scores