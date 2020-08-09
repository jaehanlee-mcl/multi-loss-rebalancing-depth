import time
import argparse

import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd

from PIL import Image
from networks.model import create_model
from utils.get_data import getTestingData
from utils.utils import print_metrics
from utils.multi_loss import compute_multi_metric, get_metric_1batch

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Arguments
    parser = argparse.ArgumentParser(description='Monocular Depth')
    parser.add_argument('--backbone', default='PNASNet5Large', type=str)
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--save_prediction', default=True, type=bool)
    parser.add_argument('--get_metric', default=True, type=bool)
    args = parser.parse_args()

    # dataset using
    test_dataset_use = {'NYUv2_test': True}

    # image size
    original_image_size = [480, 640]
    input_image_size = [288, 384]
    # interpolation function / relu
    interpolate_bicubic_fullsize = nn.Upsample(size=original_image_size, mode='bicubic')
    interpolate_bicubic_inputsize = nn.Upsample(size=input_image_size, mode='bicubic')
    relu = nn.ReLU()

    # Create model
    model = nn.DataParallel(create_model(args.backbone).half())
    print('Model created.')

    # loading training/testing data
    batch_size = args.batch_size
    test_loader, num_test_data = getTestingData(batch_size, test_dataset_use)

    # model path
    model_path = 'models'
    model_name = 'PNAS_model.pth'
    pred_path = 'prediction/PNAS_model'

    # prediction path
    if os.path.isdir(pred_path) == False:
        os.mkdir(pred_path)

    # Start testing
    N = len(test_loader)
    end = time.time()
    total_time = time.time() - end

    # load
    model.load_state_dict(torch.load(model_path + '/' + model_name))
    model.eval()
    print(model_path + '/' + model_name)
    test_metrics = np.zeros((num_test_data, 8)) # 8 metrics

    for i, sample_batched in enumerate(test_loader):

        # Prepare sample and target
        image = torch.autograd.Variable(sample_batched['image'].cuda()).half()
        depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        depth_gt_for_metric = depth_gt[:, :, 0 + 20:480 - 20, 0 + 24:640 - 24].cuda()
        current_batch_size = depth_gt.size(0)

        # Predict
        image_input = interpolate_bicubic_inputsize(image)
        depth_pred_for_loss = model(image_input)
        depth_pred_for_loss = (relu(depth_pred_for_loss - 0.0001) + 0.0001).float()
        depth_pred_full = interpolate_bicubic_fullsize(depth_pred_for_loss)
        depth_pred_for_metric = (relu(depth_pred_full[:, :, 0 + 20:480 - 20, 0 + 24:640 - 24] - 0.0001) + 0.0001).cuda()

        # save prediction
        if args.save_prediction == True:
            for index_test in range(i * batch_size + 1, i * batch_size + current_batch_size + 1):
                pred2png(depth_pred_full[index_test - (i * batch_size + 1), 0, :, :].cpu().detach().numpy(), pred_path, index_test)
        # compute metric
        if args.get_metric == True:
            rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3 \
                = compute_multi_metric(depth_pred_for_metric, depth_gt_for_metric)
            test_metrics = get_metric_1batch(batch_size, current_batch_size, i, num_test_data, test_metrics,
                                             rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3)

        # Measure elapsed time
        total_time = total_time + (time.time() - end)
        end = time.time()
        # Print to console
        if i == 0 or i % 20 == 19 or i == N - 1:
            print('Evaluation - ', str(i+1).zfill(5), '/', str(N).zfill(5), '    Time: ', str('%10.2f' % total_time), 's')

    if args.get_metric == True:
        test_metrics_mean = test_metrics.mean(axis=0)
        print_metrics(test_metrics_mean)
        # save metrics
        dataframe = pd.DataFrame(test_metrics)
        dataframe.to_csv(model_path + "/metrics" + ".csv", header=False, index=False)
        dataframe = pd.DataFrame(test_metrics_mean)
        dataframe.to_csv(model_path + "/metrics_mean" + ".csv", header=False, index=False)

    print('------------------------ FINISH -------------------------')
    print('---------------------------------------------------------')

def pred2png(data, data_path, index_data):
    data = (data / 10 * (pow(2,16)-1)).astype(np.uint16)
    data_name = data_path + '/pred' + str(index_data).zfill(4) + '.png'
    data_image = Image.fromarray(data)
    data_image.save(data_name)

if __name__ == '__main__':
    main()