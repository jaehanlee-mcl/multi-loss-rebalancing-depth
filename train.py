import time
import argparse
import datetime
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd

import networks.model as network_model
import utils.utils as utils_utils
import utils.get_data as utils_get_data
import utils.multi_loss as utils_multi_loss
try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Arguments
    parser = argparse.ArgumentParser(description='Multi-Loss Rebalancing Algorithm for Monocular Depth Estimation')
    parser.add_argument('--backbone', default='PNASNet5Large', type=str,
                        help='DenseNet161 (bs12) / PNASNet5Largea (bs6)')
    parser.add_argument('--decoder_scale', default=1024, type=int, help='valid for PNASNet5Large')
    parser.add_argument('--epochs', default=20, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--bs', default=8, type=int, help='batch size')
    parser.add_argument('--weight_initialization', default=True, type=bool)
    parser.add_argument('--weight_rebalancing', default=True, type=bool)
    parser.add_argument('--num_weight_rebalancing_per_epoch', default=4, type=int)
    parser.add_argument('--num_save_per_epoch', default=4, type=int)
    parser.add_argument('--lambda_for_adjust_start', default=3, type=float)
    parser.add_argument('--lambda_for_adjust_slope', default=-1.5, type=float)
    parser.add_argument('--lambda_for_adjust_min', default=-3, type=float)
    #parser.add_argument('--train_dataset_path', default='dataset/train_reduced05.zip', type=str)
    #parser.add_argument('--train_dataset_csv_list', default='train_reduced05/train.csv', type=str)
    parser.add_argument('--train_dataset_path', default='dataset/train795.zip', type=str)
    parser.add_argument('--train_dataset_csv_list', default='train795/train.csv', type=str)
    args = parser.parse_args()

    # image size
    original_image_size = [480, 640]
    input_image_size = [288, 384]
    # interpolation function / relu
    interpolate_bicubic_fullsize = nn.Upsample(size=original_image_size, mode='bicubic')
    relu = nn.ReLU()

    # create model
    model = network_model.create_model(args.backbone, args.decoder_scale)
    print('Summary: All Network')
    print(utils_utils.get_model_summary(model, torch.rand(1, 3, input_image_size[0], input_image_size[1]).cuda(), verbose=True))
    print('Model created.')

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    batch_size = args.bs

    # loading training/testing data
    train_loader, num_train_data = utils_get_data.getTrainingData(batch_size, args.train_dataset_path, args.train_dataset_csv_list)

    # Model path
    model_path = utils_utils.make_model_path(args.backbone, args.decoder_scale, batch_size)

    # train scores
    train_scores = np.zeros((num_train_data, 78))  # 78 scores
    train_metrics = np.zeros((num_train_data, 8)) # 8 metrics

    # loss term
    loss_weights = utils_multi_loss.get_loss_weights()
    loss_initialize_scale = utils_multi_loss.get_loss_initialize_scale()
    loss_valid = np.array(loss_weights) > 0

    # save path
    savePath = model_path + '/weight/loss_weights.csv'
    dataframe = pd.DataFrame(loss_weights)
    dataframe.to_csv(savePath, header=False, index=False)

    # weight rebalancing argument
    weight_initialization = args.weight_initialization
    weight_rebalancing = args.weight_rebalancing
    weight_initialization_done = False
    last_rebalancing_iter = 0
    previous_total_loss = 0
    previous_loss = 0

    # iter/epoch
    iter_per_epoch = len(train_loader)

    # save iteration
    iter_list_save = utils_utils.get_notable_iter(iter_per_epoch, num_per_epoch=args.num_save_per_epoch)
    iter_list_rebalancing = utils_utils.get_notable_iter(iter_per_epoch, num_per_epoch=args.num_weight_rebalancing_per_epoch)

    # mixed precision + Dataparallel
    if APEX_AVAILABLE == True:
        use_amp = True
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O2",
            keep_batchnorm_fp32=True, loss_scale="dynamic"
        )
    else:
        use_amp = False
    model = nn.DataParallel(model)

    try:
        # try to load epoch1_iter00000
        model_name = "model/epoch01_iter00000.pth"
        model.load_state_dict(torch.load(model_name))
        print('LOAD MODEL ', model_name)
    except:
        # save model
        print('THERE IS NO MODEL TO LOAD')
        model_name = model_path + "/model/epoch" + str(0 + 1).zfill(2) + '_iter' + str(0).zfill(5) + ".pth"
        print('SAVE MODEL:' + model_path)
        torch.save(model.state_dict(), model_name)

    # Start training...
    for epoch in range(args.epochs):
        print('---------------------------------------------------------')
        print('-------------- TRAINING OF EPOCH ' + str(0 + epoch + 1).zfill(2) + 'START ----------------')

        end = time.time()

        # Switch to train mode
        model.train()

        # train parameter
        current_lambda_for_adjust = max(args.lambda_for_adjust_start + epoch * args.lambda_for_adjust_slope, args.lambda_for_adjust_min)

        for i, sample_batched in enumerate(train_loader):
            optimizer.zero_grad()

            # Prepare sample and target
            image = torch.autograd.Variable(sample_batched['image'].cuda())
            depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))

            # depth gt
            depth_gt_input = depth_gt
            depth_gt_full = interpolate_bicubic_fullsize(depth_gt_input)
            depth_gt_for_loss = depth_gt_input
            depth_gt_for_loss = depth_gt_for_loss.cuda()
            depth_gt_for_metric = (relu(depth_gt_full[:, :, 0 + 20:480 - 20, 0 + 24:640 - 24] - 0.0001) + 0.0001)

            # Predict
            image_input = image
            depth_pred_for_loss = model(image_input).cuda()
            depth_pred_full = interpolate_bicubic_fullsize(depth_pred_for_loss)
            depth_pred_for_metric = (relu(depth_pred_full[:, :, 0 + 20:480 - 20, 0 + 24:640 - 24] - 0.0001) + 0.0001)

            # current batch size
            current_batch_size = depth_gt_for_loss.size(0)

            # compute loss
            losses = utils_multi_loss.compute_multi_loss(depth_pred_for_loss, depth_gt_for_loss, loss_valid)

            # compute iter loss & train_scores
            loss, l_custom, train_scores = utils_multi_loss.get_loss_1batch(batch_size, current_batch_size, i, num_train_data, loss_weights, train_scores, losses)
            metrics = utils_multi_loss.compute_multi_metric(depth_pred_for_metric, depth_gt_for_metric)
            train_metrics = utils_multi_loss.get_metric_1batch(batch_size, current_batch_size, i, num_train_data, train_metrics, metrics)

            # Update
            if use_amp == True:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            # Measure elapsed time
            end = time.time()

            # Log progress
            if (i+1) % 100 == 0 or (i+1) == iter_per_epoch:
                if epoch == 0:
                    train_scores_mean = train_scores[0:(i+1)*batch_size].mean(axis=0)
                    train_metrics_mean = train_metrics[0:(i+1)*batch_size].mean(axis=0)
                else:
                    train_scores_mean = train_scores.mean(axis=0)
                    train_metrics_mean = train_metrics.mean(axis=0)

                # Print to console
                print('Epoch: [{0}][{1}/{2}]\t'
                      .format(epoch, i+1, iter_per_epoch))
                utils_utils.print_metrics(train_metrics_mean)
                utils_utils.print_scores(train_scores_mean)

            if (i+1) == 1 or (i+1) % 1000 == 0 or (i+1) == iter_per_epoch:
                savePath = model_path + "/current" + ".csv"
                dataframe = pd.DataFrame(train_scores)
                dataframe.to_csv(savePath, header=False, index=False)

            if i in iter_list_save:
                model_name = model_path + "/model/epoch" + str(epoch+1).zfill(2) + '_iter' + str(i).zfill(5) + ".pth"
                # save model
                print('SAVE MODEL:' + model_path + '/model')
                torch.save(model.state_dict(), model_name)

            if i in iter_list_rebalancing:
                temp_train_scores_mean = train_scores[last_rebalancing_iter*batch_size:(i+1)*batch_size, :].mean(axis=0)
                total_loss = np.sum(temp_train_scores_mean * loss_weights)
                if weight_initialization == True and weight_initialization_done == False:
                    for index_loss in range(len(loss_valid)):
                        if loss_valid[index_loss] == 1:
                            loss_weights[index_loss] = (total_loss * loss_initialize_scale[index_loss]) / temp_train_scores_mean[index_loss]
                        else:
                            loss_weights[index_loss] = 0

                    # save previous record
                    weight_initialization_done = True
                    previous_total_loss = np.sum(temp_train_scores_mean * loss_weights)
                    previous_loss = temp_train_scores_mean

                elif weight_rebalancing == True and (weight_initialization_done == True or weight_initialization == False):
                    temp_train_scores_mean = train_scores[last_rebalancing_iter*batch_size:(i+1)*batch_size, :].mean(axis=0)
                    total_loss = np.sum(temp_train_scores_mean * loss_weights)
                    previous_loss_weights = np.array(loss_weights)
                    if previous_total_loss > 0:
                        for index_loss in range(len(loss_valid)):
                            if loss_valid[index_loss] == 1:
                                adjust_term = 1 + current_lambda_for_adjust * ((total_loss/previous_total_loss) * (previous_loss[index_loss]/temp_train_scores_mean[index_loss]) - 1)
                                adjust_term = min(max(adjust_term, 1.0/2.0), 2.0/1.0)
                                loss_weights[index_loss] = previous_loss_weights[index_loss] * adjust_term
                            else:
                                loss_weights[index_loss] = 0

                    # save previous record
                    previous_total_loss = np.sum(temp_train_scores_mean * loss_weights)
                    previous_loss = temp_train_scores_mean

                # save - loss weights
                savePath = model_path + "/weight/weight" + str(epoch + 1).zfill(2) + '_iter' + str(i).zfill(5) + ".csv"
                dataframe = pd.DataFrame(loss_weights)
                dataframe.to_csv(savePath, header=False, index=False)

                last_rebalancing_iter = (i+1) % iter_per_epoch

        # save - each image train score
        savePath = model_path + "/score/train_epoch" + str(0 + epoch + 1).zfill(2) + ".csv"
        dataframe = pd.DataFrame(train_scores)
        dataframe.to_csv(savePath, header=False, index=False)
        # save - train mean score
        savePath = model_path + "/score/train_mean_epoch" + str(0 + epoch + 1).zfill(2) + ".csv"
        dataframe = pd.DataFrame(train_scores_mean)
        dataframe.to_csv(savePath, header=False, index=False)
        # save - each image train score
        savePath = model_path + "/metric/train_epoch" + str(0 + epoch + 1).zfill(2) + ".csv"
        dataframe = pd.DataFrame(train_metrics)
        dataframe.to_csv(savePath, header=False, index=False)
        # save - train mean score
        savePath = model_path + "/metric/train_mean_epoch" + str(0 + epoch + 1).zfill(2) + ".csv"
        dataframe = pd.DataFrame(train_metrics_mean)
        dataframe.to_csv(savePath, header=False, index=False)

        print('-------------- TRAINING OF EPOCH ' + str(0+epoch+1).zfill(2) + 'FINISH ---------------')
        print('---------------------------------------------------------')
        print('   ')
        print('   ')
        print('   ')

if __name__ == '__main__':
    main()
