import time
import argparse

import torch
import torch.nn as nn
import os

import networks.model as network_model
import utils.utils as utils_utils
import utils.get_data as utils_get_data

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Arguments
    parser = argparse.ArgumentParser(description='Monocular Depth')
    parser.add_argument('--backbone', default='PNASNet5Large', type=str)
    parser.add_argument('--batch_size', default=4, type=int, help='batch size')
    parser.add_argument('--save_prediction', default=False, type=bool)
    parser.add_argument('--model_path', default='models', type=str)
    parser.add_argument('--model_name', default='PNAS_model.pth', type=str)
    parser.add_argument('--pred_path', default='prediction/PNAS_model', type=str)
    parser.add_argument('--test_dataset_path', default='dataset/test654.zip', type=str)
    parser.add_argument('--test_dataset_csv_list', default='test654/test.csv', type=str)
    args = parser.parse_args()

    # model path
    model_path = args.model_path
    model_name = args.model_name
    pred_path = args.pred_path

    # image size
    original_image_size = [480, 640]
    input_image_size = [288, 384]
    # interpolation function / relu
    interpolate_bicubic_fullsize = nn.Upsample(size=original_image_size, mode='bicubic')
    interpolate_bicubic_inputsize = nn.Upsample(size=input_image_size, mode='bicubic')
    relu = nn.ReLU()

    # Create model
    model = nn.DataParallel(network_model.create_model(args.backbone).half())
    print('Model created.')

    # loading training/testing data
    batch_size = args.batch_size
    test_loader, num_test_data = utils_get_data.getTestingData(batch_size, args.test_dataset_path, args.test_dataset_csv_list)

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

    for i, sample_batched in enumerate(test_loader):

        # Prepare sample and target
        image = torch.autograd.Variable(sample_batched['image'].cuda()).half()
        depth_gt = torch.autograd.Variable(sample_batched['depth'].cuda(non_blocking=True))
        current_batch_size = depth_gt.size(0)

        # Predict
        image_input = interpolate_bicubic_inputsize(image)
        depth_pred_for_loss = model(image_input)
        depth_pred_for_loss = (relu(depth_pred_for_loss - 0.0001) + 0.0001).float()
        depth_pred_full = interpolate_bicubic_fullsize(depth_pred_for_loss)

        # save prediction
        if args.save_prediction == True:
            for index_test in range(i * batch_size + 1, i * batch_size + current_batch_size + 1):
                utils_utils.pred2png(depth_pred_full[index_test - (i * batch_size + 1), 0, :, :].cpu().detach().numpy(), pred_path, index_test)

        # Measure elapsed time
        total_time = total_time + (time.time() - end)
        end = time.time()
        # Print to console
        if i == 0 or i % 20 == 19 or i == N - 1:
            print('Prediction - ', str(i+1).zfill(5), '/', str(N).zfill(5), '    Time: ', str('%10.2f' % total_time), 's')

    print('------------------------ FINISH -------------------------')
    print('---------------------------------------------------------')

if __name__ == '__main__':
    main()