from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections import namedtuple

import torch
import torch.nn as nn
import datetime
import math
import numpy as np
from PIL import Image

def pred2png(data, data_path, index_data):
    data = (data / 10 * (pow(2,16)-1)).astype(np.uint16)
    data_name = data_path + '/pred' + str(index_data).zfill(4) + '.png'
    data_image = Image.fromarray(data)
    data_image.save(data_name)

def get_model_summary(model, *input_tensors, item_length=26, verbose=False):
    """
    :param model:
    :param input_tensors:
    :param item_length:
    :return:
    """

    summary = []

    ModuleDetails = namedtuple(
        "Layer", ["name", "input_size", "output_size", "num_parameters", "multiply_adds"])
    hooks = []
    layer_instances = {}

    def add_hooks(module):

        def hook(module, input, output):
            class_name = str(module.__class__.__name__)

            instance_index = 1
            if class_name not in layer_instances:
                layer_instances[class_name] = instance_index
            else:
                instance_index = layer_instances[class_name] + 1
                layer_instances[class_name] = instance_index

            layer_name = class_name + "_" + str(instance_index)

            params = 0

            if class_name.find("Conv") != -1 or class_name.find("BatchNorm") != -1 or \
               class_name.find("Linear") != -1:
                for param_ in module.parameters():
                    params += param_.view(-1).size(0)

            flops = "Not Available"
            if class_name.find("Conv") != -1 and hasattr(module, "weight"):
                flops = (
                    torch.prod(
                        torch.LongTensor(list(module.weight.data.size()))) *
                    torch.prod(
                        torch.LongTensor(list(output.size())[2:]))).item()
            elif isinstance(module, nn.Linear):
                flops = (torch.prod(torch.LongTensor(list(output.size()))) \
                         * input[0].size(1)).item()

            if isinstance(input[0], list):
                input = input[0]
            if isinstance(output, list):
                output = output[0]

            summary.append(
                ModuleDetails(
                    name=layer_name,
                    input_size=list(input[0].size()),
                    output_size=list(output.size()),
                    num_parameters=params,
                    multiply_adds=flops)
            )

        if not isinstance(module, nn.ModuleList) \
           and not isinstance(module, nn.Sequential) \
           and module != model:
            hooks.append(module.register_forward_hook(hook))

    model.eval()
    model.apply(add_hooks)

    space_len = item_length

    model(*input_tensors)
    for hook in hooks:
        hook.remove()

    details = ''
    if verbose:
        details = "Model Summary" + \
            os.linesep + \
            "Name{}Input Size{}Output Size{}Parameters{}Multiply Adds (Flops){}".format(
                ' ' * (space_len - len("Name")),
                ' ' * (space_len - len("Input Size")),
                ' ' * (space_len - len("Output Size")),
                ' ' * (space_len - len("Parameters")),
                ' ' * (space_len - len("Multiply Adds (Flops)"))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    params_sum = 0
    flops_sum = 0
    for layer in summary:
        params_sum += layer.num_parameters
        if layer.multiply_adds != "Not Available":
            flops_sum += layer.multiply_adds
        if verbose:
            details += "{}{}{}{}{}{}{}{}{}{}".format(
                layer.name,
                ' ' * (space_len - len(layer.name)),
                layer.input_size,
                ' ' * (space_len - len(str(layer.input_size))),
                layer.output_size,
                ' ' * (space_len - len(str(layer.output_size))),
                layer.num_parameters,
                ' ' * (space_len - len(str(layer.num_parameters))),
                layer.multiply_adds,
                ' ' * (space_len - len(str(layer.multiply_adds)))) \
                + os.linesep + '-' * space_len * 5 + os.linesep

    details += os.linesep \
        + "Total Parameters: {:,}".format(params_sum) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Total Multiply Adds (For Convolution and Linear Layers only): {:,} GFLOPs".format(flops_sum/(1024**3)) \
        + os.linesep + '-' * space_len * 5 + os.linesep
    details += "Number of Layers" + os.linesep
    for layer in layer_instances:
        details += "{} : {} layers   ".format(layer, layer_instances[layer])

    return details

def print_scores(scores):
    ############
    ## line 0 ##
    ############
    print(
        "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
            'l_d0_depth', 'l_d0_depth_dx', 'l_d0_depth_dy', 'l_d0_depth_norm', 'l_d0_depth_dx2',
            'l_d0_depth_dxy', 'l_d0_depth_dy2', 'l_d0_depth_dx_norm', 'l_d0_depth_dy_norm', 'l_d0_ndepth',
            'l_d0_ndepth_w5', 'l_d0_ndepth_w17', 'l_d0_ndepth_w65'))
    print(
        "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
            scores[0],  scores[1],  scores[2],  scores[3],  scores[4],
            scores[5],  scores[6],  scores[7],  scores[8],  scores[9],
            scores[10], scores[11], scores[12]
        ))
    ############
    ## line 1 ##
    ############
    print(
        "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
            'l_d1_depth', 'l_d1_depth_dx', 'l_d1_depth_dy', 'l_d1_depth_norm', 'l_d1_depth_dx2',
            'l_d1_depth_dxy', 'l_d1_depth_dy2', 'l_d1_depth_dx_norm', 'l_d1_depth_dy_norm', 'l_d1_ndepth',
            'l_d1_ndepth_w5', 'l_d1_ndepth_w17', 'l_d1_ndepth_w65'))
    print(
        "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
            scores[13], scores[14], scores[15], scores[16], scores[17],
            scores[18], scores[19], scores[20], scores[21], scores[22],
            scores[23], scores[24], scores[25]
        ))
    ############
    ## line 2 ##
    ############
    print(
        "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
            'l_d2_depth', 'l_d2_depth_dx', 'l_d2_depth_dy', 'l_d2_depth_norm', 'l_d2_depth_dx2',
            'l_d2_depth_dxy', 'l_d2_depth_dy2', 'l_d2_depth_dx_norm', 'l_d2_depth_dy_norm', 'l_d2_ndepth',
            'l_d2_ndepth_w5', 'l_d2_ndepth_w17', 'l_d2_ndepth_w65'))
    print(
        "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
            scores[26], scores[27], scores[28], scores[29], scores[30],
            scores[31], scores[32], scores[33], scores[34], scores[35],
            scores[36], scores[37], scores[38]
        ))
    ############
    ## line 3 ##
    ############
    print(
        "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
            'l_d3_depth', 'l_d3_depth_dx', 'l_d3_depth_dy', 'l_d3_depth_norm', 'l_d3_depth_dx2',
            'l_d3_depth_dxy', 'l_d3_depth_dy2', 'l_d3_depth_dx_norm', 'l_d3_depth_dy_norm', 'l_d3_ndepth',
            'l_d3_ndepth_w5', 'l_d3_ndepth_w17', 'l_d3_ndepth_w65'))
    print(
        "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
            scores[39], scores[40], scores[41], scores[42], scores[43],
            scores[44], scores[45], scores[46], scores[47], scores[48],
            scores[49], scores[50], scores[51]
        ))
    ############
    ## line 4 ##
    ############
    print(
        "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
            'l_d4_depth', 'l_d4_depth_dx', 'l_d4_depth_dy', 'l_d4_depth_norm', 'l_d4_depth_dx2',
            'l_d4_depth_dxy', 'l_d4_depth_dy2', 'l_d4_depth_dx_norm', 'l_d4_depth_dy_norm', 'l_d4_ndepth',
            'l_d4_ndepth_w5', 'l_d4_ndepth_w17', 'l_d4_ndepth_w65'))
    print(
        "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
            scores[52], scores[53], scores[54], scores[55], scores[56],
            scores[57], scores[58], scores[59], scores[60], scores[61],
            scores[62], scores[63], scores[64]
        ))
    ############
    ## line 5 ##
    ############
    print(
        "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
            'l_d5_depth', 'l_d5_depth_dx', 'l_d5_depth_dy', 'l_d5_depth_norm', 'l_d5_depth_dx2',
            'l_d5_depth_dxy', 'l_d5_depth_dy2', 'l_d5_depth_dx_norm', 'l_d5_depth_dy_norm', 'l_d5_ndepth',
            'l_d5_ndepth_w5', 'l_d5_ndepth_w17', 'l_d5_ndepth_w65'))
    print(
        "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
            scores[65], scores[66], scores[67], scores[68], scores[69],
            scores[70], scores[71], scores[72], scores[73], scores[74],
            scores[75], scores[76], scores[77]
        ))

def print_metrics(metrics):
    print(
        "{:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}, {:>19}".format(
            'rmse', 'rmse_log', 'abs_rel', 'sqr_rel', 'log10', 'delta1', 'delta2', 'delta3'))
    print(
        "{:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}, {:19.7f}".format(
            metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5], metrics[6], metrics[7]
        ))

def make_model_path(model_name, decoder_scale, batch_size):
    now = datetime.datetime.now()
    model_path_backbone = ''
    if model_name == 'DenseNet161':
        model_path_backbone = 'D161_b' + str(batch_size).zfill(2) + '_scale' + str(decoder_scale).zfill(4)
    if model_name == 'PNASNet5Large':
        model_path_backbone = 'P5L_b' + str(batch_size).zfill(2) + '_scale' + str(decoder_scale).zfill(4)

    model_path_data = 'data-' + str(now.year).zfill(4) + str(now.month).zfill(2) + str(now.day).zfill(2) + str(
        now.hour).zfill(2) + str(now.minute).zfill(2)
    model_path = 'runs/' + model_path_backbone + '/' + model_path_data
    if not os.path.isdir('runs'):
        os.mkdir('runs')
    if not os.path.isdir('runs/' + model_path_backbone):
        os.mkdir('runs/' + model_path_backbone)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
    if not os.path.isdir(model_path + '/weight'):
        os.mkdir(model_path + '/weight')
    if not os.path.isdir(model_path + '/score'):
        os.mkdir(model_path + '/score')
    if not os.path.isdir(model_path + '/metric'):
        os.mkdir(model_path + '/metric')
    if not os.path.isdir(model_path + '/model'):
        os.mkdir(model_path + '/model')

    return model_path

def get_notable_iter(iter_per_epoch, num_per_epoch=4):
    iter_list = np.zeros((num_per_epoch, 1))
    for i in range(num_per_epoch):
        iter_list[i] = math.ceil((i+1) * iter_per_epoch/num_per_epoch) - 1
    return iter_list