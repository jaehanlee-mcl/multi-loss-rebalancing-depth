from utils.loss import loss_for_metric8

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

    return rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3

def get_metric_1batch(batch_size, current_batch_size, index_iter, num_data, metrics,
                      rmse, rmse_log, abs_rel, sqr_rel, log10, delta1, delta2, delta3):
    for index_batch in range(current_batch_size):
        index_record = batch_size * index_iter + index_batch
        if index_record < num_data:
            metrics[index_record, :] = [
                rmse[index_batch],               rmse_log[index_batch],           abs_rel[index_batch],            sqr_rel[index_batch],            log10[index_batch],
                delta1[index_batch],             delta2[index_batch],             delta3[index_batch],
            ]
    return metrics