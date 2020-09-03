# [ECCV 2020] Multi-Loss Rebalancing Algorithm for Monocular Depth Estimation
![lee2020multi_loss_rebalancing_depth](img/intro.png)

------
## Paper

If you use our code or results, please cite:

```
@InProceedings{Lee_2020_ECCV,
  author = {Lee, Jae-Han and Kim, Chang-Su},
  title = {Multi-Loss Rebalancing Algorithm for Monocular Depth Estimation}, 
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2020}
}
```

-------
## Large files
Large files can be downloaded using the following link.

### 1. dataset
Training and evaluation data are obtained by reprocessing what is provided on the [official site](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). You can easily use the reprocessed data with the link below.

All depth maps are saved as 16bit png files.

Depths are mapped by the following formula: depth = double(value) / (2^16 - 1) * 10

- 654 RGBD pairs for evaluation: [test654.zip](https://drive.google.com/file/d/1scqBb4kCB82ssDoO8UfvrWUubYH_hjXs/view?usp=sharing)

- 795 RGBD pairs for training: [train795.zip](https://drive.google.com/file/d/1VNRsXzc0MMjjXLdJpcwBTh1eosif7orU/view?usp=sharing)

- RGBD pairs obtained from the training sequence: [train_reduced05.zip](https://drive.google.com/file/d/1s6-4mm-wDwo0bwEG1LKLsadjB0K5EosP/view?usp=sharing)

### 2. models
[PNAS_model.pth](https://drive.google.com/file/d/1B1LdpOqIiyLN5JtzlDo-9nItiFIyfJeV/view?usp=sharing)

