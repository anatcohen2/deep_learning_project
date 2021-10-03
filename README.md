# deep_learning_project
# ANOMALY DETECTION ON CHEST X-RAY IMAGES

## 1. Requirements
### Environments
Currently, requires following packages
- python 3.6+
- torch 1.8+
- torchvision 0.9+
- CUDA 10.1+
- scikit-learn 0.22+
- tensorboard 2.0+
- torchlars == 0.1.2
- diffdist == 0.1

### Dataset
please download the following dataset to ~/data:

Chexpert Dataset: A Large Chest X-Ray Dataset And Competition
https://stanfordmlgroup.github.io/competitions/chexpert

## 2. Training
 
To train the model in the paper, run this command:

```train
python -m train.py --dataset chexpert --model resnet18 --mode simclr_CSI --shift_trans_type rotation --one_class_idx 0
```

> --one_class_idx denotes the in-distribution of one-class training.

## 3. Evaluation
We provide the checkpoints of the pre-trained model. It is available in the checkpoints directory.

To evaluate the model, run this command:

```eval
python eval.py --mode ood_pre --dataset chexpert --model resnet18 --ood_score CSI --shift_trans_type rotation --print_score --ood_samples 10 --resize_factor 0.54 --resize_fix --one_class_idx 0 --load_path_frontal <FRONTAL_PATH> --load_path_lateral <LATERAL_PATH>
```

> --one_class_idx denotes the in-distribution of one-class evaluation.
>
> The resize_factor & resize fix option fix the cropping size of RandomResizedCrop().

## 4. Results

Our model achieves the following performance on Chexpert dataset (AUROC):

### One-Class Out-of-Distribution Detection

|                               | Atelectasis  |  Cardiomegaly | Consolidation  |  Edema | Pleural Effusion  |  All observations |
| ------------------------------|------------- | --------------|--------------- | -------|------------------ | ------------------|
| Frontal view                  | 0.6876       |      0.6530   | 0.7090         | 0.6669 |    0.7418         |      0.6423       |
| Lateral view                  | 0.6973       |      0.6531   | 0.7368         | 0.7220 |    0.7929         |      0.6648       |
| Combined (Frontal+Lateral)    | 0.7313       |      0.6814   | 0.7685         | 0.7465 |    0.8242         |      0.6830       |


<p align="center">
    <img src=figures/roc_curves.png width="600"> 
</p>
