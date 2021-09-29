import os
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import models.transform_layers as TL
from utils.utils import set_random_seed, normalize
from evals.evals import get_auroc, get_roc
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def eval_ood_detection(P, model_frontal, model_lateral, id_loader, ood_loaders, ood_scores, train_loader_frontal=None,
                       train_loader_lateral=None, simclr_aug=None):
    auroc_dict = {'frontal': {}, 'lateral': {}, 'combined': {}}
    roc_dict = {'frontal': {}, 'lateral': {}, 'combined': {}}
    for ood in ood_loaders.keys():
        auroc_dict['frontal'][ood] = dict()
        auroc_dict['lateral'][ood] = dict()
        auroc_dict['combined'][ood] = dict()
        roc_dict['frontal'][ood] = dict()
        roc_dict['lateral'][ood] = dict()
        roc_dict['combined'][ood] = dict()

    assert len(ood_scores) == 1  # assume single ood_score for simplicity
    ood_score = ood_scores[0]

    base_path = os.path.split(P.load_path_frontal)[0]  # checkpoint directory

    prefix = f'{P.ood_samples}'
    if P.resize_fix:
        prefix += f'_resize_fix_{P.resize_factor}'
    else:
        prefix += f'_resize_range_{P.resize_factor}'

    prefix = os.path.join(base_path, f'feats_{prefix}')

    kwargs = {
        'simclr_aug': simclr_aug,
        'sample_num': P.ood_samples,
        'layers': P.ood_layer,
    }

    print('Pre-compute global statistics...')
    feats_train_frontal = get_features(P, f'{P.dataset}_train_frontal', model_frontal, train_loader_frontal,
                                       prefix=prefix, **kwargs)  # (M, T, d)
    feats_train_lateral = get_features(P, f'{P.dataset}_train_lateral', model_lateral, train_loader_lateral,
                                       prefix=prefix, **kwargs)  # (M, T, d)

    P.axis = {'frontal': [], 'lateral': []}

    for i in range(P.K_shift):
        # frontal
        f_frontal = feats_train_frontal['simclr'].chunk(P.K_shift, dim=1)[i]
        axis = f_frontal.mean(dim=1)  # (M, d)
        P.axis['frontal'].append(normalize(axis, dim=1).to(device))

        # lateral
        f_lateral = feats_train_lateral['simclr'].chunk(P.K_shift, dim=1)[i]
        axis = f_lateral.mean(dim=1)  # (M, d)
        P.axis['lateral'].append(normalize(axis, dim=1).to(device))

    print('Frontal axis size: ' + ' '.join(map(lambda x: str(len(x)), P.axis['frontal'])))
    print('Lateral axis size: ' + ' '.join(map(lambda x: str(len(x)), P.axis['lateral'])))

    f_sim_frontal = [f.mean(dim=1) for f in feats_train_frontal['simclr'].chunk(P.K_shift, dim=1)]  # list of (M, d)
    f_shi_frontal = [f.mean(dim=1) for f in feats_train_frontal['shift'].chunk(P.K_shift, dim=1)]  # list of (M, 4)

    f_sim_lateral = [f.mean(dim=1) for f in feats_train_lateral['simclr'].chunk(P.K_shift, dim=1)]  # list of (M, d)
    f_shi_lateral = [f.mean(dim=1) for f in feats_train_lateral['shift'].chunk(P.K_shift, dim=1)]  # list of (M, 4)

    weight_sim_frontal = []
    weight_shi_frontal = []
    weight_sim_lateral = []
    weight_shi_lateral = []

    for shi in range(P.K_shift):
        sim_norm = f_sim_frontal[shi].norm(dim=1)  # (M)
        shi_mean = f_shi_frontal[shi][:, shi]  # (M)
        weight_sim_frontal.append(1 / sim_norm.mean().item())
        weight_shi_frontal.append(1 / shi_mean.mean().item())

        sim_norm = f_sim_lateral[shi].norm(dim=1)  # (M)
        shi_mean = f_shi_lateral[shi][:, shi]  # (M)
        weight_sim_lateral.append(1 / sim_norm.mean().item())
        weight_shi_lateral.append(1 / shi_mean.mean().item())

    if ood_score == 'simclr':
        P.weight_sim = [1]
        P.weight_shi = [0]
    elif ood_score == 'CSI':
        P.weight_sim = {'frontal': weight_sim_frontal, 'lateral': weight_sim_lateral}
        P.weight_shi = {'frontal': weight_shi_frontal, 'lateral': weight_shi_lateral}
    else:
        raise ValueError()

    print(f'Frontal weight_sim:\t' + '\t'.join(map('{:.4f}'.format, P.weight_sim['frontal'])))
    print(f'Frontal weight_shi:\t' + '\t'.join(map('{:.4f}'.format, P.weight_shi['frontal'])))
    print(f'Lateral weight_sim:\t' + '\t'.join(map('{:.4f}'.format, P.weight_sim['lateral'])))
    print(f'Lateral weight_shi:\t' + '\t'.join(map('{:.4f}'.format, P.weight_shi['lateral'])))

    print('Pre-compute features...')
    feats_id_lateral = {'simclr': torch.Tensor(), 'shift': torch.Tensor()}
    feats_id_frontal = {'simclr': torch.Tensor(), 'shift': torch.Tensor()}

    id_df = id_loader.dataset.dataset.df.iloc[id_loader.dataset.indices].copy()
    first_idx = id_df.index.min()
    patients_list = id_df['patient'].unique().tolist()
    for patient in patients_list:
        # frontal
        frontal = id_df[(id_df['patient'] == patient) & (id_df['Frontal/Lateral'] == 'Frontal')]
        dataset = Subset(id_loader.dataset, [frontal.index[0] - first_idx])
        id_sub_loader = DataLoader(dataset, shuffle=True, batch_size=P.batch_size, pin_memory=False, num_workers=0)
        current_feats_id_frontal = get_features(P, P.dataset, model_frontal, id_sub_loader, prefix=prefix, **kwargs)  # (N, T, d)
        feats_id_frontal['simclr'] = torch.cat((feats_id_frontal['simclr'], current_feats_id_frontal['simclr']))
        feats_id_frontal['shift'] = torch.cat((feats_id_frontal['shift'], current_feats_id_frontal['shift']))

        # lateral
        lateral = id_df[(id_df['patient'] == patient) & (id_df['Frontal/Lateral'] == 'Lateral')]
        dataset = Subset(id_loader.dataset, [lateral.index[0] - first_idx])
        id_sub_loader = DataLoader(dataset, shuffle=True, batch_size=P.batch_size, pin_memory=False, num_workers=0)
        current_feats_id_lateral = get_features(P, P.dataset, model_lateral, id_sub_loader, prefix=prefix, **kwargs)  # (N, T, d)
        feats_id_lateral['simclr'] = torch.cat((feats_id_lateral['simclr'], current_feats_id_lateral['simclr']))
        feats_id_lateral['shift'] = torch.cat((feats_id_lateral['shift'], current_feats_id_lateral['shift']))

    feats_ood_frontal = dict((el, {'simclr': torch.Tensor(), 'shift': torch.Tensor()}) for el in ood_loaders.keys())
    feats_ood_lateral = dict((el, {'simclr': torch.Tensor(), 'shift': torch.Tensor()}) for el in ood_loaders.keys())

    for ood, ood_loader in ood_loaders.items():
        if ood == 'interp':
            feats_ood[ood] = get_features(P, ood, model, id_loader, interp=True, prefix=prefix, **kwargs)
        else:
            ood_df = ood_loader.dataset.dataset.df.iloc[ood_loader.dataset.indices].copy()
            first_idx = ood_df.index.min()
            patients_list = ood_df['patient'].unique().tolist()
            for patient in patients_list:
                # frontal
                frontal = ood_df[(ood_df['patient'] == patient) & (ood_df['Frontal/Lateral'] == 'Frontal')]
                dataset = Subset(ood_loader.dataset, [frontal.index[0] - first_idx])
                ood_sub_loader = DataLoader(dataset, shuffle=True, batch_size=P.batch_size, pin_memory=False,
                                            num_workers=0)
                current_feats_ood_frontal = get_features(P, ood, model_frontal, ood_sub_loader, prefix=prefix, **kwargs)
                feats_ood_frontal[ood]['simclr'] = torch.cat((feats_ood_frontal[ood]['simclr'], current_feats_ood_frontal['simclr']))
                feats_ood_frontal[ood]['shift'] = torch.cat((feats_ood_frontal[ood]['shift'], current_feats_ood_frontal['shift']))

                # lateral
                lateral = ood_df[(ood_df['patient'] == patient) & (ood_df['Frontal/Lateral'] == 'Lateral')]
                dataset = Subset(ood_loader.dataset, [lateral.index[0] - first_idx])
                ood_sub_loader = DataLoader(dataset, shuffle=True, batch_size=P.batch_size, pin_memory=False,
                                            num_workers=0)
                current_feats_ood_lateral = get_features(P, ood, model_lateral, ood_sub_loader, prefix=prefix, **kwargs)
                feats_ood_lateral[ood]['simclr'] = torch.cat((feats_ood_lateral[ood]['simclr'], current_feats_ood_lateral['simclr']))
                feats_ood_lateral[ood]['shift'] = torch.cat((feats_ood_lateral[ood]['shift'], current_feats_ood_lateral['shift']))

    print(f'Compute OOD scores... (score: {ood_score})')
    scores_id_frontal = get_scores(P, feats_id_frontal, ood_score, type='frontal').numpy()
    scores_id_lateral = get_scores(P, feats_id_lateral, ood_score, type='lateral').numpy()

    scores_ood = {'frontal': {}, 'lateral': {}, 'combined': {}}
    # if P.one_class_idx is not None:
        # one_class_score_frontal = []
        # one_class_score_lateral = []
        # one_class_score_combined = []

    for ood in ood_loaders.keys():
        feats = feats_ood_frontal[ood]
        scores_ood['frontal'][ood] = get_scores(P, feats, ood_score, type='frontal').numpy()

        feats = feats_ood_lateral[ood]
        scores_ood['lateral'][ood] = get_scores(P, feats, ood_score, type='lateral').numpy()

        auroc_dict['frontal'][ood][ood_score] = get_auroc(scores_id_frontal, scores_ood['frontal'][ood])
        auroc_dict['lateral'][ood][ood_score] = get_auroc(scores_id_lateral, scores_ood['lateral'][ood])
        roc_dict['frontal'][ood][ood_score] = get_roc(scores_id_frontal, scores_ood['frontal'][ood])
        roc_dict['lateral'][ood][ood_score] = get_roc(scores_id_lateral, scores_ood['lateral'][ood])

        combind_scores_id = np.vstack((scores_id_frontal, scores_id_lateral)).mean(axis=0)
        scores_ood['combined'][ood] = np.vstack((scores_ood['frontal'][ood], scores_ood['lateral'][ood])).mean(axis=0)
        auroc_dict['combined'][ood][ood_score] = get_auroc(combind_scores_id, scores_ood['combined'][ood])
        roc_dict['combined'][ood][ood_score] = get_roc(combind_scores_id, scores_ood['combined'][ood])

        # if P.one_class_idx is not None:
        #     one_class_score_frontal.append(scores_ood['frontal'][ood])
        #     one_class_score_lateral.append(scores_ood['lateral'][ood])
        #     one_class_score_combined.append(scores_ood['combined'][ood])

    # if P.one_class_idx is not None:
        # one_class_score = np.concatenate(one_class_score)
        # one_class_total = get_auroc(scores_id, one_class_score)
        # print(f'One_class_real_mean: {one_class_total}')

    if P.print_score:
        print_score(P.dataset, scores_id_frontal)
        print_score(P.dataset, scores_id_lateral)
        print_score(P.dataset, combind_scores_id)

        for ood in ood_loaders.keys():
            scores = scores_ood['frontal'][ood]
            print_score(ood, scores)

            scores = scores_ood['lateral'][ood]
            print_score(ood, scores)

            scores = scores_ood['combined'][ood]
            print_score(ood, scores)

    return auroc_dict, roc_dict


def get_scores(P, feats_dict, ood_score, type):
    # convert to gpu tensor
    feats_sim = feats_dict['simclr'].to(device)
    feats_shi = feats_dict['shift'].to(device)
    N = feats_sim.size(0)

    # compute scores
    scores = []
    for f_sim, f_shi in zip(feats_sim, feats_shi):
        f_sim = [f.mean(dim=0, keepdim=True) for f in f_sim.chunk(P.K_shift)]  # list of (1, d)
        f_shi = [f.mean(dim=0, keepdim=True) for f in f_shi.chunk(P.K_shift)]  # list of (1, 4)
        score = 0
        for shi in range(P.K_shift):
            score += (f_sim[shi] * P.axis[type][shi]).sum(dim=1).max().item() * P.weight_sim[type][shi]
            score += f_shi[shi][:, shi].item() * P.weight_shi[type][shi]
        score = score / P.K_shift
        scores.append(score)
    scores = torch.tensor(scores)

    assert scores.dim() == 1 and scores.size(0) == N  # (N)
    return scores.cpu()


def get_features(P, data_name, model, loader, interp=False, prefix='',
                 simclr_aug=None, sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # load pre-computed features if exists
    feats_dict = dict()
    # for layer in layers:
    #     path = prefix + f'_{data_name}_{layer}.pth'
    #     if os.path.exists(path):
    #         feats_dict[layer] = torch.load(path)

    # pre-compute features and save to the path
    left = [layer for layer in layers if layer not in feats_dict.keys()]
    if len(left) > 0:
        _feats_dict = _get_features(P, model, loader, interp, P.dataset == 'imagenet',
                                    simclr_aug, sample_num, layers=left)

        for layer, feats in _feats_dict.items():
            path = prefix + f'_{data_name}_{layer}.pth'
            torch.save(_feats_dict[layer], path)
            feats_dict[layer] = feats  # update value

    return feats_dict


def _get_features(P, model, loader, interp=False, imagenet=False, simclr_aug=None,
                  sample_num=1, layers=('simclr', 'shift')):

    if not isinstance(layers, (list, tuple)):
        layers = [layers]

    # check if arguments are valid
    assert simclr_aug is not None

    if imagenet is True:  # assume batch_size = 1 for ImageNet
        sample_num = 1

    # compute features in full dataset
    model.eval()
    feats_all = {layer: [] for layer in layers}  # initialize: empty list
    for i, (x, _) in enumerate(loader):
        if interp:
            x_interp = (x + last) / 2 if i > 0 else x  # omit the first batch, assume batch sizes are equal
            last = x  # save the last batch
            x = x_interp  # use interp as current batch

        if imagenet is True:
            x = torch.cat(x[0], dim=0)  # augmented list of x

        x = x.to(device)  # gpu tensor

        # compute features in one batch
        feats_batch = {layer: [] for layer in layers}  # initialize: empty list
        for seed in range(sample_num):
            set_random_seed(seed)

            if P.K_shift > 1:
                x_t = torch.cat([P.shift_trans(hflip(x), k) for k in range(P.K_shift)])
            else:
                x_t = x # No shifting: SimCLR
            x_t = simclr_aug(x_t)

            # compute augmented features
            with torch.no_grad():
                kwargs = {layer: True for layer in layers}  # only forward selected layers
                _, output_aux = model(x_t, **kwargs)

            # add features in one batch
            for layer in layers:
                feats = output_aux[layer].cpu()
                if imagenet is False:
                    feats_batch[layer] += feats.chunk(P.K_shift)
                else:
                    feats_batch[layer] += [feats]  # (B, d) cpu tensor

        # concatenate features in one batch
        for key, val in feats_batch.items():
            if imagenet:
                feats_batch[key] = torch.stack(val, dim=0)  # (B, T, d)
            else:
                feats_batch[key] = torch.stack(val, dim=1)  # (B, T, d)

        # add features in full dataset
        for layer in layers:
            feats_all[layer] += [feats_batch[layer]]

    # concatenate features in full dataset
    for key, val in feats_all.items():
        feats_all[key] = torch.cat(val, dim=0)  # (N, T, d)

    # reshape order
    if imagenet is False:
        # Convert [1,2,3,4, 1,2,3,4] -> [1,1, 2,2, 3,3, 4,4]
        for key, val in feats_all.items():
            N, T, d = val.size()  # T = K * T'
            val = val.view(N, -1, P.K_shift, d)  # (N, T', K, d)
            val = val.transpose(2, 1)  # (N, 4, T', d)
            val = val.reshape(N, T, d)  # (N, T, d)
            feats_all[key] = val

    return feats_all


def print_score(data_name, scores):
    quantile = np.quantile(scores, np.arange(0, 1.1, 0.1))
    print('{:18s} '.format(data_name) +
          '{:.4f} +- {:.4f}    '.format(np.mean(scores), np.std(scores)) +
          '    '.join(['q{:d}: {:.4f}'.format(i * 10, quantile[i]) for i in range(11)]))

