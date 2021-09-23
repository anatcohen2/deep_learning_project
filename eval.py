from common.eval import *

model_frontal.eval()
model_lateral.eval()

if P.mode == 'test_acc':
    from evals import test_classifier
    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, logger=None)

elif P.mode == 'test_marginalized_acc':
    from evals import test_classifier
    with torch.no_grad():
        error = test_classifier(P, model, test_loader, 0, marginal=True, logger=None)

elif P.mode in ['ood', 'ood_pre']:
    if P.mode == 'ood':
        from evals import eval_ood_detection
    else:
        from evals.ood_pre import eval_ood_detection

    with torch.no_grad():
        auroc_dict = eval_ood_detection(P, model_frontal=model_frontal, model_lateral=model_lateral,
                                        id_loader=test_loader, ood_loaders=ood_test_loader, ood_scores=P.ood_score,
                                        train_loader_frontal=train_loader_frontal,
                                        train_loader_lateral=train_loader_lateral, simclr_aug=simclr_aug)

    if P.one_class_idx is not None:
        mean_dict = {'frontal': dict(), 'lateral': dict(), 'combined': dict()}
        for ood_score in P.ood_score:
            mean = 0
            for type in auroc_dict.keys():
                for ood in auroc_dict[type].keys():
                    mean += auroc_dict[type][ood][ood_score]
                mean_dict[type][ood_score] = mean / len(auroc_dict[type].keys())
            # auroc_dict['one_class_mean'] = mean_dict

    bests = []
    for type in auroc_dict.keys():
        for ood in auroc_dict[type].keys():
            message = f'[{type}] '
            best_auroc = 0
            for ood_score, auroc in auroc_dict[type][ood].items():
                message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
                if auroc > best_auroc:
                    best_auroc = auroc
            message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
            if P.print_score:
                print(message)
            bests.append(best_auroc)

    bests = map('{:.4f}'.format, bests)
    print('\t'.join(bests))

else:
    raise NotImplementedError()


