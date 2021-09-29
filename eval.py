from common.eval import *
from matplotlib import pyplot as plt

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
        auroc_dict, roc_dict = eval_ood_detection(P, model_frontal=model_frontal, model_lateral=model_lateral,
                                        id_loader=test_loader, ood_loaders=ood_test_loader, ood_scores=P.ood_score,
                                        train_loader_frontal=train_loader_frontal,
                                        train_loader_lateral=train_loader_lateral, simclr_aug=simclr_aug)

    if P.one_class_idx is not None:
        mean_dict = {'frontal': dict(), 'lateral': dict(), 'combined': dict()}
        for ood_score in P.ood_score:
            for type in auroc_dict.keys():
                mean = 0
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


            # for ood_score, roc in roc_dict[type][ood].items():
            #     plt.figure()
            #     fpr = roc_dict[type][ood][ood_score][0]
            #     tpr = roc_dict[type][ood][ood_score][1]
            #     plt.plot(fpr, tpr, color='darkorange', lw=2)
            #     plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            #     plt.xlim([0.0, 1.0])
            #     plt.ylim([0.0, 1.05])
            #     plt.xlabel('False Positive Rate')
            #     plt.ylabel('True Positive Rate')
            #     plt.title('ROC')
            #     plt.legend(loc="lower right")
            #     plt.savefig(f'roc_curve_{type}_{ood}_{ood_score}.png')
            #     print(f'saved roc_curve_{type}_{ood}_{ood_score}.png')

            message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
            if P.print_score:
                print(message)
            bests.append(best_auroc)

    bests = map('{:.4f}'.format, bests)
    print('\t'.join(bests))

    plt.figure()
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    fpr = roc_dict['lateral']['one_class_1']['CSI'][0]
    tpr = roc_dict['lateral']['one_class_1']['CSI'][1]
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='lateral')
    fpr = roc_dict['frontal']['one_class_1']['CSI'][0]
    tpr = roc_dict['frontal']['one_class_1']['CSI'][1]
    plt.plot(fpr, tpr, color='darkgreen', lw=2, label='frontal')
    fpr = roc_dict['combined']['one_class_1']['CSI'][0]
    tpr = roc_dict['combined']['one_class_1']['CSI'][1]
    plt.plot(fpr, tpr, color='brown', lw=2, label='combined')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig(f'roc_curve.png')
    print(f'saved roc_curve.png')

else:
    raise NotImplementedError()


