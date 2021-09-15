from utils.utils import Logger
from utils.utils import save_checkpoint
from utils.utils import save_linear_checkpoint

from common.train import *
from evals import test_classifier

from torch.utils.data import DataLoader
from datasets import get_dataset, get_superclass_list, get_subclass_dataset
from evals.ood_pre import eval_ood_detection


if 'sup' in P.mode:
    from training.sup import setup
else:
    from training.unsup import setup
train, fname = setup(P.mode, P)

logger = Logger(fname, ask=False, local_rank=P.local_rank)
logger.log(P)
logger.log(model)

if P.multi_gpu:
    linear = model.module.linear
else:
    linear = model.linear
linear_optim = torch.optim.Adam(linear.parameters(), lr=1e-3, betas=(.9, .999), weight_decay=P.weight_decay)

# Run experiments
for epoch in range(start_epoch, P.epochs + 1):
    logger.log_dirname(f"Epoch {epoch}")
    model.train()

    if P.multi_gpu:
        train_sampler.set_epoch(epoch)

    kwargs = {}
    kwargs['linear'] = linear
    kwargs['linear_optim'] = linear_optim
    kwargs['simclr_aug'] = simclr_aug

    train(P, epoch, model, criterion, optimizer, scheduler, train_loader, logger=logger, **kwargs)

    model.eval()

    if epoch % P.save_step == 0 and P.local_rank == 0:
        if P.multi_gpu:
            save_states = model.module.state_dict()
        else:
            save_states = model.state_dict()
        save_checkpoint(epoch, save_states, optimizer.state_dict(), logger.logdir)
        save_linear_checkpoint(linear_optim.state_dict(), logger.logdir)

    if epoch % P.error_step == 0:
        if ('sup' in P.mode):
            error = test_classifier(P, model, test_loader, epoch, logger=logger)

            is_best = (best > error)
            if is_best:
                best = error

            logger.scalar_summary('eval/best_error', best, epoch)
            logger.log('[Epoch %3d] [Test %5.2f] [Best %5.2f]' % (epoch, error, best))

        else:
            ood_eval = P.mode == 'ood_pre'
            cls_list = get_superclass_list(P.dataset)
            ood_test_loader = dict()
            for ood in P.ood_dataset:
                if ood == 'interp':
                    ood_test_loader[ood] = None  # dummy loader
                    continue

                if P.one_class_idx is not None:
                    ood_test_set = get_subclass_dataset(full_test_set, classes=cls_list[ood])
                    ood = f'one_class_{ood}'  # change save name
                else:
                    ood_test_set = get_dataset(P, dataset=ood, test_only=True, image_size=P.image_size, eval=ood_eval)
                
                kwargs = {'pin_memory': False, 'num_workers': 0}
                ood_test_loader[ood] = DataLoader(ood_test_set, shuffle=False, batch_size=P.test_batch_size, **kwargs)

            with torch.no_grad():
                auroc_dict = eval_ood_detection(P, model, test_loader, ood_test_loader, P.ood_score,
                                                train_loader=train_loader, simclr_aug=simclr_aug)

            bests = []
            for ood in auroc_dict.keys():
                message = ''
                best_auroc = 0
                for ood_score, auroc in auroc_dict[ood].items():
                    message += '[%s %s %.4f] ' % (ood, ood_score, auroc)
                    if auroc > best_auroc:
                        best_auroc = auroc
                message += '[%s %s %.4f] ' % (ood, 'best', best_auroc)
                if P.print_score:
                    print(message)
                bests.append(best_auroc)

            # bests = map('{:.4f}'.format, bests)
            # print('\t'.join(bests))

            if logger is not None:
                logger.scalar_summary('train/bests', bests[0], epoch)

    
