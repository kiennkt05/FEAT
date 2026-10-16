import argparse
import os
try:
    import wandb
except ImportError:
    wandb = None
from utils.dmc import DataManager, setup_seed
from utils.toolkit import count_parameters

from methods.FEAT import FEAT
from methods.FEAT_repo import FEAT as RepoFEAT
from methods.FEAT_exp5a_global import FEATExp5aGlobal
from methods.FEAT_repo_exp5a_global import RepoFEATExp5aGlobal

import warnings
import datetime
import logging

warnings.filterwarnings('ignore')


def get_learner(model_name, args):   
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=args['method'] + '_' + args.get('implementation', 'paper') + '_training_' + args['dataset'] + '_tasks_' + str(args['tasks']) + '_beta_' + str(
            args['beta']) + '_users_' + str(args['num_users']) + '_frac_' + str(args['frac']) + '_mem_size_' + str(args['mem_size']) + '_k_value_' + str(args['k_value']) + '_p_value_' + str(args['p_value']) + '_seed_' + str(args['seed']) + '_local_epoch_' + str(
            args['local_ep']) + '_experiment_log-%s' % (datetime.datetime.now().strftime("%Y-%m-%d-%H%M-%S")) + '.log',
        filemode='a'
    )
    logger = logging.getLogger(__name__)

    name = model_name.lower()
    if name == "feat":
        implementation = args.get("implementation", "paper")
        if implementation == "paper":
            return FEAT(args), logger
        if implementation == "repo":
            return RepoFEAT(args), logger
        raise ValueError(f"Unknown implementation: {implementation}")
    if name == "exp5a_global":
        implementation = args.get("implementation", "paper")
        if implementation == "paper":
            return FEATExp5aGlobal(args), logger
        if implementation == "repo":
            return RepoFEATExp5aGlobal(args), logger
        raise ValueError(f"Unknown implementation: {implementation}")
    else:
        raise ValueError(f"Unknown method: {model_name}. Available: feat, exp5a_global")


def train(args):
    setup_seed(args["seed"])
    # setup the dataset and labels
    data_manager = DataManager(
        args["dataset"],
        args["class_shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args
    )
    args["class_order"] = data_manager.get_class_order()
    print("xxxxxxxx", args["class_order"])
    # ipdb.set_trace()
    learner, logger = get_learner(args["method"], args)
    logger.info(str(args))
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    # ipdb.set_trace()
    # train for each task
    for task in range(data_manager.nb_tasks):
        # for task in range(2):
        print("All params: {}, Trainable params: {}".format(count_parameters(learner._network),
                                                            count_parameters(learner._network, True)))
        learner.incremental_train(data_manager, logger)  # train for one task
        if args["method"] != "hr":
            cnn_accy, nme_accy = learner.eval_task(args, task * data_manager.get_task_size(task), learner.PH, learner.PT, learner.bar_eH_G, learner.r_H, learner.r_T)
            learner.after_task()
            logger.info(str(cnn_accy["grouped"]))
            print("CNN: {}".format(cnn_accy["grouped"]))
            cnn_curve["top1"].append(cnn_accy["top1"])
            print("CNN top1 curve: {}".format(cnn_curve["top1"]))


def args_parser():
    parser = argparse.ArgumentParser(description='benchmark for federated continual learning')
    # Exp settings
    parser.add_argument('--exp_name', type=str, default='lander_b0', help='name of this experiment')
    parser.add_argument('--wandb', type=int, default=None, help='1 for using wandb')
    parser.add_argument('--save_dir', type=str, default="", help='save the syn data')
    parser.add_argument('--project', type=str, default="LANDER", help='wandb project')
    parser.add_argument('--group', type=str, default="c100", help='wandb group')
    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--spec', type=str, default="t1", help='choose a model')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

    # federated continual learning settings
    parser.add_argument('--dataset', type=str, default="cifar100", help='which dataset')
    parser.add_argument('--tasks', type=int, default=5, help='num of tasks')
    parser.add_argument('--method', type=str, default="feat", help='choose a learner')
    parser.add_argument('--implementation', choices=('repo', 'paper'), default='paper',
                        help='FEAT training path: original repository or paper-based implementation')
    parser.add_argument('--net', type=str, default="resnet18", help='choose a model')
    parser.add_argument('--com_round', type=int, default=100, help='communication rounds')
    parser.add_argument('--num_class', type=int, default=10, help='all class number')
    parser.add_argument('--local_ep', type=int, default=2, help='local training epochs')
    parser.add_argument('--num_users', type=int, default=5, help='num of clients')
    parser.add_argument('--local_bs', type=int, default=128, help='local batch size')
    parser.add_argument('--beta', type=float, default=None, help='control the degree of label skew')
    parser.add_argument('--frac', type=float, default=1.0, help='the fraction of selected clients')
    parser.add_argument('--class_shuffle', type=int, default=1, help='class shuffle')

    # Data-free Generation
    parser.add_argument('--lr_g', default=2e-3, type=float, help='learning rate of generator')
    parser.add_argument('--synthesis_batch_size', default=256, type=int, help='synthetic data batch size')
    parser.add_argument('--bn', default=1.0, type=float, help='parameter for batchnorm regularization')
    parser.add_argument('--oh', default=0.5, type=float, help='parameter for similarity')
    parser.add_argument('--adv', default=1.0, type=float, help='parameter for diversity')
    parser.add_argument('--nz', default=256, type=int, help='output size of noisy nayer')
    parser.add_argument('--nums', type=int, default=10000, help='the num of synthetic data')
    parser.add_argument('--warmup', default=10, type=int,
                        help='number of epoches generator only warmups not stores images')
    parser.add_argument('--syn_round', default=40, type=int, help='number of synthetize round.')
    parser.add_argument('--g_steps', default=40, type=int, help='number of generation steps.')

    # Client Training
    parser.add_argument('--num_worker', type=int, default=4, help='number of worker for dataloader')
    parser.add_argument('--mulc', type=str, default="fork", help='type of multi process for dataloader')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='weight decay for optimizer')
    parser.add_argument('--syn_bs', default=1, type=int,
                        help='number of old synthetic data in training, 1 for similar to local_bs')
    parser.add_argument('--local_lr', default=4e-2, type=float, help='learning rate for optimizer')
    parser.add_argument('--kd', default=0.1, type=float, help='weight for kd')
    parser.add_argument('--mode', default='mse', type=str, help='compose_mode')
    parser.add_argument('--aug_mode', default='color_jitter', type=str,
                        help='augmentation_mode: hflip,vflip,color_jitter,invert')
    parser.add_argument('--n', default=1, type=int, help='number of retained dataset')
    parser.add_argument('--lambda', default=0.1, type=float, help='GSA loss weight')
    parser.add_argument('--temp', default=0.5, type=float, help='GSA temperature')
    parser.add_argument('--tttt', default=255, type=int, help='transparency setting')
    parser.add_argument('--tau_old', default=0.9, type=float, help='temperature parameter')
    parser.add_argument('--tau_new', default=1.1, type=float, help='temperature parameter')
    parser.add_argument('--scale', default=True, type=bool, help='weighted parameter')
    parser.add_argument('--mem_size', default=50, type=int, help='memory size')
    parser.add_argument('--w_old', default=1.1, type=float, help='weighted parameter')
    parser.add_argument('--w_new', default=None, type=float, help='weighted parameter')
    parser.add_argument('--syn_size', default=32, type=int, help='weighted parameter')
    parser.add_argument('--ema_decay', type=float, default=0.7, help='EGC prior EMA coefficient')
    parser.add_argument('--eval_each_round', action='store_true', help='evaluate after each communication round')
    parser.add_argument('--conv_dim', type=int, default=64)
    parser.add_argument('--tau', type=int, default=16)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--num_head', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.03125)
    parser.add_argument('--k_value', type=int, default=0)
    parser.add_argument('--p_value', type=int, default=0)
    # LANDER Exp5a-Global replay selector controls for the FEAT learner.
    parser.add_argument('--gdr_task_budget', type=int, default=None, help='Exp5 replay budget per task')
    parser.add_argument('--exp5a_rank', type=int, default=128, help='global trajectory projection rank')
    parser.add_argument('--exp5a_svd_oversampling', type=int, default=16)
    parser.add_argument('--exp5a_target_mode', choices=('budget_scaled_sum', 'full_sum'),
                        default='budget_scaled_sum')
    parser.add_argument('--exp5a_tau', type=float, default=0.05,
                        help='trajectory attribution invariant tolerance')
    parser.add_argument('--repeat_rate', type=int, default=1,
                        help='virtual replay copies per retained item')
    parser.add_argument('--sample_weighted_fedavg', action='store_true')
    exp5_loss = parser.add_mutually_exclusive_group()
    exp5_loss.add_argument('--repo_dual', action='store_true')
    exp5_loss.add_argument('--exp5_distill_loss', action='store_true')
    exp5_loss.add_argument('--exp5_single_distill_loss', action='store_true')
    parser.add_argument('--logits_mode', choices=('none', 'LTE_margin', 'proto_margin', 'global_margin'),
                        default='none')
    parser.add_argument('--logits_lambda', type=float, default=0.1)
    parser.add_argument('--logits_gamma', type=float, default=0.5)
    parser.add_argument('--logits_temperature', type=float, default=2.0)
    parser.add_argument('--logits_margin', type=float, default=1.0)
    parser.add_argument('--logits_warmup', type=int, default=10)
    parser.add_argument('--logits_refresh', type=int, default=5)
    parser.add_argument('--logits_proto_bs', type=int, default=256)
    parser.add_argument('--logits_lte_path', type=str, default=None)
    parser.add_argument('--logits_output_dir', type=str, default='')
    parser.add_argument('--logits_probe_rounds', type=str, default='')
    parser.add_argument('--eval_interval', type=int, default=1)
    parser.add_argument('--fedcbdr_lr_scheduler', choices=('constant', 'cosine', 'multi_step'),
                        default='constant')
    parser.add_argument('--fast_cuda', action='store_true')
    parser.add_argument('--t4_parralel', '--t4_parallel', dest='t4_parralel', action='store_true')
    # LANDER
    parser.add_argument('--r', default=0.015, type=float, help='LTE center radius')
    parser.add_argument('--ltc', default=5, type=float, help='lamda_ltc parameter for LTE center')
    parser.add_argument('--pre', type=float, default=0.4, help='alpha_pre for distilling from previous task')
    parser.add_argument('--cur', type=float, default=0.2, help='alpha_cur for current task training')
    parser.add_argument('--type', default=-1, type=int,
                        help='seed for initializing training.')  # 0 for train forward, 1 pretrain stage 1, 2 pretrain stage 2
    parser.add_argument('--syn', default=1, type=int,
                        help='seed for initializing training.')  # 0 for train forward, 1 pretrain stage 1, 2 pretrain stage 2
    args = parser.parse_args()

    exp5 = args.method.lower() == 'exp5a_global'
    if args.beta is None:
        args.beta = 0.5
    if args.wandb is None:
        args.wandb = 0 if exp5 else 1
    if args.w_new is None:
        args.w_new = 1.0
    if exp5:
        if (args.repeat_rate != 1 or args.sample_weighted_fedavg or args.repo_dual or
                args.exp5_distill_loss or args.exp5_single_distill_loss or
                args.fast_cuda or args.t4_parralel or args.eval_interval != 1 or
                args.fedcbdr_lr_scheduler != 'constant'):
            parser.error('LANDER training and loss switches are unavailable with FEAT training')
        if args.gdr_task_budget is None:
            budgets = {
                'cifar10': {3: 450, 5: 300},
                'cifar100': {5: 1000, 10: 500, 20: 250},
                'tiny_imagenet': {10: 2000, 20: 1000},
            }
            try:
                args.gdr_task_budget = budgets[args.dataset][args.tasks]
            except KeyError:
                parser.error('Specify --gdr_task_budget for this dataset/task count')
        if args.gdr_task_budget < 1:
            parser.error('--gdr_task_budget must be positive')
        if args.exp5a_rank < 1 or args.exp5a_svd_oversampling < 0 or args.exp5a_tau < 0:
            parser.error('invalid Exp5 projection rank, oversampling, or tolerance')
    if args.logits_mode != 'none':
        parser.error('--logits_mode is unavailable with FEAT training')

    return args


if __name__ == '__main__':

    args = args_parser()
    if args.dataset == "tiny_imagenet":
        args.num_class = 200
    elif args.dataset == "cifar100":
        args.num_class = 100
    elif args.dataset == "imagenet100":
        args.num_class = 100
    elif args.dataset == "cifar10":
        args.num_class = 10
    elif args.dataset == "imagenet":
        args.num_class = 1000
    args.init_cls = int(args.num_class / args.tasks)
    args.increment = args.init_cls

    print(args)

    args.exp_name = f"{args.beta}_{args.method}_{args.exp_name}"

    dir = "run"
    if not os.path.exists(dir):
        os.makedirs(dir)
    if args.method.lower() != 'exp5a_global' or not args.save_dir:
        args.save_dir = os.path.join(dir, args.group + "_" + args.exp_name + "" + args.spec)

    if args.method.lower() == 'exp5a_global' and args.wandb == 1:
        if wandb is None:
            raise RuntimeError('--wandb=1 requires the wandb package')
        wandb.init(config=args, project=args.project, group=args.group, name=args.exp_name)
        wandb.run.log_code('.')

    # if args.wandb == 1:
    #     wandb.init(config=args, project=args.project, group=args.group, name=args.exp_name)
    #     wandb.run.log_code(".")
    args = vars(args)

    train(args)
