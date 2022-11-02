import argparse
import collections
import json
import os
import random
import sys
import time
import uuid
import copy
import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets_aug
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc_aug,misc
from domainbed.lib.fast_data_loader_aug import InfiniteDataLoader, FastDataLoader, UnNormalize
from torchvision import transforms
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str,default= 'DATA')
    parser.add_argument('--csv_root', type= str,default= 'PACS_splits/sketch/seed_12')
    parser.add_argument('--dataset', type=str, default="PACS")
    parser.add_argument('--algorithm', type=str, default="DREAME")
    parser.add_argument('--task', type=str, default="domain_generalization",
        help='domain_generalization | domain_adaptation')
    parser.add_argument('--hparams', type=str,default= '{"batch_size":2,"data_augmentation":0}',
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[1])
    parser.add_argument('--output_dir', type=str, default="Inevnio_validationaug_train_aug_heldout_val_debug/OfficeHome_M3_trial_seed_2/env0/indiv_ensemble_no_beta")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0)
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true',default=True)
    parser.add_argument('--compute_test_beta_DREAME',default=False)
    parser.add_argument('--out_augs',default=True, help = "augmentations for out splits of observed domains")
    parser.add_argument('--split_indata',default=True,help="create held out validation \
        and use a portion of train data as meta test")
    parser.add_argument('--user_defined_step',  default=-1)
    args = parser.parse_args()
    compute_test_beta= args.compute_test_beta_DREAME

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc_aug.Tee(os.path.join(args.output_dir, 'out_eval.txt'))
    sys.stderr = misc_aug.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    #print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc_aug.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets_aug):
        if args.dataset=='PACS_splits': 
            dataset = vars(datasets_aug)[args.dataset](args.csv_root,args.data_dir,args.test_envs,hparams)
        else:
            dataset = vars(datasets_aug)[args.dataset](args.data_dir,
                args.test_envs, hparams, out_augs = True)

    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.
    
    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    if args.split_indata:
        in_splits = []
        out_splits = []
        uda_splits = []
        in_val_splits=[]
        for env_i, envs in enumerate(dataset):
            in_vals = []
            ins_ = []
            outs_ = []
            uda = []
            for tfm_idx, env in enumerate(envs):
                if env_i in args.test_envs:
                    out, in_ = misc_aug.split_dataset(env, int(len(env)*args.holdout_fraction), misc_aug.seed_hash(args.trial_seed, env_i))
                    outs_.append(out)
                    ins_.append(in_)
                    in_val, in_ = misc_aug.split_dataset(in_, int(len(in_)*args.holdout_fraction), misc_aug.seed_hash(args.trial_seed, env_i))
                    in_vals.append(in_val)
                else: # traain_env
                    out, in_ = misc.split_dataset(env,
                        int(len(env)*args.holdout_fraction),
                        misc.seed_hash(args.trial_seed, env_i))
                    in_val, in_ = misc_aug.split_dataset(in_, int(len(in_)*args.holdout_fraction), misc_aug.seed_hash(args.trial_seed, env_i))
                    in_vals.append(in_val)
                    if tfm_idx == 0:
                        ins_.append(in_) #append only the datasets without our out_augs as the in splits
                        outs_.append(out) #make sure no augmentation on out for train env val val
                    
            
            if env_i in args.test_envs:
                uda, in_ = misc_aug.split_dataset(in_,
                    int(len(in_)*args.uda_holdout_fraction),
                    misc_aug.seed_hash(args.trial_seed, env_i))

            if hparams['class_balanced']:
                in_weights = misc_aug.make_weights_for_balanced_classes(in_)
                out_weights = [misc_aug.make_weights_for_balanced_classes(out) for out in outs_]
                if uda is not None:
                    uda_weights = misc_aug.make_weights_for_balanced_classes(uda)
            else:
                in_weights,in_val_weights, out_weights, uda_weights = None, None, None , None
            in_splits.append((ins_[0], in_weights))
            out_splits.append((outs_[0],out_weights))
            in_val_splits.append([(in_vals[i], in_val_weights) for i in range(len(in_vals))])
            if len(uda):
                uda_splits.append((uda, uda_weights))
        
        orig_in_val_splits = in_val_splits
        in_val_splits = [item for sublist in in_val_splits for item in sublist]
    # in_splits = []
    # out_splits = []
    # uda_splits = []
    # for env_i, envs in enumerate(dataset):
    #     outs = []
    #     ins_ = []
    #     uda = []
    #     for tfm_idx, env in enumerate(envs):
    #         if env_i in args.test_envs:
    #             out, in_ = misc.split_dataset(env,
    #                         int(len(env)*args.holdout_fraction),
    #                         misc.seed_hash(args.trial_seed, env_i))
                
    #             out, in_ = misc_aug.split_dataset(env, int(len(env)*args.holdout_fraction), misc_aug.seed_hash(args.trial_seed, env_i))
    #             outs.append(out)
    #             ins_.append(in_)
    #         else:
    #             out, in_ = misc_aug.split_dataset(env, int(len(env)*args.holdout_fraction), misc_aug.seed_hash(args.trial_seed, env_i))
    #             outs.append(out)
    #             if tfm_idx == 0:
    #                 ins_.append(in_) #append only the datasets without our out_augs as the in splits

        
    #     if env_i in args.test_envs:
    #         uda, in_ = misc_aug.split_dataset(in_,
    #             int(len(in_)*args.uda_holdout_fraction),
    #             misc_aug.seed_hash(args.trial_seed, env_i))

    #     if hparams['class_balanced']:
    #         in_weights = misc_aug.make_weights_for_balanced_classes(in_)
    #         out_weights = [misc_aug.make_weights_for_balanced_classes(out) for out in outs]
    #         if uda is not None:
    #             uda_weights = misc_aug.make_weights_for_balanced_classes(uda)
    #     else:
    #         in_weights, out_weights, uda_weights = None, None , None
    #     in_splits.append((ins_[0], in_weights))
    #     out_splits.append([(outs[i], out_weights) for i in range(len(outs))])
    #     if len(uda):
    #         uda_splits.append((uda, uda_weights))
    
    # orig_out_splits = out_splits
    # out_splits = [item for sublist in out_splits for item in sublist]

    # train_envs = np.setdiff1d(np.arange(len(orig_out_splits)), args.test_envs)
    # test_out_split_idx = args.test_envs * len(train_envs)
    # test_out_split = out_splits[test_out_split_idx]
    # train_out_splits = out_splits.remove(test_out_split_idx)
    # out_splits = train_out_splits.append(test_out_split) # append test out split towards the end 


    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]
    
    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]
    
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]

    if args.split_indata: 
        DREAME_mata_out_splits = in_val_splits
    else: 
        DREAME_mata_out_splits = out_splits


    to_drop_from_meta_test_out_splits = [t*(len(dataset)-len(args.test_envs)) for t in args.test_envs]
    val_loaders_DREAME = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(DREAME_mata_out_splits) if i not in to_drop_from_meta_test_out_splits]
    
    eval_weights = [None for _, weights in (in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in{}'.format(i, 0)
        for i in range(len(in_splits))]
    # train_envs = np.setdiff1d(np.arange(len(orig_out_splits)), args.test_envs)
    # eval_loader_names = ['env{}_out'.format(test_out_split_idx)]
    eval_loader_names += ['env{}_out{}'.format(i,0)
        for i in range(len(out_splits))]
    
    # for i, envs in enumerate(orig_out_splits):
    #     for j in range(len(envs)):
    #         eval_loader_names.append('env{}_out{}'.format(i, j))

    # eval_loader_names += ['env{}_out'.format(i)
    #     for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda{}'.format(i,i)
        for i in range(len(uda_splits))]
    

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    val_minibatches_iterator_DREAME = zip(*val_loaders_DREAME)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size'] for env,_ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.cpu().state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))
    
    def load_model(filename):
        dump = torch.load(filename)
        algorithm_class = algorithms.get_algorithm_class(dump["args"]["algorithm"])
        algorithm = algorithm_class(
            dump["model_input_shape"],
            dump["model_num_classes"],
            dump["model_num_domains"],
            dump["model_hparams"])
        algorithm.load_state_dict(dump["model_dict"])
        return algorithm

    last_results_keys = None

    ckpt_names = [os.path.join(args.output_dir,f) for f in os.listdir(args.output_dir) if 'model' in f and 'preds' not in f and 'selected' not in f] 
    try:
        ckpt_names.remove(os.path.join(args.output_dir,'model.pkl'))# remove the last checkpoint as it is already saved
    except:
        pass
    ckpt_names= sorted(ckpt_names,key= lambda x: int(x.split(args.output_dir)[1].split('model_')[1].split('step')[1].strip('.pkl')))
    
    if int(args.user_defined_step)>0:
        ckpt_names= [os.path.join(args.output_dir,'model_step'+str(int(args.user_defined_step))+'.pkl')]


    for i in range( len(ckpt_names)):

        algorithm = load_model(ckpt_names[i])
        algorithm= algorithm.to(device)
        evals = zip(eval_loader_names, eval_loaders, eval_weights)
        # if nt(args.user_defined_step)>0: that means we need not compute only for that checkpoint else for all.
        # for now just dump the entropies for all. 
        # We nned only outs for obs and ins for obs

        step_number =   int(ckpt_names[i].split(args.output_dir)[1].split('model_')[1].split('step')[1].strip('.pkl'))
        if args.algorithm == 'DREAME': 
            eval_dict ={}
            for name, loader, weights in evals:
                eval_dict[name]= [loader,weights] 
            models_selected = [1,1,1]#step_vals['models_selected'] # random stuff. we will not use it anyway
            
            correct_models_selected_for_each_domain = np.nan* np.ones(len(dataset))
            train_envs = [ i for i in range(len(dataset)) if i not in args.test_envs]
            for t,m in zip(train_envs,models_selected):
                correct_models_selected_for_each_domain[t]=m
            acc_flags={'ensemble_for_obs':True,'compute_test_beta':False}

            results_DREAME = misc_aug.DREAME_accuracy(algorithm, eval_dict, args.test_envs, correct_models_selected_for_each_domain,device,acc_flags)
            results_DREAME['step']= step_number
            results_keys  = sorted([k for k in results_DREAME.keys() if 'acc' in k])
            results_keys.append('step')
           
            if i ==0:
                misc_aug.print_row(results_keys, colwidth=25)
            misc_aug.print_row([results_DREAME[key] for key in results_keys],colwidth=25)

            misc.save_obj_with_filename(results_DREAME,os.path.join(args.output_dir,'entropies_labels_step'+str(step_number)+'.pkl'))
            

