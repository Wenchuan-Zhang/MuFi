#----> general imports
import torch
import numpy as np
import torch.nn as nn
import pdb
import os
import pandas as pd 

import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler, WeightedRandomSampler, RandomSampler, SequentialSampler, sampler

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _prepare_for_experiment(args):
    r"""
    Creates experiment code which will be used for identifying the experiment later on. Uses the experiment code to make results dir.
    Prints and logs the important settings of the experiment. Loads the pathway composition dataframe and stores in args for future use.

    Args:
        - args : argparse.Namespace
    
    Returns:
        - args : argparse.Namespace

    """

    args.device = device
    print(args.device)
    args.split_dir = os.path.join("splits", args.which_splits, args.study)
    args.combined_study = args.study
    args = _get_custom_exp_code(args)
    _seed_torch(args.seed)

    assert os.path.isdir(args.split_dir)
    print('Split dir:', args.split_dir)

    #---> where to stroe the experiment related assets
    _create_results_dir(args)

    #---> store the settings
    settings = {'num_splits': args.k, 
                'k_start': args.k_start,
                'k_end': args.k_end,
                'task': args.task,
                'max_epochs': args.max_epochs, 
                'results_dir': args.results_dir, 
                'lr': args.lr,
                'experiment': args.study,
                'reg': args.reg,
                # 'label_frac': args.label_frac,
                'bag_loss': args.bag_loss,
                'seed': args.seed,
                'weighted_sample': args.weighted_sample,
                'opt': args.opt,
                "num_patches":args.num_patches,
                "dropout":args.encoder_dropout,
                "type_of_path":args.type_of_path,
                'split_dir': args.split_dir
                }
    
    #---> bookkeping
    _print_and_log_experiment(args, settings)

    #---> load composition df 
    composition_df = pd.read_csv("./datasets_csv/subsets_compositions/{}_comps.csv".format(args.type_of_path), index_col=0)
    composition_df.sort_index(inplace=True)
    args.composition_df = composition_df

    return args

def _print_and_log_experiment(args, settings):
    r"""
    Prints the expeirmental settings and stores them in a file 
    
    Args:
        - args : argspace.Namespace
        - settings : dict 
    
    Return:
        - None
        
    """
    with open(args.results_dir + '/experiment_{}.txt'.format(args.param_code), 'w') as f:
        print(settings, file=f)

    f.close()

    print("")
    print("################# Settings ###################")
    for key, val in settings.items():
        print("{}:  {}".format(key, val))
    print("")

def _get_custom_exp_code(args):
    r"""
    Updates the argparse.NameSpace with a custom experiment code.

    Args:
        - args (NameSpace)

    Returns:
        - args (NameSpace)

    """
    dataset_path = 'datasets_csv/all_outcomes'
    param_code = ''

    #----> Study 
    param_code += args.study + "_"

    #----> Loss Function
    param_code += '_%s' % args.bag_loss
    param_code += '_a%s' % str(args.alpha_surv)
    
    #----> Learning Rate
    param_code += '_lr%s' % format(args.lr, '.0e')

    #----> Regularization
    # if args.reg_type == 'L1':
    #   param_code += '_%sreg%s' % (args.reg_type, format(args.reg, '.0e'))

    # if args.reg and args.reg_type == "L2":
    param_code += "_l2Weight_{}".format(args.reg)

    param_code += '_%s' % args.which_splits.split("_")[0]

    #----> Batch Size
    param_code += '_b%s' % str(args.batch_size)

    # label col 
    param_code += "_" + args.label_col

    param_code += "_dim1_" + str(args.encoding_dim)
    # param_code += "_dim2_" + str(args.encoding_layer_2_dim)
    
    param_code += "_patches_" + str(args.num_patches)
    # param_code += "_dropout_" + str(args.encoder_dropout)

    param_code += "_wsiDim_" + str(args.wsi_projection_dim)
    param_code += "_epochs_" + str(args.max_epochs)
    param_code += "_fusion_" + str(args.fusion)
    param_code += "_modality_" + str(args.modality)
    param_code += "_pathT_" + str(args.type_of_path)

    #----> Updating
    args.param_code = param_code
    args.dataset_path = dataset_path

    return args


def _seed_torch(seed=7):
    r"""
    Sets custom seed for torch 

    Args:
        - seed : Int 
    
    Returns:
        - None

    """
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _create_results_dir(args):
    r"""
    Creates a dir to store results for this experiment. Adds .gitignore 
    
    Args:
        - args: argspace.Namespace
    
    Return:
        - None 
    
    """
    args.results_dir = os.path.join("./results", args.results_dir) # create an experiment specific subdir in the results dir 
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
        #---> add gitignore to results dir
        f = open(os.path.join(args.results_dir, ".gitignore"), "w")
        f.write("*\n")
        f.write("*/\n")
        f.write("!.gitignore")
        f.close()
    
    #---> results for this specific experiment
    args.results_dir = os.path.join(args.results_dir, args.param_code)
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

def _get_start_end(args):
    r"""
    Which folds are we training on
    
    Args:
        - args : argspace.Namespace
    
    Return:
       folds : np.array 
    
    """
    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end
    folds = np.arange(start, end)
    return folds

def _save_splits(split_datasets, column_keys, filename, boolean_style=False):
    splits = [split_datasets[i].metadata['slide_id'] for i in range(len(split_datasets))]
    if not boolean_style:
        df = pd.concat(splits, ignore_index=True, axis=1)
        df.columns = column_keys
    else:
        df = pd.concat(splits, ignore_index = True, axis=0)
        index = df.values.tolist()
        one_hot = np.eye(len(split_datasets)).astype(bool)
        bool_array = np.repeat(one_hot, [len(dset) for dset in split_datasets], axis=0)
        df = pd.DataFrame(bool_array, index=index, columns = ['train', 'val'])

    df.to_csv(filename)
    print()

def _series_intersection(s1, s2):
    r"""
    Return insersection of two sets
    
    Args:
        - s1 : set
        - s2 : set 
    
    Returns:
        - pd.Series
    
    """
    return pd.Series(list(set(s1) & set(s2)))

def _print_network(results_dir, net):
    r"""

    Print the model in terminal and also to a text file for storage 
    
    Args:
        - results_dir : String 
        - net : PyTorch model 
    
    Returns:
        - None 
    
    """
    num_params = 0
    num_params_train = 0

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n

    print('Total number of parameters: %d' % num_params)
    print('Total number of trainable parameters: %d' % num_params_train)

    # print(net)

    fname = "model_" + results_dir.split("/")[-1] + ".txt"
    path = os.path.join(results_dir, fname)
    f = open(path, "w")
    f.write(str(net))
    f.write("\n")
    f.write('Total number of parameters: %d \n' % num_params)
    f.write('Total number of trainable parameters: %d \n' % num_params_train)
    f.close()


def _collate_radiomics(batch):
    r"""
    Collate function for the unimodal radiomics models 
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - radiomics : torch.Tensor 
        - label : torch.LongTensor 
        - clinical_data_list : List
        
    """
  
    img = torch.ones([1,1])
    radiomics = torch.stack([item[1] for item in batch], dim = 0)
    label = torch.LongTensor([item[2].long() for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    return [img, radiomics, label, clinical_data_list]


def _collate_wsi_radiomics(batch):
    r"""
    Collate function for the unimodal wsi and multimodal wsi + radiomics  models 
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - radiomics : torch.Tensor 
        - label : torch.LongTensor 
        - clinical_data_list : List
        - mask : torch.Tensor
        
    """
  
    img = torch.stack([item[0] for item in batch])
    radiomics = torch.stack([item[1] for item in batch], dim = 0)
    label = torch.LongTensor([item[2].long() for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    mask = torch.stack([item[6] for item in batch], dim=0)

    return [img, radiomics, label, clinical_data_list, mask]


def _collate_MuFi(batch):
    r"""
    Collate function for MuFi
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - radiomic_data_list : List
        - label : torch.LongTensor 
        - clinical_data_list : List
        - mask : torch.Tensor
        
    """
    
    img = torch.stack([item[0] for item in batch])

    radiomic_data_list = []
    for item in batch:
        radiomic_data_list.append(item[1])

    label = torch.LongTensor([item[2].long() for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[5])

    mask = torch.stack([item[6] for item in batch], dim=0)

    return [img, radiomic_data_list, label, clinical_data_list, mask]

def _make_weights_for_balanced_classes_split(dataset):
    r"""
    Returns the weights for each class. The class will be sampled proportionally.
    
    Args: 
        - dataset : pCRDataset
    
    Returns:
        - final_weights : torch.DoubleTensor 
    
    """
    N = float(len(dataset))                                           
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]                                                                                                     
    weight = [0] * int(N)                                           
    for idx in range(len(dataset)):   
        y = dataset.getlabel(idx)                   
        weight[idx] = weight_per_class[y]   

    final_weights = torch.DoubleTensor(weight)

    return final_weights

class SubsetSequentialSampler(Sampler):
	"""Samples elements sequentially from a given list of indices, without replacement.

	Arguments:
		indices (sequence): a sequence of indices
	"""
	def __init__(self, indices):
		self.indices = indices

	def __iter__(self):
		return iter(self.indices)

	def __len__(self):
		return len(self.indices)


def _get_split_loader(args, split_dataset, training = False, testing = False, weighted = False, batch_size=1):
    r"""
    Take a dataset and make a dataloader from it using a custom collate function. 

    Args:
        - args : argspace.Namespace
        - split_dataset : SurvivalDataset
        - training : Boolean
        - testing : Boolean
        - weighted : Boolean 
        - batch_size : Int 
    
    Returns:
        - loader : Pytorch Dataloader 
    
    """

    kwargs = {'num_workers': 12} if device.type == "cuda" else {}
    
    if args.modality in ["radiomics", "snn", "mlp_per_path"]:
        collate_fn = _collate_radiomics
    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways", "frmil_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways"]:
        collate_fn = _collate_wsi_radiomics
    elif args.modality == "MuFi":
         collate_fn = _collate_MuFi 
    else:
        raise NotImplementedError

    if not testing:
        if training:
            if weighted:
                weights = _make_weights_for_balanced_classes_split(split_dataset)
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn = collate_fn, drop_last=False, **kwargs)	
            else:
                loader = DataLoader(split_dataset, batch_size=batch_size, sampler = RandomSampler(split_dataset), collate_fn = collate_fn, drop_last=False, **kwargs)
        else:
            loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SequentialSampler(split_dataset), collate_fn = collate_fn, drop_last=False, **kwargs)

    else:
        ids = np.random.choice(np.arange(len(split_dataset), int(len(split_dataset)*0.1)), replace = False)
        loader = DataLoader(split_dataset, batch_size=batch_size, sampler = SubsetSequentialSampler(ids), collate_fn = collate_fn, drop_last=False, **kwargs )

    return loader
