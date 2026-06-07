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
    # if not args.testing:
    args.split_dir = os.path.join("splits", args.which_splits, args.study)
    args.combined_study = args.study
    args = _get_custom_exp_code(args)
    _seed_torch(args.seed)
    if not args.testing:
        os.path.isdir(args.split_dir)
        print('Split dir:', args.split_dir)

    #---> where to stroe the experiment related assets
    _create_results_dir(args)

    #---> store the settings
    settings = {'num_splits': args.k, #5
                'k_start': args.k_start, #-1
                'k_end': args.k_end, #-1
                'task': args.task, #response
                'max_epochs': args.max_epochs,  #2
                'results_dir': args.results_dir, 
                'lr': args.lr, #0.0005
                'experiment': args.study, #tcga_brca_demo
                'reg': args.reg, #0.0001
                # 'label_frac': args.label_frac,
                'bag_loss': args.bag_loss, #ce
                'seed': args.seed, #1
                'weighted_sample': args.weighted_sample, #True
                'opt': args.opt, #radam
                "num_patches":args.num_patches, #4096
                "dropout":args.encoder_dropout, #0.25
                "type_of_path":args.type_of_path, #combine
                'split_dir': args.split_dir #splits/5foldcv/tcga_brca_demo
                }
    
    #---> bookkeping
    _print_and_log_experiment(args, settings)

    #---> load composition df 
    composition_df = pd.read_csv("./datasets_csv/pathway_compositions/{}_comps.csv".format(args.type_of_path), index_col=0)
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
    dataset_path = 'datasets_csv/all_survival_endpoints'
    param_code = ''

    #----> Study 
    param_code += args.study + "_" #tcga_brac_demo

    #----> Loss Function
    param_code += '_%s' % args.bag_loss
    param_code += '_a%s' % str(args.alpha_surv) #0.5
    
    #----> Learning Rate
    param_code += '_lr%s' % format(args.lr, '.0e') #0.0005

    #----> Regularization
    # if args.reg_type == 'L1':
    #   param_code += '_%sreg%s' % (args.reg_type, format(args.reg, '.0e'))

    # if args.reg and args.reg_type == "L2":
    param_code += "_l2Weight_{}".format(args.reg) #0.0001

    param_code += '_%s' % args.which_splits.split("_")[0] #5foldcv

    #----> Batch Size
    param_code += '_b%s' % str(args.batch_size) #1

    # label col 
    param_code += "_" + args.label_col #response_lable

    param_code += "_dim1_" + str(args.encoding_dim) #1024
    # param_code += "_dim2_" + str(args.encoding_layer_2_dim)
    
    param_code += "_patches_" + str(args.num_patches) #4096
    # param_code += "_dropout_" + str(args.encoder_dropout)

    param_code += "_wsiDim_" + str(args.wsi_projection_dim) #256
    param_code += "_epochs_" + str(args.max_epochs) #2
    param_code += "_fusion_" + str(args.fusion) #none
    param_code += "_modality_" + str(args.modality) #surpath
    param_code += "_pathT_" + str(args.type_of_path) #combine
    param_code += "_feature_" + str(args.data_root_dir_cls256.split('/')[-1])
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


def _collate_omics(batch):
    r"""
    Collate function for the unimodal omics models 
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omics : torch.Tensor 
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        
    """
  
    img1 = torch.stack([item[0] for item in batch])
    img2 = torch.stack([item[5] for item in batch])
    img3 = torch.stack([item[6] for item in batch])
    omics = torch.stack([item[1] for item in batch], dim = 0)
    label = torch.LongTensor([item[2].long() for item in batch])
    # event_time = torch.FloatTensor([item[3] for item in batch])
    # c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[3])
    mask = torch.stack([item[4] for item in batch], dim=0)
    case_id = [item[7] for item in batch]
    idx = [item[8] for item in batch]
    # return [img, omics, label, event_time, c, clinical_data_list]
    return [img1, omics, label, clinical_data_list, mask, img2, img3, case_id, idx]


def _collate_wsi_omics(batch):
    r"""
    Collate function for the unimodal wsi and multimodal wsi + omics  models 
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omics : torch.Tensor 
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        - mask : torch.Tensor
        
    """
  
    img1 = torch.stack([item[0] for item in batch])
    img2 = torch.stack([item[5] for item in batch])
    img3 = torch.stack([item[6] for item in batch])
    omics = torch.stack([item[1] for item in batch], dim = 0)
    label = torch.LongTensor([item[2].long() for item in batch])
    # event_time = torch.FloatTensor([item[3] for item in batch])
    # c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[3])

    mask = torch.stack([item[4] for item in batch], dim=0)
    case_id = [item[7] for item in batch]
    idx = [item[8] for item in batch]
    # return [img, omics, label, event_time, c, clinical_data_list, mask]
    return [img1, omics, label, clinical_data_list, mask, img2, img3, case_id, idx]

def _collate_MCAT(batch):
    r"""
    Collate function MCAT (pathways version) model
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omic1 : torch.Tensor 
        - omic2 : torch.Tensor 
        - omic3 : torch.Tensor 
        - omic4 : torch.Tensor 
        - omic5 : torch.Tensor 
        - omic6 : torch.Tensor 
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        
    """
    
    img = torch.stack([item[0] for item in batch])

    omic1 = torch.cat([item[1] for item in batch], dim = 0).type(torch.FloatTensor)
    omic2 = torch.cat([item[2] for item in batch], dim = 0).type(torch.FloatTensor)
    omic3 = torch.cat([item[3] for item in batch], dim = 0).type(torch.FloatTensor)
    omic4 = torch.cat([item[4] for item in batch], dim = 0).type(torch.FloatTensor)
    omic5 = torch.cat([item[5] for item in batch], dim = 0).type(torch.FloatTensor)
    omic6 = torch.cat([item[6] for item in batch], dim = 0).type(torch.FloatTensor)


    label = torch.LongTensor([item[7].long() for item in batch])
    event_time = torch.FloatTensor([item[8] for item in batch])
    c = torch.FloatTensor([item[9] for item in batch])

    clinical_data_list = []
    for item in batch:
        clinical_data_list.append(item[10])

    mask = torch.stack([item[11] for item in batch], dim=0)

    return [img, omic1, omic2, omic3, omic4, omic5, omic6, label, event_time, c, clinical_data_list, mask]

def _collate_survpath(batch):
    r"""
    Collate function for survpath
    
    Args:
        - batch 
    
    Returns:
        - img : torch.Tensor 
        - omic_data_list : List
        - label : torch.LongTensor 
        - event_time : torch.FloatTensor 
        - c : torch.FloatTensor 
        - clinical_data_list : List
        - mask : torch.Tensor
        
    """
    
    img1 = torch.stack([item[0] for item in batch])
    img2 = torch.stack([item[5] for item in batch])
    img3 = torch.stack([item[6] for item in batch])
    omic_data_list = []
    for item in batch:
        omic_data_list.append(item[1])

    label = torch.LongTensor([item[2].long() for item in batch])
    # event_time = torch.FloatTensor([item[3] for item in batch])
    # c = torch.FloatTensor([item[4] for item in batch])

    clinical_data_list = []
    for item in batch:
        # clinical_data_list.append(item[5])
        clinical_data_list.append(item[3])

    # mask = torch.stack([item[6] for item in batch], dim=0)
    mask = torch.stack([item[4] for item in batch], dim=0)
    case_id = [item[7] for item in batch]
    idx = [item[8] for item in batch]
    coords = [item[9] for item in batch]
    # return [img, omic_data_list, label, event_time, c, clinical_data_list, mask]
    return [img1, omic_data_list, label, clinical_data_list, mask, img2, img3, case_id, idx,coords]

def _make_weights_for_balanced_classes_split(dataset):
    r"""
    Returns the weights for each class. The class will be sampled proportionally.
    
    Args: 
        - dataset : SurvivalDataset
    
    Returns:
        - final_weights : torch.DoubleTensor 
    
    """
    N = float(len(dataset))   #288                               
    weight_per_class = [N/len(dataset.slide_cls_ids[c]) for c in range(len(dataset.slide_cls_ids))]     #1.56, 2.76                                                                                                
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

    kwargs = {'num_workers': 8} if device.type == "cuda" else {}
    
    if args.modality in ["omics", "snn", "mlp_per_path"]:
        collate_fn = _collate_omics
    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways","Frmil_wsi", "Frmil_wsi_pathways"]:
        collate_fn = _collate_wsi_omics
    elif args.modality in ["coattn", "coattn_motcat"]:  
        collate_fn = _collate_survpath #_collate_MCAT
    elif args.modality == "survpath":
         collate_fn = _collate_survpath 
    elif args.modality == "coattn_cmta":
         collate_fn = _collate_survpath 
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
