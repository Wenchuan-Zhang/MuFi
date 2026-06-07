from ast import Lambda
import numpy as np
import pdb
import os
import torch.nn as nn
from custom_optims.radam import RAdam
from models.model_ABMIL import ABMIL
from models.model_DeepMISL import DeepMISL
from models.model_MLPOmics import MLPOmics
from models.model_MLPWSI import MLPWSI
from models.model_SNNOmics import SNNOmics
from models.model_MaskedOmics import MaskedOmics
from models.model_MCATPathways import MCATPathways
from models.model_SurvPath import SurvPath
from models.model_SurvPath_with_nystrom import SurvPath_with_nystrom
from models.model_TMIL import TMIL
from models.model_FRMIL import FRMIL
from models.model_motcat import MCATPathwaysMotCat
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw, brier_score, integrated_brier_score, cumulative_dynamic_auc
from sksurv.util import Surv
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support, accuracy_score, f1_score

from transformers import (
    get_constant_schedule_with_warmup, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup
)


#----> pytorch imports
import torch
from torch.nn.utils.rnn import pad_sequence

from utils.general_utils import _get_split_loader, _print_network, _save_splits
from utils.loss_func import NLLSurvLoss

import torch.optim as optim
from file_utils import save_pkl #addnew
import pandas as pd
from vis_utils.heatmap_utils import initialize_wsi, drawHeatmap
from wsi_core.wsi_utils import sample_rois
# from torch.cuda.amp import autocast, GradScaler #addnew
# from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
def _get_splits(datasets, cur, args):
    r"""
    Summarize the train and val splits and return them individually
    
    Args:
        - datasets : tuple
        - cur : Int 
        - args: argspace.Namespace
    
    Return:
        - train_split : SurvivalDataset
        - val_split : SurvivalDataset
    
    """

    print('\nTraining Fold {}!'.format(cur))
    print('\nInit train/val splits...', end=' ')
    train_split, val_split = datasets
    _save_splits(datasets, ['train', 'val'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))

    return train_split,val_split


def _init_loss_function(args):
    r"""
    Init the survival loss function
    
    Args:
        - args : argspace.Namespace 
    
    Returns:
        - loss_fn : NLLSurvLoss or NLLRankSurvLoss
    
    """
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'ce':
        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = nn.CrossEntropyLoss(weight= torch.tensor([1.565, 15], dtype=torch.float, device='cuda'))
    else:
        raise NotImplementedError
    print('Done!')
    return loss_fn

def _init_optim(args, model):
    r"""
    Init the optimizer 
    
    Args: 
        - args : argspace.Namespace 
        - model : torch model 
    
    Returns:
        - optimizer : torch optim 
    """
    print('\nInit optimizer ...', end=' ')

    if args.opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.reg)
    elif args.opt == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "radam":
        optimizer = RAdam(model.parameters(), lr=args.lr, weight_decay=args.reg)
    elif args.opt == "lamb":
        optimizer = Lambda(model.parameters(), lr=args.lr, weight_decay=args.reg)
    else:
        raise NotImplementedError

    return optimizer

def _init_model(args):
    
    print('\nInit Model...', end=' ')
    if args.type_of_path == "xena":
        omics_input_dim = 1577
    elif args.type_of_path == "hallmarks":
        omics_input_dim = 4241
    elif args.type_of_path == "combine":
        # omics_input_dim = 4999
        omics_input_dim = 591 #3655
    elif args.type_of_path == "multi":
        if args.study == "tcga_brca":
            omics_input_dim = 9947
        else:
            omics_input_dim = 14933
    else:
        omics_input_dim = 0
    
    # omics baselines
    if args.modality == "mlp_per_path":

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "dropout" : args.encoder_dropout, "num_classes" : args.n_classes,
        }
        model = MaskedOmics(**model_dict)

    elif args.modality == "omics":

        model_dict = {
             "input_dim" : omics_input_dim, "projection_dim": 64, "dropout": args.encoder_dropout, 
             "n_classes" : args.n_classes
        }
        model = MLPOmics(**model_dict)

    elif args.modality == "snn":

        model_dict = {
             "omic_input_dim" : omics_input_dim,'n_classes': args.n_classes
        }
        model = SNNOmics(**model_dict)

    elif args.modality in ["abmil_wsi", "abmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion, 'n_classes': args.n_classes
        }

        model = ABMIL(**model_dict)


    # unimodal and multimodal baselines
    elif args.modality in ["deepmisl_wsi", "deepmisl_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion
        }

        model = DeepMISL(**model_dict)

    elif args.modality == "mlp_wsi":
        
        model_dict = {
            "wsi_embedding_dim":args.encoding_dim, "input_dim_omics":omics_input_dim, "dropout":args.encoder_dropout,
            "device": args.device

        }
        model = MLPWSI(**model_dict)

    elif args.modality in ["transmil_wsi", "transmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion, 'n_classes': args.n_classes
        }

        model = TMIL(**model_dict)

    elif args.modality in ["Frmil_wsi", "Frmil_wsi_pathways"]:

        model_dict = {
            "device" : args.device, "df_comp" : args.composition_df, "omic_input_dim" : omics_input_dim,
            "dim_per_path_1" : args.encoding_layer_1_dim, "dim_per_path_2" : args.encoding_layer_2_dim,
            "fusion":args.fusion, 'n_classes': args.n_classes
        }

        model = FRMIL(**model_dict)

    elif args.modality == "coattn":

        model_dict = {'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCATPathways(**model_dict)

    elif args.modality == "coattn_motcat":

        model_dict = {
            'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes,
            "ot_reg":0.1, "ot_tau":0.5, "ot_impl":"pot-uot-l2"
        }
        model = MCATPathwaysMotCat(**model_dict)

    # survpath 
    elif args.modality == "survpath":

        model_dict = {'omic_sizes': args.omic_sizes, 'num_classes': args.n_classes}

        if args.use_nystrom:
            model = SurvPath_with_nystrom(**model_dict)
        else:
            model = SurvPath(**model_dict)
    elif args.modality == "coattn_cmta":
        from models.cmta.network import CMTA
        # from models.cmta.engine import Engine

        model_dict = {
            "omic_sizes": args.omic_sizes,
            "n_classes": args.n_classes,
            "fusion": args.fusion
        }
        model = CMTA(**model_dict)
    else:
        raise NotImplementedError

    if torch.cuda.is_available():
        model = model.to(torch.device('cuda'))

    print('Done!')
    _print_network(args.results_dir, model)

    return model

def _init_loaders(args, train_split, val_split):
    r"""
    Init dataloaders for the train and val datasets 

    Args:
        - args : argspace.Namespace 
        - train_split : SurvivalDataset 
        - val_split : SurvivalDataset 
    
    Returns:
        - train_loader : Pytorch Dataloader 
        - val_loader : Pytorch Dataloader

    """

    print('\nInit Loaders...', end=' ')
    if train_split:
        train_loader = _get_split_loader(args, train_split, training=True, testing=False, weighted=args.weighted_sample, batch_size=args.batch_size)
    else:
        train_loader = None

    if val_split:
        val_loader = _get_split_loader(args, val_split,  testing=False, batch_size=1)
    else:
        val_loader = None
    print('Done!')

    return train_loader,val_loader

def _extract_survival_metadata(train_loader, val_loader):
    r"""
    Extract censorship and survival times from the train and val loader and combine to get numbers for the fold
    We need to do this for train and val combined because when evaulating survival metrics, the function needs to know the 
    distirbution of censorhsip and survival times for the trainig data
    
    Args:
        - train_loader : Pytorch Dataloader
        - val_loader : Pytorch Dataloader
    
    Returns:
        - all_survival : np.array
    
    """

    all_censorships = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.censorship_var].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.censorship_var].to_numpy()],
        axis=0)

    all_event_times = np.concatenate(
        [train_loader.dataset.metadata[train_loader.dataset.label_col].to_numpy(),
        val_loader.dataset.metadata[val_loader.dataset.label_col].to_numpy()],
        axis=0)

    all_survival = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    return all_survival

def _unpack_data(modality, device, data):
    r"""
    Depending on the model type, unpack the data and put it on the correct device
    
    Args:
        - modality : String 
        - device : torch.device 
        - data : tuple 
    
    Returns:
        - data_WSI : torch.Tensor
        - mask : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - data_omics : torch.Tensor
        - clinical_data_list : list
        - mask : torch.Tensor
    
    """
    
    if modality in ["mlp_per_path", "omics", "snn"]:
        data_WSI1 = data[0]
        data_WSI2 = data[5].to(device)
        data_WSI3 = data[6].to(device)
        data_omics = data[1].to(device)
        mask = None

        # y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        y_disc, clinical_data_list, case_id, idx = data[2], data[3], data[7], data[8]
    
    elif modality in ["mlp_per_path_wsi", "abmil_wsi", "abmil_wsi_pathways", "deepmisl_wsi", "deepmisl_wsi_pathways", "mlp_wsi", "transmil_wsi", "transmil_wsi_pathways","Frmil_wsi", "Frmil_wsi_pathways"]:
        data_WSI1 = data[0].to(device)
        data_WSI2 = data[5].to(device)
        data_WSI3 = data[6].to(device)
        data_omics = data[1].to(device)
        
        if data[4][0,0] == 1:
            mask = None
        else:
            mask = data[4].to(device)

        # y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        y_disc, clinical_data_list, case_id, idx = data[2], data[3], data[7], data[8]

    elif modality in ["coattn", "coattn_motcat","coattn_cmta"]:
        
        # data_WSI = data[0].to(device)
        # data_omic1 = data[1].type(torch.FloatTensor).to(device)
        # data_omic2 = data[2].type(torch.FloatTensor).to(device)
        # data_omic3 = data[3].type(torch.FloatTensor).to(device)
        # data_omic4 = data[4].type(torch.FloatTensor).to(device)
        # data_omic5 = data[5].type(torch.FloatTensor).to(device)
        # data_omic6 = data[6].type(torch.FloatTensor).to(device)
        # data_omics = [data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6]

        # y_disc, event_time, censor, clinical_data_list, mask = data[7], data[8], data[9], data[10], data[11]
        # mask = mask.to(device)
        data_WSI1 = data[0].to(device)
        data_WSI2 = data[5].to(device)
        data_WSI3 = data[6].to(device)

        data_omics = []
        for item in data[1][0]:
            data_omics.append(item.to(device))
        
        if data[4][0,0] == 1:
            mask = None
        else:
            mask = data[4].to(device)

        y_disc, clinical_data_list, case_id, idx = data[2], data[3], data[7], data[8]

    elif modality in ["survpath"]:

        data_WSI1 = data[0].to(device)
        data_WSI2 = data[5].to(device)
        data_WSI3 = data[6].to(device)

        data_omics = []
        for item in data[1][0]:
            data_omics.append(item.to(device))
        
        # if data[6][0,0] == 1:
        if data[4][0,0] == 1:
            mask = None
        else:
            # mask = data[6].to(device)
            mask = data[4].to(device)

        # y_disc, event_time, censor, clinical_data_list = data[2], data[3], data[4], data[5]
        y_disc, clinical_data_list, case_id, idx = data[2], data[3], data[7], data[8]
        
    else:
        raise ValueError('Unsupported modality:', modality)
    
    # y_disc, event_time, censor = y_disc.to(device), event_time.to(device), censor.to(device)
    y_disc = y_disc.to(device)

    # return data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask
    return data_WSI1, y_disc, data_omics, clinical_data_list, mask, data_WSI2, data_WSI3, case_id, idx

def _process_data_and_forward(model, modality, device, data):
    r"""
    Depeding on the modality, process the input data and do a forward pass on the model 
    
    Args:
        - model : Pytorch model
        - modality : String
        - device : torch.device
        - data : tuple
    
    Returns:
        - out : torch.Tensor
        - y_disc : torch.Tensor
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - clinical_data_list : List
    
    """
    # data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
    data_WSI1, y_disc, data_omics, clinical_data_list, mask, data_WSI2, data_WSI3, case_id, idx = _unpack_data(modality, device, data)

    if modality in ["coattn", "coattn_motcat","coattn_cmta"]:  
        
        # out = model(
        #     x_path=data_WSI, 
        #     x_omic1=data_omics[0], 
        #     x_omic2=data_omics[1], 
        #     x_omic3=data_omics[2], 
        #     x_omic4=data_omics[3], 
        #     x_omic5=data_omics[4], 
        #     x_omic6=data_omics[5]
        #     )  
        input_args = {"x_path1": data_WSI1.to(device)}
        input_args.update({"x_path2": data_WSI2.to(device)})
        input_args.update({"x_path3": data_WSI3.to(device)})
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
        input_args["return_attn"] = False
        # input_args["return_attn"] = True
        out, Y_hat, _ = model(**input_args)

    elif modality == 'survpath':
        input_args = {"x_path1": data_WSI1.to(device)}
        input_args.update({"x_path2": data_WSI2.to(device)})
        input_args.update({"x_path3": data_WSI3.to(device)})
        for i in range(len(data_omics)):
            input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
        input_args["return_attn"] = False
        # input_args["case_id"] = case_id
        # input_args["return_attn"] = True
        # out, Y_hat, _ = model(**input_args)
        out, Y_hat, _, logits2 = model(**input_args)
        
    else:
        out, Y_hat, _ = model(
            data_omics = data_omics, 
            data_WSI = data_WSI1, 
            mask = mask
            )
        
    if len(out.shape) == 1:
            out = out.unsqueeze(0)
    # return out, y_disc, event_time, censor, clinical_data_list
    return out, y_disc, clinical_data_list, Y_hat, case_id, idx
    # return out, y_disc, clinical_data_list, Y_hat, case_id, idx, logits2


def _calculate_risk(h):
    r"""
    Take the logits of the model and calculate the risk for the patient 
    
    Args: 
        - h : torch.Tensor 
    
    Returns:
        - risk : torch.Tensor 
    
    """
    hazards = torch.sigmoid(h)
    survival = torch.cumprod(1 - hazards, dim=1)
    risk = -torch.sum(survival, dim=1).detach().cpu().numpy()
    return risk, survival.detach().cpu().numpy()

def _update_arrays(all_risk_scores, all_censorships, all_event_times, all_clinical_data, event_time, censor, risk, clinical_data_list):
    r"""
    Update the arrays with new values 
    
    Args:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
        - event_time : torch.Tensor
        - censor : torch.Tensor
        - risk : torch.Tensor
        - clinical_data_list : List
    
    Returns:
        - all_risk_scores : List
        - all_censorships : List
        - all_event_times : List
        - all_clinical_data : List
    
    """
    all_risk_scores.append(risk)
    all_censorships.append(censor.detach().cpu().numpy())
    all_event_times.append(event_time.detach().cpu().numpy())
    all_clinical_data.append(clinical_data_list)
    return all_risk_scores, all_censorships, all_event_times, all_clinical_data

def _train_loop_survival(epoch, model, modality, loader, optimizer, scheduler, loss_fn):
    r"""
    Perform one epoch of training 

    Args:
        - epoch : Int
        - model : Pytorch model
        - modality : String 
        - loader : Pytorch dataloader
        - optimizer : torch.optim
        - loss_fn : custom loss function class 
    
    Returns:
        - c_index : Float
        - total_loss : Float 
    
    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    n_classes=2
    acc_logger = Accuracy_Logger(n_classes=n_classes) 
    total_loss = 0.
    train_error = 0.

    y_disc_list = []
    case_id_list = []
    idx_list = []
    # all_risk_scores = []
    # all_censorships = []
    # all_event_times = []
    # all_clinical_data = []

    # 使用混合精度训练
    # scaler = GradScaler()  #addnew
    # one epoch
    print('\n')
    for batch_idx, data in enumerate(loader):

        # print(f"batch_idx: {batch_idx}")
        optimizer.zero_grad()
        # with autocast(): #addnew
        batch_size=data[0].size(0)

        # h, y_disc, event_time, censor, clinical_data_list = _process_data_and_forward(model, modality, device, data)
        h, y_disc, clinical_data_list, Y_hat, case_id, idx = _process_data_and_forward(model, modality, device, data)
        # h, y_disc, clinical_data_list, Y_hat, case_id, idx, logits2 = _process_data_and_forward(model, modality, device, data)

        # loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor) 
        loss = loss_fn(h.view(batch_size,-1), y_disc)
        # loss1 = loss_fn(h.view(batch_size,-1), y_disc)
        # loss2 = loss_fn(logits2.view(batch_size,-1), y_disc)
        # loss = loss1 + loss2
        loss_value = loss.item()
        # loss = loss / y_disc.shape[0]
        
        # risk, _ = _calculate_risk(h)

        # all_risk_scores, all_censorships, all_event_times, all_clinical_data = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
        total_loss += loss_value
             

        loss.backward()
        optimizer.step()
        scheduler.step()
        # torch.cuda.empty_cache()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()

        # scheduler.step()
        acc_logger.log(Y_hat, y_disc)
        # total_loss += loss_value
        error = calculate_error(Y_hat, y_disc)
        train_error += error

        # Append results to lists
        y_disc_list.append(y_disc.cpu().numpy())
        case_id_list.extend(case_id)
        idx_list.extend(idx)

        if (batch_idx % 20) == 0:
            print("batch: {}, loss: {:.3f}".format(batch_idx, loss.item()))
    
    # Convert lists to numpy arrays for easier handling
    y_disc_array = np.concatenate(y_disc_list, axis=0)
    case_id_array = np.array(case_id_list)
    idx_array = np.array(idx_list)

    # Create a DataFrame from the arrays
    # results_df = pd.DataFrame({
    #     'idx': idx_array,
    #     'case_id': case_id_array,
    #     'y_disc': y_disc_array
    # })
    # Save DataFrame to a CSV file
    # results_df.to_csv(os.path.join(r'/home/zwc/project/MuFi-main/', f"epoch_{epoch}_results.csv"), index=False)

    total_loss /= len(loader.dataset)
    train_error /= len(loader.dataset)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, total_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
        # if writer:
        #     writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)
    # all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    # all_censorships = np.concatenate(all_censorships, axis=0)
    # all_event_times = np.concatenate(all_event_times, axis=0)
    # c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]

    # print('Epoch: {}, train_loss: {:.4f}, train_c_index: {:.4f}'.format(epoch, total_loss, c_index))
    print('Epoch: {}, train_loss: {:.4f}'.format(epoch, total_loss))

    # return c_index, total_loss
    return total_loss

def _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores):
    r"""
    Calculate various survival metrics 
    
    Args:
        - loader : Pytorch dataloader
        - dataset_factory : SurvivalDatasetFactory
        - survival_train : np.array
        - all_risk_scores : np.array
        - all_censorships : np.array
        - all_event_times : np.array
        - all_risk_by_bin_scores : np.array
        
    Returns:
        - c_index : Float
        - c_index_ipcw : Float
        - BS : np.array
        - IBS : Float
        - iauc : Float
    
    """
    
    data = loader.dataset.metadata["survival_months_dss"]
    bins_original = dataset_factory.bins
    which_times_to_eval_at = np.array([data.min() + 0.0001, bins_original[1], bins_original[2], data.max() - 0.0001])

    #---> delete the nans and corresponding elements from other arrays 
    original_risk_scores = all_risk_scores
    all_risk_scores = np.delete(all_risk_scores, np.argwhere(np.isnan(original_risk_scores)))
    all_censorships = np.delete(all_censorships, np.argwhere(np.isnan(original_risk_scores)))
    all_event_times = np.delete(all_event_times, np.argwhere(np.isnan(original_risk_scores)))
    #<---

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    c_index_ipcw, BS, IBS, iauc = 0., 0., 0., 0.

    # change the datatype of survival test to calculate metrics 
    try:
        survival_test = Surv.from_arrays(event=(1-all_censorships).astype(bool), time=all_event_times)
    except:
        print("Problem converting survival test datatype, so all metrics 0.")
        return c_index, c_index_ipcw, BS, IBS, iauc
   
    # cindex2 (cindex_ipcw)
    try:
        c_index_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=all_risk_scores)[0]
    except:
        print('An error occured while computing c-index ipcw')
        c_index_ipcw = 0.
    
    # brier score 
    try:
        _, BS = brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing BS')
        BS = 0.
    
    # IBS
    try:
        IBS = integrated_brier_score(survival_train, survival_test, estimate=all_risk_by_bin_scores, times=which_times_to_eval_at)
    except:
        print('An error occured while computing IBS')
        IBS = 0.

    # iauc
    try:
        _, iauc = cumulative_dynamic_auc(survival_train, survival_test, estimate=1-all_risk_by_bin_scores[:, 1:], times=which_times_to_eval_at[1:])
    except:
        print('An error occured while computing iauc')
        iauc = 0.
    
    return c_index, c_index_ipcw, BS, IBS, iauc

def _summary(dataset_factory, model, modality, loader, loss_fn, survival_train=None):
    r"""
    Run a validation loop on the trained model 
    
    Args: 
        - dataset_factory : SurvivalDatasetFactory
        - model : Pytorch model
        - modality : String
        - loader : Pytorch loader
        - loss_fn : custom loss function clas
        - survival_train : np.array
    
    Returns:
        - patient_results : dictionary
        - c_index : Float
        - c_index_ipcw : Float
        - BS : List
        - IBS : Float
        - iauc : Float
        - total_loss : Float

    """
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    n_classes = 2
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    total_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    all_pred = []
    all_label = []
    # all_risk_scores = []
    # all_risk_by_bin_scores = []
    # all_censorships = []
    # all_event_times = []
    # all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    bag_logit, bag_labels=[], []
    
    print('\n')
    with torch.no_grad():
        # for data in loader:
        for batch_idx, data in enumerate(loader):
            batch_size=data[0].size(0)
            if len(data[2]) > 1:
                bag_labels.extend(data[2].tolist())
            else:
                bag_labels.append(data[2].item())
            # data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
            # data_WSI, y_disc, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
            data_WSI1, y_disc, data_omics, clinical_data_list, mask, data_WSI2, data_WSI3, case_id, _ = _unpack_data(modality, device, data)
            if modality in ["coattn", "coattn_motcat","coattn_cmta"]:  
                # h = model(
                #     x_path=data_WSI, 
                #     x_omic1=data_omics[0], 
                #     x_omic2=data_omics[1], 
                #     x_omic3=data_omics[2], 
                #     x_omic4=data_omics[3], 
                #     x_omic5=data_omics[4], 
                #     x_omic6=data_omics[5]
                # )  
                input_args = {"x_path1": data_WSI1.to(device)}
                input_args.update({"x_path2": data_WSI2.to(device)})
                input_args.update({"x_path3": data_WSI3.to(device)})
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False
                h, Y_hat, Y_prob = model(**input_args)
            # elif modality == "survpath":

            #     input_args = {"x_path": data_WSI.to(device)}
            #     for i in range(len(data_omics)):
            #         input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
            #     input_args["return_attn"] = False
                
            #     h = model(**input_args)
            
            elif modality == 'survpath':
                input_args = {"x_path1": data_WSI1.to(device)}
                input_args.update({"x_path2": data_WSI2.to(device)})
                input_args.update({"x_path3": data_WSI3.to(device)})
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False
                # h, Y_hat, Y_prob = model(**input_args)
                h, Y_hat, Y_prob, logits2 = model(**input_args)
            else:
                h, Y_hat, Y_prob = model(
                    data_omics = data_omics, 
                    data_WSI = data_WSI1, 
                    mask = mask
                    )
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            
            acc_logger.log(Y_hat, y_disc)
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = y_disc.item()

            # loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss = loss_fn(h.view(batch_size,-1), y_disc)
            # loss1 = loss_fn(h.view(batch_size,-1), y_disc)
            # loss2 = loss_fn(logits2.view(batch_size,-1), y_disc)
            # loss = loss1 + loss2
            bag_logit.append(torch.softmax(h, dim=-1)[:,1].cpu().squeeze().numpy())

            loss_value = loss.item()
            error = calculate_error(Y_hat, y_disc)
            val_error += error
            all_pred.append(Y_hat.cpu())
            all_label.append(y_disc.cpu())
            # loss = loss / y_disc.shape[0]


            # risk, risk_by_bin = _calculate_risk(h)
            # all_risk_by_bin_scores.append(risk_by_bin)
            # all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
            all_logits.append(h.detach().cpu().numpy())
            total_loss += loss_value
            # all_slide_ids.append(slide_ids.values[count])
            all_slide_ids.append(case_id)
            count += 1

    val_error /= len(loader)
    total_loss /= len(loader)
    all_label = [label.item() for label in all_label] #C
    all_pred = [pred.item() for pred in all_pred] #C
    val_f1 = f1_score(all_label, all_pred, average='macro') #MG
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    # if writer:
    #     writer.add_scalar('val/loss', val_loss, epoch)
    #     writer.add_scalar('val/auc', auc, epoch)
    #     writer.add_scalar('val/error', val_error, epoch)

    print('\nVal Set, val_loss: {:.4f}, val_error: {:.4f}, auc: {:.4f}, f1: {: .4f}'.format(total_loss, val_error, auc, val_f1))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {:.4f}, correct {}/{}'.format(i, acc, correct, count))
    
    accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit)
    total_loss /= len(loader.dataset)
    # all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    # all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    # all_censorships = np.concatenate(all_censorships, axis=0)
    # all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    patient_results = {}
    for i in range(len(all_slide_ids)):
        slide_id = slide_ids.values[i]
        case_id = slide_id[:12]
        patient_results[case_id] = {}
        # patient_results[case_id]["time"] = all_event_times[i]
        # patient_results[case_id]["risk"] = all_risk_scores[i]
        # patient_results[case_id]["censorship"] = all_censorships[i]
        # patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
    
    # c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    # return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss
    return patient_results, total_loss, accuracy, auc_value, precision, recall, fscore


def _get_lr_scheduler(args, optimizer, dataloader):
    scheduler_name = args.lr_scheduler
    warmup_epochs = args.warmup_epochs
    epochs = args.max_epochs if hasattr(args, 'max_epochs') else args.epochs

    if warmup_epochs > 0:
        warmup_steps = warmup_epochs * len(dataloader)
    else:
        warmup_steps = 0
    if scheduler_name=='constant':
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps
        )
    elif scheduler_name=='cosine':
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    elif scheduler_name=='linear':
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(dataloader) * epochs,
        )
    return lr_scheduler

def _step(ckc_metric, train_ckc_metric, cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    r"""
    Trains the model for the set number of epochs and validates it.
    
    Args:
        - cur
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler 
        - train_loader
        - val_loader
        
    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """
    acs,pre,rec,fs,auc,te_auc,te_fs = ckc_metric
    train_acs, train_pre, train_rec,train_fs,train_auc= train_ckc_metric#addnew
    # all_survival = _extract_survival_metadata(train_loader, val_loader)
    optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch,opt_loss = 0, 0, 0, 0,0,0,0
    for epoch in range(args.max_epochs):
        train_loss = _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)
        # _, val_cindex, _, _, _, _, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
        # print('Val loss:', total_loss, ', val_c_index:', val_cindex)
    # save the trained model
    # torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    # results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
        # _, train_loss1, train_accuracy, train_auc_value, train_precision, train_recall, train_fscore = _summary(args.dataset_factory, model, args.modality, train_loader, loss_fn)
        # print('\r Epoch [%d/%d] train loss: %.1E, train loss1: %.1E, train accuracy: %.3f, train auc_value:%.3f, train precision: %.3f, train recall: %.3f, train fscore: %.3f' % 
        #     (epoch+1, args.max_epochs, train_loss, train_loss1, train_accuracy, train_auc_value, train_precision, train_recall, train_fscore))
        patient_results, val_loss, accuracy, auc_value, precision, recall, fscore = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn)
        print('\r Epoch [%d/%d] train loss: %.1E, val loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f' % 
            (epoch+1, args.max_epochs, train_loss, val_loss, accuracy, auc_value, precision, recall, fscore))
    # print('Final Val c-index: {:.4f}'.format(val_cindex))
    # print('Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
    #     val_cindex, 
    #     val_cindex_ipcw,
    #     val_IBS,
    #     val_iauc
    #     ))
        if auc_value > opt_auc and epoch >= args.save_best_model_stage*args.max_epochs:
            opt_loss = val_loss
            print(f'Validation loss decreased ({opt_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
            print(f'Validation AUC increased ({opt_auc:.6f} --> {auc_value:.6f}).  Saving model ...')
            optimal_ac = accuracy
            opt_pre = precision
            opt_re = recall
            opt_fs = fscore
            opt_auc = auc_value
            opt_epoch = epoch


            if not os.path.exists(args.results_dir):
                os.mkdir(args.results_dir)
            if not args.no_log:
                best_pt = {
                    'model': model.state_dict(),
                }
                torch.save(best_pt, os.path.join(args.results_dir, 'fold_{fold}_model_best_auc.pt'.format(fold=cur)))
    
    # save infos
    if not args.no_log:
        best_std = torch.load(os.path.join(args.results_dir, 'fold_{fold}_model_best_auc.pt'.format(fold=cur)))
        info = model.load_state_dict(best_std['model'])
        print(info)
        
    accuracy, auc_value, precision, recall, fscore,test_loss_log,test_patient_results = test(model, args.modality, val_loader, loss_fn)
    print('Val loss: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(test_loss_log, auc_value, fscore))
    train_accuracy, train_auc_value, train_precision, train_recall, train_fscore,train_test_loss_log,train_patient_results = test(model, args.modality, train_loader, loss_fn) #addnew
    print('Train loss: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(train_test_loss_log, train_auc_value, train_fscore))
    save_pkl(os.path.join(args.results_dir, f'split_{cur}_train_results.pkl'), train_patient_results)
    save_pkl(os.path.join(args.results_dir, f'split_{cur}_val_results.pkl'), test_patient_results)

    if not args.no_log:
        print('\n Optimal accuracy: %.3f ,Optimal auc: %.3f,Optimal precision: %.3f,Optimal recall: %.3f,Optimal fscore: %.3f' % (optimal_ac,opt_auc,opt_pre,opt_re,opt_fs))
    acs.append(accuracy)
    pre.append(precision)
    rec.append(recall)
    fs.append(fscore)
    auc.append(auc_value)
    train_acs.append(train_accuracy) #addnew
    train_pre.append(train_precision)#addnew
    train_rec.append(train_recall)#addnew
    train_fs.append(train_fscore)#addnew
    train_auc.append(train_auc_value)#addnew
    # if args.always_test:
    #     te_auc.append(opt_te_auc)
    #     te_fs.append(opt_te_fs)
         
    return [acs,pre,rec,fs,auc,te_auc,te_fs], [train_acs, train_pre, train_rec,train_fs,train_auc]


 
    # return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)

def _step_test(ckc_metric, cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
# def _step_test(ckc_metric, train_ckc_metric, cur, args, loss_fn, model, optimizer, scheduler, train_loader, val_loader):
    r"""
    Trains the model for the set number of epochs and validates it.
    
    Args:
        - cur
        - args
        - loss_fn
        - model
        - optimizer
        - lr scheduler 
        - train_loader
        - val_loader
        
    Returns:
        - results_dict : dictionary
        - val_cindex : Float
        - val_cindex_ipcw  : Float
        - val_BS : List
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """
    acs,pre,rec,fs,auc,te_auc,te_fs = ckc_metric
    # train_acs, train_pre, train_rec,train_fs,train_auc= train_ckc_metric#addnew
    # all_survival = _extract_survival_metadata(train_loader, val_loader)
    # optimal_ac, opt_pre, opt_re, opt_fs, opt_auc,opt_epoch,opt_loss = 0, 0, 0, 0,0,0,0
    # for epoch in range(args.max_epochs):
        # train_loss = _train_loop_survival(epoch, model, args.modality, train_loader, optimizer, scheduler, loss_fn)
        # _, val_cindex, _, _, _, _, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
        # print('Val loss:', total_loss, ', val_c_index:', val_cindex)
    # save the trained model
    # torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    
    # results_dict, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn, all_survival)
        # _, train_loss1, train_accuracy, train_auc_value, train_precision, train_recall, train_fscore = _summary(args.dataset_factory, model, args.modality, train_loader, loss_fn)
        # print('\r Epoch [%d/%d] train loss: %.1E, train loss1: %.1E, train accuracy: %.3f, train auc_value:%.3f, train precision: %.3f, train recall: %.3f, train fscore: %.3f' % 
        #     (epoch+1, args.max_epochs, train_loss, train_loss1, train_accuracy, train_auc_value, train_precision, train_recall, train_fscore))
        # patient_results, val_loss, accuracy, auc_value, precision, recall, fscore = _summary(args.dataset_factory, model, args.modality, val_loader, loss_fn)
        # print('\r Epoch [%d/%d] train loss: %.1E, val loss: %.1E, accuracy: %.3f, auc_value:%.3f, precision: %.3f, recall: %.3f, fscore: %.3f' % 
            # (epoch+1, args.max_epochs, train_loss, val_loss, accuracy, auc_value, precision, recall, fscore))
    # print('Final Val c-index: {:.4f}'.format(val_cindex))
    # print('Final Val c-index: {:.4f} | Final Val c-index2: {:.4f} | Final Val IBS: {:.4f} | Final Val iauc: {:.4f}'.format(
    #     val_cindex, 
    #     val_cindex_ipcw,
    #     val_IBS,
    #     val_iauc
    #     ))
        # if auc_value > opt_auc and epoch >= args.save_best_model_stage*args.max_epochs:
        #     opt_loss = val_loss
        #     print(f'Validation loss decreased ({opt_loss:.6f} --> {val_loss:.6f}).  Saving model ...')
        #     print(f'Validation AUC increased ({opt_auc:.6f} --> {auc_value:.6f}).  Saving model ...')
        #     optimal_ac = accuracy
        #     opt_pre = precision
        #     opt_re = recall
        #     opt_fs = fscore
        #     opt_auc = auc_value
        #     opt_epoch = epoch


            # if not os.path.exists(args.results_dir):
            #     os.mkdir(args.results_dir)
            # if not args.no_log:
            #     best_pt = {
            #         'model': model.state_dict(),
            #     }
            #     torch.save(best_pt, os.path.join(args.results_dir, 'fold_{fold}_model_best_auc.pt'.format(fold=cur)))
    
    # save infos
    # if not args.no_log:
    best_std = torch.load(os.path.join(args.model_path, 'fold_{fold}_model_best_auc.pt'.format(fold=cur)))
    info = model.load_state_dict(best_std['model'])
    print(info)
    
    # accuracy, auc_value, precision, recall, fscore,test_loss_log,test_patient_results = test(model, args.modality, val_loader, loss_fn)
    accuracy, auc_value, precision, recall, fscore,test_loss_log,test_patient_results,attention_result,df = _test_(model, args.modality, train_loader, loss_fn) #val_loader / train_loader
    print('Val loss: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(test_loss_log, auc_value, fscore))
    # train_accuracy, train_auc_value, train_precision, train_recall, train_fscore,train_test_loss_log,train_patient_results = test(model, args.modality, train_loader, loss_fn) #addnew
    # print('Train loss: {:.4f}, ROC AUC: {:.4f}, F1: {:.4f}'.format(train_test_loss_log, train_auc_value, train_fscore))
    # save_pkl(os.path.join(args.results_dir, f'split_{cur}_train_results.pkl'), train_patient_results)
    save_pkl(os.path.join(args.results_dir, f'split_{cur}_test_results.pkl'), test_patient_results)
    torch.save(attention_result, os.path.join(args.results_dir, 'attention.pt'))#addnew
    
    for slide_id, data in attention_result.items():
        # Sample data from attention_result
        signature = attention_result[slide_id]['signature']
        # relative_ranking = attention_result[slide_id]['relative_ranking']
        relative_ranking = np.abs(attention_result[slide_id]['relative_ranking'])

        # sorted_indices = np.argsort(relative_ranking)[::-1]
        sorted_indices = np.argsort(relative_ranking)
        signature = np.array(signature)[sorted_indices]  
        relative_ranking = relative_ranking[sorted_indices]
        # Number of bars
        num_bars = len(signature)
        norm = plt.Normalize(relative_ranking.min(), relative_ranking.max())
        # Create alternating colors (purple and cyan)
        cmap = plt.cm.plasma_r #magma_r

        colors = cmap(norm(relative_ranking))
        # Create the bar chart
        fig, ax = plt.subplots()

        bars = ax.barh(signature, relative_ranking, color=colors)

        # Add labels and title
        # ax.set_xlabel('Relative Ranking')
        # ax.set_ylabel('Signature')
        # ax.set_title('Signature vs. Relative Ranking')
        margin = 0.05 * (relative_ranking.max() - relative_ranking.min())
        ax.set_xlim(left=relative_ranking.min() - margin, right=relative_ranking.max() + margin)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_ticks([norm.vmin, norm.vmax]) 
        cbar.set_ticklabels(['Low', 'High'])
        # cbar.set_label('Relative Ranking')
        ax.locator_params(axis='x', nbins=5)
        output_file = os.path.join(args.results_dir, f'{slide_id}_signature.png')
        plt.savefig(output_file, format='png', bbox_inches='tight')
        # Show the plot
        plt.close(fig)
        #slide_path = os.path.join(args.wsi_folder, slide_id + '.ndpi') #.tiff
        
        # 构建文件路径
        tiff_path = os.path.join(args.wsi_folder, slide_id + '.tiff')
        ndpi_path = os.path.join(args.wsi_folder, slide_id + '.ndpi')
        if os.path.exists(tiff_path):
            slide_path = tiff_path
        else:
            slide_path = ndpi_path
        wsi_object = initialize_wsi(slide_path, seg_mask_path=None, seg_params=None, filter_params=None)
        coords = data['coords']
        # attention_scores = data['attention_scores']
        for i, attention_scores in enumerate(data['attention_scores']):
            heatmap = drawHeatmap(attention_scores, coords, slide_path, wsi_object=wsi_object, cmap= 'rainbow', alpha=0.4, use_holes=True, binarize=False, vis_level= 3, blank_canvas=False, #Oranges,Purples,Reds,gist_rainbow,cubehelix,gnuplot,gnuplot2
                                thresh=-1, patch_size = 224, convert_to_percentiles=True) #vis_level = -1, cmap= 'jet','inferno','viridis,'plasma','magma','cividis','Greys','Blues','coolwarm','bwr','seismic','PiYG','RdBu','twilight','twilight_shifted','rainbow','hsv','gist_rainbow','nipy_spectral','flag','PuOr','Spectral','RdYlBu'
            
            # heatmap.save(os.path.join(args.results_dir, '{}_blockmap.png'.format(slide_id)))
            heatmap.save(os.path.join(args.results_dir, f'{slide_id}_attention_{i}_blockmap.png'))
            del heatmap

            y_true = df[df['slide_id'] == slide_id]['Y'].values[0]
            y_pred = df[df['slide_id'] == slide_id]['Y_hat'].values[0]

            samples = [
                    {
                        "name": "topk_high_attention",
                        "sample": True,
                        "seed": 1,
                        "k": 15,
                        "mode": "topk"
                    }
                ]
            # Sample patches
            for sample in samples:
                if sample['sample']:
                    tag = f"label_{y_true}_pred_{y_pred}"
                    
                    sample_save_dir = os.path.join(args.results_dir, 'sampled_patches', str(tag), sample['name'])
                    os.makedirs(sample_save_dir, exist_ok=True)
                    # print('sampling {}'.format(sample['name']))
                    print(f'sampling {sample["name"]} for attention score {i}')

                    sample_results = sample_rois(attention_scores, coords, k=sample['k'], mode=sample['mode'], seed=sample['seed'],
                                                    score_start=sample.get('score_start', 0), score_end=sample.get('score_end', 1))
                    for idx, (s_coord, s_score) in enumerate(zip(sample_results['sampled_coords'], sample_results['sampled_scores'])):
                        print('coord: {} score: {:.3f}'.format(s_coord, s_score))
                        # Scale the coordinates by a factor of 4
                        scaled_coord = (int(s_coord[0] * 2), int(s_coord[1] * 2)) #恢复到最大level == 0

                        patch = wsi_object.wsi.read_region(scaled_coord, 1, (224, 224)).convert('RGB') #保存20倍放大 #1 / 2
                        #patch = wsi_object.wsi.read_region(tuple(s_coord), 2, (256, 256)).convert('RGB')
                        # patch.save(os.path.join(sample_save_dir, '{}_{}_x_{}_y_{}_a_{:.3f}.png'.format(idx, slide_id, s_coord[0], s_coord[1], s_score)))
                        patch.save(os.path.join(sample_save_dir, f'{idx}_{slide_id}_attention_{i}_x_{s_coord[0]}_y_{s_coord[1]}_a_{s_score:.3f}.png'))
    # if not args.no_log:
        # print('\n Optimal accuracy: %.3f ,Optimal auc: %.3f,Optimal precision: %.3f,Optimal recall: %.3f,Optimal fscore: %.3f' % (optimal_ac,opt_auc,opt_pre,opt_re,opt_fs))
    acs.append(accuracy)
    pre.append(precision)
    rec.append(recall)
    fs.append(fscore)
    auc.append(auc_value)
    # train_acs.append(train_accuracy) #addnew
    # train_pre.append(train_precision)#addnew
    # train_rec.append(train_recall)#addnew
    # train_fs.append(train_fscore)#addnew
    # train_auc.append(train_auc_value)#addnew
    # if args.always_test:
    #     te_auc.append(opt_te_auc)
    #     te_fs.append(opt_te_fs)
         
    # return [acs,pre,rec,fs,auc,te_auc,te_fs], [train_acs, train_pre, train_rec,train_fs,train_auc]
    return [acs,pre,rec,fs,auc,te_auc,te_fs]


 
    # return results_dict, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)
def _train_val(datasets, cur, args,ckc_metric, train_ckc_metric):
    """   
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int 
        - args : argspace.Namespace 
    
    Returns:
        - results_dict : dict
        - val_cindex : Float
        - val_cindex2 : Float
        - val_BS : Float
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    #----> gets splits and summarize
    train_split, val_split = _get_splits(datasets, cur, args)
    
    #----> init loss function
    loss_fn = _init_loss_function(args)

    #----> init model
    model = _init_model(args)
    
    #---> init optimizer
    optimizer = _init_optim(args, model)

    #---> init loaders
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # lr scheduler 
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    #---> do train val
    # acs, pre, rec,fs,auc,te_auc,te_fs=[],[],[],[],[],[],[]
    # ckc_metric = [acs, pre, rec,fs,auc,te_auc,te_fs]
    # train_acs, train_pre, train_rec,train_fs,train_auc=[],[],[],[],[] #addnew
    # train_ckc_metric = [train_acs, train_pre, train_rec,train_fs,train_auc]  #addnew
    # results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss) = _step(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)
    ckc_metric, train_ckc_metric = _step(ckc_metric, train_ckc_metric, cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)

    # return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss)
    return ckc_metric, train_ckc_metric

def _test(datasets, cur, args,ckc_metric):
# def _test(datasets, cur, args,ckc_metric, train_ckc_metric):
    """   
    Performs train val test for the fold over number of epochs

    Args:
        - datasets : tuple
        - cur : Int 
        - args : argspace.Namespace 
    
    Returns:
        - results_dict : dict
        - val_cindex : Float
        - val_cindex2 : Float
        - val_BS : Float
        - val_IBS : Float
        - val_iauc : Float
        - total_loss : Float
    """

    #----> gets splits and summarize
    train_split, val_split = _get_splits(datasets, cur, args)
    
    #----> init loss function
    loss_fn = _init_loss_function(args)

    #----> init model
    model = _init_model(args)
    
    #---> init optimizer
    optimizer = _init_optim(args, model)

    #---> init loaders
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # lr scheduler 
    lr_scheduler = _get_lr_scheduler(args, optimizer, train_loader)

    #---> do train val
    # acs, pre, rec,fs,auc,te_auc,te_fs=[],[],[],[],[],[],[]
    # ckc_metric = [acs, pre, rec,fs,auc,te_auc,te_fs]
    # train_acs, train_pre, train_rec,train_fs,train_auc=[],[],[],[],[] #addnew
    # train_ckc_metric = [train_acs, train_pre, train_rec,train_fs,train_auc]  #addnew
    # results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss) = _step(cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)
    ckc_metric = _step_test(ckc_metric, cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)

    # ckc_metric, train_ckc_metric = _step_test(ckc_metric, train_ckc_metric, cur, args, loss_fn, model, optimizer, lr_scheduler, train_loader, val_loader)

    # return results_dict, (val_cindex, val_cindex2, val_BS, val_IBS, val_iauc, total_loss)
    return ckc_metric
    # return ckc_metric, train_ckc_metric
# def five_scores(bag_labels, bag_predictions,sub_typing=False):
def five_scores(bag_labels, bag_predictions):
    fpr, tpr, threshold = roc_curve(bag_labels, bag_predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    # threshold_optimal=0.5
    auc_value = roc_auc_score(bag_labels, bag_predictions)
    this_class_label = np.array(bag_predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    bag_predictions = this_class_label
    # avg = 'macro' if sub_typing else 'binary'
    avg = 'macro'
    precision, recall, fscore, _ = precision_recall_fscore_support(bag_labels, bag_predictions, average=avg)
    accuracy = accuracy_score(bag_labels, bag_predictions)
    # return accuracy, auc_value, precision, recall, fscore
    return accuracy, auc_value, precision, recall, fscore,bag_predictions

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def test(model, modality, loader, loss_fn):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    # all_risk_scores = []
    # all_risk_by_bin_scores = []
    # all_censorships = []
    # all_event_times = []
    # all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    bag_logit, bag_labels=[], []
    # patient_results = {} #addnew

    with torch.no_grad():
        for data in loader:
            batch_size=data[0].size(0)
            # case_id = data[7]
            if len(data[2]) > 1:
                bag_labels.extend(data[2].tolist())
            else:
                bag_labels.append(data[2].item())
            # data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
            # data_WSI, y_disc, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
            data_WSI1, y_disc, data_omics, clinical_data_list, mask, data_WSI2, data_WSI3, case_id, _ = _unpack_data(modality, device, data)
            if modality in ["coattn", "coattn_motcat","coattn_cmta"]:  
                # h = model(
                #     x_path=data_WSI, 
                #     x_omic1=data_omics[0], 
                #     x_omic2=data_omics[1], 
                #     x_omic3=data_omics[2], 
                #     x_omic4=data_omics[3], 
                #     x_omic5=data_omics[4], 
                #     x_omic6=data_omics[5]
                # )  
                input_args = {"x_path1": data_WSI1.to(device)}
                input_args.update({"x_path2": data_WSI2.to(device)})
                input_args.update({"x_path3": data_WSI3.to(device)})
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False
                h,_ ,_ = model(**input_args)
            # elif modality == "survpath":

            #     input_args = {"x_path": data_WSI.to(device)}
            #     for i in range(len(data_omics)):
            #         input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
            #     input_args["return_attn"] = False
                
            #     h = model(**input_args)
            
            elif modality == 'survpath':
                input_args = {"x_path1": data_WSI1.to(device)}
                input_args.update({"x_path2": data_WSI2.to(device)})
                input_args.update({"x_path3": data_WSI3.to(device)})
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False
                # h,_ ,_ = model(**input_args)
                h,_ ,_, logits2 = model(**input_args)
            
            else:
                h, Y_hat, _ = model(
                    data_omics = data_omics, 
                    data_WSI = data_WSI1, 
                    mask = mask
                    )
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
                
            # loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss = loss_fn(h.view(batch_size,-1), y_disc)
            # loss1 = loss_fn(h.view(batch_size,-1), y_disc)
            # loss2 = loss_fn(logits2.view(batch_size,-1), y_disc)
            # loss = loss1 + loss2
            bag_logit.append(torch.softmax(h, dim=-1)[:,1].cpu().squeeze().numpy())

            loss_value = loss.item()
            loss = loss / y_disc.shape[0]


            # risk, risk_by_bin = _calculate_risk(h)
            # all_risk_by_bin_scores.append(risk_by_bin)
            # all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
            # all_logits.append(h.detach().cpu().numpy())
            all_logits.append(torch.softmax(h, dim=-1).cpu().numpy())
            total_loss += loss_value
            # all_slide_ids.append(slide_ids.values[count])
            all_slide_ids.append(case_id[0])
            count += 1

    # accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit, not args.datasets.lower() == 'camelyon16')
    accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit)
    total_loss /= len(loader.dataset)
    # all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    # all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    # all_censorships = np.concatenate(all_censorships, axis=0)
    # all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    patient_results = {}
    for i in range(len(all_slide_ids)):
        # slide_id = slide_ids.values[i]
        slide_id = all_slide_ids[i]
        # case_id = slide_id[:12]
        case_id = slide_id
        patient_results[case_id] = {}
        # patient_results[case_id]["time"] = all_event_times[i]
        # patient_results[case_id]["risk"] = all_risk_scores[i]
        # patient_results[case_id]["censorship"] = all_censorships[i]
        # patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
        patient_results[case_id]["label"] = bag_labels[i]
    
    # c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)

    # return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss
    return  accuracy, auc_value, precision, recall, fscore, total_loss, patient_results

def _test_(model, modality, loader, loss_fn):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = 0.

    # all_risk_scores = []
    # all_risk_by_bin_scores = []
    # all_censorships = []
    # all_event_times = []
    # all_clinical_data = []
    all_logits = []
    all_slide_ids = []

    slide_ids = loader.dataset.metadata['slide_id']
    count = 0
    bag_logit, bag_labels=[], []
    all_probs = []
    # patient_results = {} #addnew
    attention_result = {} #addnew
    with torch.no_grad():
        for data in loader:
            batch_size=data[0].size(0)
            coords = data[9]
            # case_id = data[7]
            if len(data[2]) > 1:
                bag_labels.extend(data[2].tolist())
            else:
                bag_labels.append(data[2].item())
            # data_WSI, mask, y_disc, event_time, censor, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
            # data_WSI, y_disc, data_omics, clinical_data_list, mask = _unpack_data(modality, device, data)
            data_WSI1, y_disc, data_omics, clinical_data_list, mask, data_WSI2, data_WSI3, case_id, _ = _unpack_data(modality, device, data)
            if modality in ["coattn", "coattn_motcat","coattn_cmta"]:  
                # h = model(
                #     x_path=data_WSI, 
                #     x_omic1=data_omics[0], 
                #     x_omic2=data_omics[1], 
                #     x_omic3=data_omics[2], 
                #     x_omic4=data_omics[3], 
                #     x_omic5=data_omics[4], 
                #     x_omic6=data_omics[5]
                # )  
                input_args = {"x_path1": data_WSI1.to(device)}
                input_args.update({"x_path2": data_WSI2.to(device)})
                input_args.update({"x_path3": data_WSI3.to(device)})
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                input_args["return_attn"] = False
                h,_ ,_ = model(**input_args)
            # elif modality == "survpath":

            #     input_args = {"x_path": data_WSI.to(device)}
            #     for i in range(len(data_omics)):
            #         input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
            #     input_args["return_attn"] = False
                
            #     h = model(**input_args)
            
            elif modality == 'survpath':
                input_args = {"x_path1": data_WSI1.to(device)}
                input_args.update({"x_path2": data_WSI2.to(device)})
                input_args.update({"x_path3": data_WSI3.to(device)})
                for i in range(len(data_omics)):
                    input_args['x_omic%s' % str(i+1)] = data_omics[i].type(torch.FloatTensor).to(device)
                # input_args["return_attn"] = False
                input_args["return_attn"] = True
                # h,_ ,_ = model(**input_args)
                # h,_ ,_, logits2 = model(**input_args)
                h,attn_pathways, cross_attn_pathways, cross_attn_histology = model(**input_args)
            else:
                h, Y_hat, _ = model(
                    data_omics = data_omics, 
                    data_WSI = data_WSI1, 
                    mask = mask
                    )
                    
            if len(h.shape) == 1:
                h = h.unsqueeze(0)
            
            # Existing code
            attention_scores = cross_attn_pathways.cpu().numpy()  # Shape: (202, 2345)
            coords_array = coords[0].cpu().numpy()  # No need to squeeze

            # Compute row sums
            row_sums = attention_scores.sum(axis=1)  # Shape: (202,)

            # Get indices of top 6 rows
            top_indices = np.argsort(row_sums)[-10:][::-1]  # Top 6 indices with highest sums
            top_values = row_sums[top_indices]
            relative_ranking = top_values / row_sums.sum()

            # Extract top attention scores and corresponding coordinates
            top_attention_scores = attention_scores[top_indices]  # Shape: (6, 2345)
            top_signature = loader.dataset.signature.columns[top_indices]
            # Save the results
            attention_result[case_id[0]] = {
                'attention_scores': top_attention_scores,
                'coords': coords_array,
                'signature': top_signature,
                'relative_ranking': relative_ranking
            }

            # attention_result[case_id] = {'attention_scores': cross_attn_pathways.cpu().numpy(), 'coords': np.squeeze(coords.cpu().numpy(), axis=0)}
            # loss = loss_fn(h=h, y=y_disc, t=event_time, c=censor)
            loss = loss_fn(h.view(batch_size,-1), y_disc)
            # loss1 = loss_fn(h.view(batch_size,-1), y_disc)
            # loss2 = loss_fn(logits2.view(batch_size,-1), y_disc)
            # loss = loss1 + loss2
            bag_logit.append(torch.softmax(h, dim=-1)[:,1].cpu().squeeze().numpy())
            all_probs.append(torch.softmax(h, dim=-1).cpu().squeeze().numpy())

            loss_value = loss.item()
            loss = loss / y_disc.shape[0]


            # risk, risk_by_bin = _calculate_risk(h)
            # all_risk_by_bin_scores.append(risk_by_bin)
            # all_risk_scores, all_censorships, all_event_times, clinical_data_list = _update_arrays(all_risk_scores, all_censorships, all_event_times,all_clinical_data, event_time, censor, risk, clinical_data_list)
            # all_logits.append(h.detach().cpu().numpy())
            all_logits.append(torch.softmax(h, dim=-1).cpu().numpy())
            total_loss += loss_value
            # all_slide_ids.append(slide_ids.values[count])
            all_slide_ids.append(case_id[0])
            count += 1

    # accuracy, auc_value, precision, recall, fscore = five_scores(bag_labels, bag_logit, not args.datasets.lower() == 'camelyon16')
    accuracy, auc_value, precision, recall, fscore,bag_predictions = five_scores(bag_labels, bag_logit)
    total_loss /= len(loader.dataset)
    # all_risk_scores = np.concatenate(all_risk_scores, axis=0)
    # all_risk_by_bin_scores = np.concatenate(all_risk_by_bin_scores, axis=0)
    # all_censorships = np.concatenate(all_censorships, axis=0)
    # all_event_times = np.concatenate(all_event_times, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)
    
    patient_results = {}
    for i in range(len(all_slide_ids)):
        # slide_id = slide_ids.values[i]
        slide_id = all_slide_ids[i]
        # case_id = slide_id[:12]
        case_id = slide_id
        patient_results[case_id] = {}
        # patient_results[case_id]["time"] = all_event_times[i]
        # patient_results[case_id]["risk"] = all_risk_scores[i]
        # patient_results[case_id]["censorship"] = all_censorships[i]
        # patient_results[case_id]["clinical"] = all_clinical_data[i]
        patient_results[case_id]["logits"] = all_logits[i]
        patient_results[case_id]["label"] = bag_labels[i]

    
    # c_index, c_index2, BS, IBS, iauc = _calculate_metrics(loader, dataset_factory, survival_train, all_risk_scores, all_censorships, all_event_times, all_risk_by_bin_scores)
    results_dict = {'slide_id': all_slide_ids, 'Y': bag_labels, 'Y_hat': [int(x) for x in bag_predictions.tolist()]}
    all_probs_np = np.array(all_probs)
    for c in range(2):
        results_dict.update({'p_{}'.format(c): all_probs_np[:,c]})
    df = pd.DataFrame(results_dict)
    # return patient_results, c_index, c_index2, BS, IBS, iauc, total_loss
    return  accuracy, auc_value, precision, recall, fscore, total_loss, patient_results, attention_result, df

class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count

def calculate_error(Y_hat, Y):
	error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

	return error