#----> pytorch imports
import torch

#----> general imports
import pandas as pd
import numpy as np
import pdb
import os
from timeit import default_timer as timer
from datasets.dataset_survival import SurvivalDatasetFactory
from utils.core_utils import _train_val
from utils.file_utils import _save_pkl
from utils.general_utils import _get_start_end, _prepare_for_experiment

from utils.process_args import _process_args

def main(args):

    #----> prep for 5 fold cv study
    folds = _get_start_end(args)
    
    #----> storing the val and test cindex for 5 fold cv
    # all_val_cindex = []
    # all_val_cindex_ipcw = []
    # all_val_BS = []
    # all_val_IBS = []
    # all_val_iauc = []
    # all_val_loss = []

    acs, pre, rec,fs,auc,te_auc,te_fs=[],[],[],[],[],[],[]
    ckc_metric = [acs, pre, rec,fs,auc,te_auc,te_fs]
    train_acs, train_pre, train_rec,train_fs,train_auc=[],[],[],[],[] #addnew
    train_ckc_metric = [train_acs, train_pre, train_rec,train_fs,train_auc]  #addnew

        
    for i in folds:
        
        datasets = args.dataset_factory.return_splits(
            args,
            csv_path='{}/splits_{}.csv'.format(args.split_dir, i),
            fold=i
        )
        
        print("Created train and val datasets for fold {}".format(i))

        # results, (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss) = _train_val(datasets, i, args)
        ckc_metric, train_ckc_metric = _train_val(datasets, i, args, ckc_metric, train_ckc_metric)

        # all_val_cindex.append(val_cindex)
        # all_val_cindex_ipcw.append(val_cindex_ipcw)
        # all_val_BS.append(val_BS)
        # all_val_IBS.append(val_IBS)
        # all_val_iauc.append(val_iauc)
        # all_val_loss.append(total_loss)
    
        #write results to pkl
        # filename = os.path.join(args.results_dir, 'split_{}_results.pkl'.format(i))
        # print("Saving results...")
        # _save_pkl(filename, results)
    
    # final_df = pd.DataFrame({
        # 'folds': folds,
        # 'val_cindex': all_val_cindex,
        # 'val_cindex_ipcw': all_val_cindex_ipcw,
        # 'val_IBS': all_val_IBS,
        # 'val_iauc': all_val_iauc,
        # "val_loss": all_val_loss,
        # 'val_BS': all_val_BS,
    # })
    final_df = pd.DataFrame({ #addnew
    'folds': list(range(folds[0], folds[-1]+1)),
    'test_acc': acs,
    'test_pre': pre,
    'test_rec': rec,
    'test_f1': fs,
    'test_auc': auc,
    'train_acc': train_acs,
    'train_pre': train_pre,
    'train_rec': train_rec,
    'train_f1': train_fs,
    'train_auc': train_auc
})
    # 计算每个度量的平均值和方差
    result_df = pd.DataFrame({
    'folds': ['mean', 'var'],
    'test_acc': [np.mean(acs), np.std(acs)],
    'test_pre': [np.mean(pre), np.std(pre)],
    'test_rec': [np.mean(rec), np.std(rec)],
    'test_f1': [np.mean(fs), np.std(fs)],
    'test_auc': [np.mean(auc), np.std(auc)],
    'train_acc': [np.mean(train_acs), np.std(train_acs)],
    'train_pre': [np.mean(train_pre), np.std(train_pre)],
    'train_rec': [np.mean(train_rec), np.std(train_rec)],
    'train_f1': [np.mean(train_fs), np.std(train_fs)],
    'train_auc': [np.mean(train_auc), np.std(train_auc)],
})
    # Concatenate final_df and result_df along the rows
    combined_df = pd.concat([final_df, result_df], ignore_index=True)
    # result_df.to_csv(os.path.join(args.results_dir, 'result.csv'), index=False)
    combined_df.to_csv(os.path.join(args.results_dir, "summary.csv"))

    # if len(folds) != args.k:
        # save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    # else:
        # save_name = 'summary.csv'
        
    # final_df.to_csv(os.path.join(args.results_dir, save_name))
    if not args.no_log:
        print('Cross validation accuracy mean: %.3f, std %.3f ' % (np.mean(np.array(acs)), np.std(np.array(acs))))
        print('Cross validation auc mean: %.3f, std %.3f ' % (np.mean(np.array(auc)), np.std(np.array(auc))))
        print('Cross validation precision mean: %.3f, std %.3f ' % (np.mean(np.array(pre)), np.std(np.array(pre))))
        print('Cross validation recall mean: %.3f, std %.3f ' % (np.mean(np.array(rec)), np.std(np.array(rec))))
        print('Cross validation fscore mean: %.3f, std %.3f ' % (np.mean(np.array(fs)), np.std(np.array(fs))))


if __name__ == "__main__":
    start = timer()

    #----> read the args
    args = _process_args()
    
    #----> Prep
    args = _prepare_for_experiment(args)
    
    #----> create dataset factory
    args.dataset_factory = SurvivalDatasetFactory(
        study=args.study, #tcga_brca_demo
        label_file=args.label_file, 
        omics_dir=args.omics_dir, 
        seed=args.seed, 
        print_info=True, 
        n_bins=args.n_classes, 
        label_col=args.label_col, 
        eps=1e-6,
        num_patches=args.num_patches,
        is_mcat = True if "coattn" in args.modality else False,
        is_survpath = True if args.modality == "survpath" else False,
        type_of_pathway=args.type_of_path) #combine

    #---> perform the experiment
    results = main(args)

    #---> stop timer and print
    end = timer()
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))