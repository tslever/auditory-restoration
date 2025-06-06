import os
import glob
import yaml
import joblib
import pathlib
import argparse
import numpy as np
import pandas as pd
from core import get_unit_names
from copy import deepcopy as dcp
from sklearn.metrics import r2_score                                                                                
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

dirname = os.path.dirname(__file__)

def main():
    method = 'pls'
    with open(os.path.join(dirname, "../inputs/parameters.yml")) as cfgfile:
        params = yaml.safe_load(cfgfile)
    win  = int((params['rde_end'] - params['rde_start'])/params['t'])
    nbasis = params['n_basis']
    ncomps = np.arange(params['nchan'], 0, -1)

    # cohorts 1 and 2:
    exp = 'nat8a'
    spectrograms = pd.read_csv(
        os.path.join(dirname, f'../build/{exp}/spectrograms.csv'),
        index_col=['stimulus', 'time'])
    spectrograms.sort_index(inplace=True)
    
    for dataset in ['alpha', 'beta', '178B', '180B']:
        if dataset in ['alpha', 'beta']:
            responses = pd.read_hdf(
                os.path.join(dirname, f'../build/{exp}/{dataset}_delemb_win{win}_basis{nbasis}.h5'),
                key='Induction'
            )
        else:
            alpha_responses = pd.read_hdf(
                os.path.join(dirname, f'../build/{exp}/alpha_delemb_win{win}_basis{nbasis}.h5'),
                key='Induction'
            )
            with open(os.path.join(dirname, f"../inputs/units/{exp}-alpha-{dataset}.txt")) as ufile:
                dset_units = ufile.read().split('\n')
            responses = alpha_responses[dset_units].copy()
        model_file = os.path.join(dirname, f"../output/{exp}/{dataset}_PLS_models.pkl")
        decode(exp, dataset, model_file, spectrograms, responses, dirname)
        
    # cohort 3:
    with open(os.path.join(dirname, "../inputs/decoder-datasets.yml")) as dsetfile:
        datasets = yaml.safe_load(dsetfile)
    for exp in ['synth8b', 'nat8b']:
        spectrograms = pd.read_csv(
            os.path.join(dirname, f'../build/{exp}/spectrograms.csv'),
            index_col=['stimulus', 'time'])
        spectrograms.sort_index(inplace=True)
    
        expdatasets = datasets[exp]
        for dataset in expdatasets:
            if dataset=='cohort':
                dset_responses = []
                for h5file in glob.glob(
                    os.path.join(dirname, f"../build/{exp}/**_delemb_win{win}_basis{nbasis}.h5")):
                    dset_responses.append(pd.read_hdf(h5file, key='Induction'))
                responses = pd.concat(dset_responses, axis=1)
                model_file = os.path.join(dirname, f"../output/{exp}/{dataset}_PLS_models.pkl")
                decode(exp, dataset, model_file, spectrograms, responses, dirname, train_all=True)
            else: # subject
                if type(dataset) == str:
                    responses = pd.read_hdf(
                        os.path.join(dirname, f"../build/{exp}/{dataset}_delemb_win{win}_basis{nbasis}.h5")
                    )
                    model_file = os.path.join(dirname, f"../output/{exp}/subject/{dataset}_PLS_models.pkl")
                    decode(exp, dataset, model_file, spectrograms, responses, dirname)

                elif type(dataset) == dict:
                    subject = list(dataset.keys())[0]
                    subject_responses = pd.read_hdf(
                        os.path.join(dirname, f"../build/{exp}/{subject}_delemb_win{win}_basis{nbasis}.h5")
                    )
                    model_file = os.path.join(dirname, f"../output/{exp}/subject/{subject}_PLS_models.pkl")
                    decode(exp,
                           dataset=subject,
                           model_file=model_file,
                           spectrograms=spectrograms,
                           responses=subject_responses,
                           dirname=dirname)

                    for region_dataset in dataset[subject]:
                        region = list(region_dataset.keys())[0]
                        recording = region_dataset[region]
                        recording_units = get_unit_names(recording, os.path.join(dirname, f"../datasets/{exp}-responses/"))
                        recording_responses = subject_responses[recording_units].copy()        
                        model_file = os.path.join(dirname, f"../output/{exp}/region/{region}_{recording}_PLS_models.pkl")
                        decode(exp,
                               dataset=region,
                               model_file=model_file,
                               spectrograms=spectrograms,
                               responses=recording_responses,
                               dirname=dirname)


def decode(exp, dataset, model_file, spectrograms, responses, dirname, train_all=False):
    print(f"Training decoder for dataset {dataset} of experiment {exp}.")
    pathlib.Path(model_file).parent.mkdir(parents=True, exist_ok=True)
    if os.path.isfile(model_file):
        print(f"Dataset {dataset} for exp {exp} already completed. Skipping")
        return
    # Stimulus sets nat8b and synth8b contains CM and GM conditions
    training_conditions = ['C', 'CM', 'N', 'G'] if exp != 'nat8a' else ['continuous', 'noise', 'gap']
    gaplocs = [1,2] if exp != 'synth8b' else [2,4]
    stim_info = pd.read_csv(
        os.path.join(dirname, f'../inputs/stimuli/{exp}-info.csv'),
        index_col='stimulus')
    motifs = stim_info.motif.unique()
    n_targets = spectrograms.shape[1]
    n_features = responses.shape[1]
    ncomps = np.arange(min(n_targets, n_features), 0, -1)
    scores = {}
    variance = {}
    for im, m in enumerate(motifs):
        if im+1==len(motifs):
            m_val = motifs[0]
        else:
            m_val = motifs[im+1]
            
        mt_train = [i for i in motifs if i not in [m, m_val]]
        scores[m] = {}
        
        print(f" - withholding {m}, validating on {m_val}, training on {mt_train}")
        
        stim_train = stim_info[(stim_info.motif.isin(mt_train)&stim_info.type.isin(training_conditions))].index
        stim_val = stim_info[(stim_info.motif==m_val)&(stim_info.type.isin(training_conditions))].index
        
        Y_train_data = spectrograms.loc[stim_train]
        X_train_data = responses.loc[Y_train_data.index]

        if not (Y_train_data.index==X_train_data.index).all():
            Y_train_data, X_train_data = Y_train_data.align(X_train_data, join='left', axis=0)
        Y_tr = Y_train_data.values.copy()
        X_tr = X_train_data.values.copy()
        Y_val_data = spectrograms.loc[stim_val]
        X_val_data = responses.loc[Y_val_data.index]
        if not (Y_val_data.index==X_val_data.index).all():
            Y_val_data, X_val_data = Y_val_data.align(X_val_data, join='left', axis=0)
        Y_vl = Y_val_data.values.copy()
        X_vl = X_val_data.values.copy()
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_vl = scaler.transform(X_vl)
        
        # variance per feature should be 1 after scaling (roughly)
        X_variance = np.var(X_vl, axis=0).sum()
        variance[m] = {}
        Y_std = Y_train_data.std(axis=0, ddof=1).values
        model = PLSRegression(n_components=ncomps[0])
        model.fit(X_tr,Y_tr)
        X_rotations = model.x_rotations_ # n_features x n_comps
        assert X_rotations.shape==(n_features, ncomps[0]), f"{X_rotations.shape}"
        Y_loadings = model.y_loadings_ # n_targets x n_comps
        assert Y_loadings.shape==(n_targets, ncomps[0]), f"{Y_loadings.shape}"
        for nc in ncomps: # from 50 to 1: dimension slice endpoint
            
            # calculate coeficients
            # refer to sklearn's _pls.py for formula
            Xrot_Yload = np.dot(X_rotations[:, :nc], Y_loadings[:, :nc].T)
            assert Xrot_Yload.shape==(n_features, n_targets), f"{Xrot_Yload.shape}"
            coefs = (Xrot_Yload * Y_std).T
            assert coefs.shape==(n_targets, n_features), f"{coefs.shape}"
            
            # predict
            Y_vl_pred = X_vl @ coefs.T + model.intercept_
            assert Y_vl_pred.shape == Y_vl.shape
            r2score = r2_score(Y_vl, Y_vl_pred)
            scores[m][nc-1] = r2score
    
            # calculate variance
            X_scores = np.dot(X_vl, X_rotations[:, :nc])
            proj_var = np.var(X_scores, axis=0).sum()
            pc_var = proj_var / X_variance
            variance[m][nc-1] = pc_var # we save variance as cumsum by dimension #
    
    pdscores = pd.DataFrame.from_dict(scores)
    optparam = pdscores.mean(axis=1).idxmax()
    print(f" + best parameter selected as {optparam}, retraining models.")
    
    models = { 'best_param': optparam,
               'scores': pdscores,
               'variances': variance}

    # retrain on best params
    for im, m in enumerate(motifs):
            
        mt_train = [i for i in motifs if i!=m]    
        stim_train = stim_info[(stim_info.motif.isin(mt_train)&stim_info.type.isin(training_conditions))].index
                
        Y_train_data = spectrograms.loc[stim_train]
        X_train_data = responses.loc[Y_train_data.index]
        if not (Y_train_data.index==X_train_data.index).all():
            Y_train_data, X_train_data = Y_train_data.align(X_train_data, join='left', axis=0)
        Y_tr = Y_train_data.values.copy()
        X_tr = X_train_data.values.copy()
    
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        
        model = PLSRegression(n_components=optparam)
        model.fit(X_tr,Y_tr)    
        models[m] = dcp(model)
        
    # single model trained on all motifs for unified projection analysis
    if train_all:
        stim_train = stim_info[stim_info.type.isin(training_conditions)].index
        Y_train_data = spectrograms.loc[stim_train]
        X_train_data = responses.loc[Y_train_data.index]
        if not (Y_train_data.index==X_train_data.index).all():
            Y_train_data, X_train_data = Y_train_data.align(X_train_data, join='left', axis=0)
        Y_tr = Y_train_data.values.copy()
        X_tr = X_train_data.values.copy()
        
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        model = PLSRegression(n_components=optparam)
        model.fit(X_tr,Y_tr)
        models['all'] = dcp(model)
    joblib.dump(models, model_file, compress=3)    
    

if __name__ == "__main__":
    main()