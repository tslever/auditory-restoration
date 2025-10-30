import os
import glob
import yaml
import joblib
import numpy as np
import pandas as pd
import scipy.stats as ss
from copy import deepcopy as dcp
from core import get_unit_names
import scipy.spatial.distance as dst
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse

dirname = os.path.dirname(__file__)

def main():
    win = 100 
    nbasis = 15
    min_dim=50
    # Calculate euclidean distances for cohort 1 & 2:
    exp = 'nat8a'
    stim_info = pd.read_csv(os.path.join(dirname, f"../inputs/stimuli/{exp}-info.csv"))
    spectrograms = pd.read_csv(
        os.path.join(dirname, f"../build/{exp}/spectrograms.csv"),
        index_col=[0,1])
    motifs = stim_info.motif.unique()
    gaps = stim_info[stim_info.type=='gap'].groupby(['motif','gap']).first()[['gap_start', 'gap_stop']]
    with open(os.path.join(dirname, f"../inputs/nat8-familiarity-coding.yml")) as famfile:
        familiarity = yaml.safe_load(famfile)
    conditions = ['C','G','CB','GB','N']
    comparisons = [('GB','CB'),('GB','C'),('GB','N')]
    comp_scores = []
    ## Split by familiarity in cohort 1
    alpha_resps = pd.read_hdf(
        os.path.join(dirname, f"../build/{exp}/alpha_delemb_win{win}_basis{nbasis}.h5"),
        key="reinduction")    
    for dataset in ['178B', '180B']:
        with open(os.path.join(dirname, f'../inputs/units/{exp}-alpha-{dataset}.txt')) as unitfile:
            set_units = unitfile.read().split('\n')
            set_units = [u for u in set_units if u!='']
        responses = alpha_resps[set_units]
        models = joblib.load(os.path.join(dirname, f"../output/{exp}/{dataset}_PLS_models.pkl"))
        min_dim = models['best_param'] if models['best_param'] < min_dim else min_dim
        set_fam = familiarity[dataset]
        cs = get_scores(
            exp, models, spectrograms, responses, gaps,
            conditions, comparisons)
        cs.loc[cs.motif.isin(set_fam['familiar']), 'fam'] = 'familiar'
        cs.loc[cs.motif.isin(set_fam['unfamiliar']), 'fam'] = 'unfamiliar'
        comp_scores.append(cs.copy())
    ## Cohort 1 all units
    models = joblib.load(
        os.path.join(dirname, f"../output/{exp}/alpha_PLS_models.pkl"))
    min_dim = models['best_param'] if models['best_param'] < min_dim else min_dim
    cs = get_scores(
        exp, models, spectrograms, alpha_resps, gaps,
        conditions, comparisons)
    cs['fam'] = 'alpha'
    comp_scores.append(cs.copy())
    ## Cohort 2 all units
    responses = pd.read_hdf(
        os.path.join(dirname, f"../build/{exp}/beta_delemb_win{win}_basis{nbasis}.h5"),
        key="reinduction")
    models = joblib.load(
        os.path.join(dirname, f"../output/{exp}/beta_PLS_models.pkl"))
    min_dim = models['best_param'] if models['best_param'] < min_dim else min_dim
    cs = get_scores(
        exp, models, spectrograms, responses, gaps,
        conditions, comparisons)
    cs['fam'] = 'beta'
    comp_scores.append(cs.copy())

    results = pd.concat([
        cs[cs['ndim']==min_dim] for cs in comp_scores
    ]).drop('ndim', axis=1)
    results.to_csv(os.path.join(dirname, '../output/nat8a/distances-nat8a.csv'))

    
    # Cohort 3:
    conditions = ['C','G','CB','GB','GM','CM','N']
    comparisons = [
        ('GB','CB'),('GB','C'),('GB','N'),('GB', 'G'),('GB', 'GM'),
        ('GM', 'C'),('GM','CB'),('CM', 'C'),('CB','C')
    ]
    RI_conds = ['GB','CB','GM']
    RI_comps = [('GB','CB'),('GB','GM'),('GM','CB')]

    RI_cohort = []
    RI_subject = []
    RI_region = []
    # RI, cohort 3 individuals and regions:
    with open(os.path.join(dirname, "../inputs/decoder-datasets.yml")) as dsetfile:
        datasets = yaml.safe_load(dsetfile)
    for exp in ['synth8b', 'nat8b']:
        stim_info = pd.read_csv(
            os.path.join(dirname, f"../inputs/stimuli/{exp}-info.csv"))
        spectrograms = pd.read_csv(
            os.path.join(dirname, f"../build/{exp}/spectrograms.csv"),
            index_col=[0,1])
        motifs = stim_info.motif.unique()
        gaps = stim_info[stim_info.type=='G'].groupby(['motif','gap']).first()[['gap_start', 'gap_stop']]
        gaplocs = gaps.index.levels[1].to_numpy().astype(int)
        expdatasets = datasets[exp]
        for dataset in expdatasets:
            if dataset=='cohort':
                dset_responses = []
                for h5file in glob.glob(
                    os.path.join(dirname, f"../build/{exp}/**_delemb_win{win}_basis{nbasis}.h5")):
                    dset_responses.append(pd.read_hdf(h5file, key='reinduction'))
                responses = pd.concat(dset_responses, axis=1)
                models = joblib.load(
                    os.path.join(dirname, f"../output/{exp}/{dataset}_PLS_models.pkl")) 
                cscores = get_scores(
                    exp, models, spectrograms, responses, gaps,
                    conditions, comparisons
                )
                cscores = cscores.set_index(['ndim','comp','motif','gap']).loc[models['best_param']]
                cscores.to_csv(os.path.join(dirname, f'../output/{exp}/distances-{dataset}.csv'))
                # ri_scores = calc_RI(exp, models, spectrograms, responses, gaps,
                #                     nlatent=models['best_param'],conditions=RI_conds,comparisons=RI_comps)
                ri_scores = calc_RI(cscores, models['best_param'])
                ri_scores['exp'] = exp
                RI_cohort.append(dcp(ri_scores))
            else: # subject
                if type(dataset) == str:
                    responses = pd.read_hdf(
                        os.path.join(dirname, f"../build/{exp}/{dataset}_delemb_win{win}_basis{nbasis}.h5"),
                        key='reinduction'
                    )
                    models = joblib.load(
                        os.path.join(dirname, f"../output/{exp}/subject/{dataset}_PLS_models.pkl"))
                    cscores = get_scores(
                        exp, models, spectrograms, responses, gaps,
                        conditions, comparisons
                    ).set_index(['ndim','comp','motif','gap']).loc[models['best_param']]
                    # ri_scores = calc_RI(exp, models, spectrograms, responses, gaps,
                    #                 nlatent=models['best_param'],conditions=RI_conds,comparisons=RI_comps)
                    ri_scores = calc_RI(cscores, models['best_param'])
                    ri_scores['exp'] = exp
                    ri_scores['subject'] = dataset
                    RI_subject.append(dcp(ri_scores))

                elif type(dataset) == dict:
                    subject = list(dataset.keys())[0]
                    subject_responses = pd.read_hdf(
                        os.path.join(dirname, f"../build/{exp}/{subject}_delemb_win{win}_basis{nbasis}.h5"),
                        key='reinduction'
                    )
                    models = joblib.load(
                        os.path.join(dirname, f"../output/{exp}/subject/{subject}_PLS_models.pkl"))
                    cscores = get_scores(
                        exp, models, spectrograms, subject_responses, gaps,
                        conditions, comparisons
                    ).set_index(['ndim','comp','motif','gap']).loc[models['best_param']]
                    # ri_scores = calc_RI(exp, models, spectrograms, subject_responses, gaps,
                    #                     nlatent=models['best_param'],conditions=RI_conds,comparisons=RI_comps)
                    ri_scores = calc_RI(cscores, models['best_param'])
                    ri_scores['exp'] = exp
                    ri_scores['subject'] = subject
                    RI_subject.append(dcp(ri_scores))

                    for region_dataset in dataset[subject]:
                        region = list(region_dataset.keys())[0]
                        recording = region_dataset[region]
                        recording_units = get_unit_names(recording, os.path.join(dirname, f"../datasets/{exp}-responses/"))
                        recording_responses = subject_responses[recording_units].copy()        
                        models = joblib.load(os.path.join(
                            dirname, f"../output/{exp}/region/{region}_{recording}_PLS_models.pkl"))
                        cscores = get_scores(
                            exp, models, spectrograms, recording_responses, gaps,
                            conditions, comparisons
                        ).set_index(['ndim','comp','motif','gap']).loc[models['best_param']]
                        # ri_scores = calc_RI(exp, models, spectrograms, recording_responses, gaps,
                        #                     nlatent=models['best_param'],conditions=RI_conds,comparisons=RI_comps)
                        ri_scores = calc_RI(cscores, models['best_param'])
                        ri_scores['exp'] = exp
                        ri_scores['subject'] = subject
                        ri_scores['recording'] = recording
                        ri_scores['region'] = region
                        RI_region.append(dcp(ri_scores))
    pd.concat(RI_cohort).to_csv(os.path.join(dirname, f"../output/RI_cohort.csv"))
    pd.concat(RI_subject).to_csv(os.path.join(dirname, f"../output/RI_subject.csv"))
    pd.concat(RI_region).to_csv(os.path.join(dirname, f"../output/RI_region.csv"))
    
def bname(m, c, g=None):
    gi = 'a' if 'synth' in m else ''
    if g:
        g = int(g)
    if c in ['GB','CB','N','GM']:
        return f"ep_{m}_{c}_g{g}{gi}_snr0"
    elif c=='CM':
        return f"ep_{m}_{c}_snr0"
    elif c=='C':
        return f"ep_{m}_{c}"
    elif c=='G':
        return f"ep_{m}_{c}_g{g}{gi}"

def aname(m, c, g=None):
    if g:
        g = int(g)
    if c == 'N':
        return f"{m}_noise{g}"
    elif c=='C':
        return f"{m}_continuous"
    elif c=='G':
        return f"{m}_gap{g}"
    elif c=='CB':
        return f"{m}_continuousnoise{g}"
    elif c=='GB':
        return f"{m}_gapnoise{g}"
        
# def calc_RI(exp, models, spectrograms, responses, gaps, nlatent, conditions, comparisons):
#     riscores = []
#     sn =  aname if exp=='nat8a' else bname
#     gaplocs = gaps.index.levels[1].to_numpy().astype(int)
#     for (m, g), gaptimes in gaps.iterrows():
#         model = models[m]
#         cidx = spectrograms.loc[sn(m, 'C', None)].index
#         projections = {}
#         ga, gb = gaptimes.astype(int)
#         for c in conditions:
#             stim = sn(m, c, g)
#             X = model.transform(X = responses.loc[stim].loc[cidx])
#             projections[c] = X
            
#         for a, b in comparisons:
#             distances = np.array([dst.euclidean(
#                 projections[a][t, :nlatent],
#                 projections[b][t, :nlatent],
#             ) for t in np.arange(ga, gb)])
#             riscores.append({
#                     'motif': m,
#                     'gap': g,
#                     'comp': f"{a}{b}",
#                     'distances': distances
#             })
#     scores = pd.DataFrame(riscores).set_index(['motif','gap','comp']).unstack('comp').droplevel(0,axis=1)
#     ri_results = []       
#     for ir, row in scores.iterrows():
#         S = 0.25 *\
#             np.sqrt(np.array(row['GBCB']+row['GBGM']+row['GMCB']).round(decimals=16)) *\
#             np.sqrt(np.clip(-row['GBCB']+row['GBGM']+row['GMCB'], a_min=1e-16, a_max=None)) *\
#             np.sqrt(np.clip(row['GBCB']-row['GBGM']+row['GMCB'], a_min=1e-16, a_max=None)) *\
#             np.sqrt(np.clip(row['GBCB']+row['GBGM']-row['GMCB'], a_min=1e-16, a_max=None))
#         S = np.clip(S, a_min=1e-16, a_max=0.5)
#         d = np.clip(S*2/row['GMCB'], a_min=1e-16, a_max=np.sqrt(2))
#         ri = (row['GBGM']-row['GBCB']) / np.abs(row['GBGM']-row['GBCB']) *\
#                 np.sqrt(np.abs(row['GBGM']-row['GBCB']) / d)
#         ri_results.append({
#             'motif': row.name[0],
#             'gap': row.name[1],
#             'RI': ri.mean()
#         })
#     return pd.DataFrame(ri_results)
    
def calc_RI(cscores, param):
    RI = cscores.unstack('comp').droplevel(0, axis=1)[['GBCB','GBGM','GMCB']]
    RI['S'] = 0.25 *\
            np.sqrt(RI['GBCB']+RI['GBGM']+RI['GMCB']) *\
            np.sqrt(np.clip(-RI['GBCB']+RI['GBGM']+RI['GMCB'], a_min=1e-16, a_max=None)) *\
            np.sqrt(np.clip(RI['GBCB']-RI['GBGM']+RI['GMCB'], a_min=1e-16, a_max=None)) *\
            np.sqrt(np.clip(RI['GBCB']+RI['GBGM']-RI['GMCB'], a_min=1e-16, a_max=None))
    RI['d'] = RI['S']*2/RI['GMCB']
    RI['RI'] = (RI['GBGM']-RI['GBCB']) / np.abs(RI['GBGM']-RI['GBCB']) *\
            np.sqrt(np.abs(RI['GBGM']-RI['GBCB'])) / RI['d']
    RI.reset_index(drop=False, inplace=True)
    return RI
    
def get_scores(
    exp, models, spectrograms, responses, gaps,
    conditions, comparisons,
    full=True,
):
    comp_scores = []
    nlatent = models['best_param']
    sn=aname if exp=='nat8a' else bname
    gaplocs = gaps.index.levels[1].to_numpy().astype(int)
    for (m, g), gaptimes in gaps.iterrows():
        model = models[m]
        cidx = spectrograms.loc[sn(m, 'C', None)].index
        projections = {}
        ga, gb = gaptimes.astype(int)
        for c in conditions:
            stim = sn(m, c, g)
            X= model.transform(
                X = responses.loc[stim].loc[cidx],
            )
            projections[c] = X
            
        for a, b in comparisons:
            if full:
                for e in np.arange(1, nlatent+1):
                    distances = np.array([dst.euclidean(
                        projections[a][t, :e],
                        projections[b][t, :e],
                    ) for t in np.arange(ga, gb)])
                    score = np.mean(distances)
                    comp_scores.append({
                        'motif': m,
                        'gap': g,
                        'comp': f"{a}{b}",
                        'ndim': e,
                        'score': score
                    })
            else:
                distances = np.array([dst.euclidean(
                    projections[a][t],
                    projections[b][t],
                ) for t in np.arange(ga, gb)])
                score = np.mean(distances)
                
                comp_scores.append({
                    'motif': m,
                    'gap': g,
                    'comp': f"{a}{b}",
                    'score': score
                })
    return pd.DataFrame(comp_scores)

if __name__ == "__main__":
    main()