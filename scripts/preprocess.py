import os
import yaml
from core import *
import numpy as np
import pandas as pd
import pathlib

dirname = os.path.dirname(__file__)
with open(os.path.join(dirname, "../inputs/parameters.yml")) as cfgfile:
    params = yaml.safe_load(cfgfile)

# Basis set
window  = int((params['rde_end'] - params['rde_start'])/params['t'])
nbasis = params['n_basis']
r_delays = np.arange(params['rde_start'], params['rde_end'], params['t'])
rbasis = basis_set(
    r_delays, linfac=params['linearity_factor'],
    nbasis=params['n_basis'], min_offset=1e-20)
basis_columns = np.arange(params['n_basis'])


def main():
    for exp in ['synth8b', 'nat8a', 'nat8b']:
        pathlib.Path(os.path.join(dirname, f'../build/{exp}')).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(dirname,
                               f'../inputs/stimuli/{exp}-stimuli.txt'), 'r') as file:
            stim_names = file.read().split('\n')
        spectrogram_params = {
            "window_time": params['t'] / 1000,
            "channels": params['nchan'],
            "f_min": params['f_min'],
            "f_max": params['f_max'],
        }
        spectros = get_stimuli(stim_names,
                               spectrogram_params,
                               input_loc = os.path.join(dirname,
                                                        f'../datasets/{exp}-stimuli'),
                               t=params['t']/1000,
                               compression=params['compression'],
                               target_sr=20000,
                               export=None)
        spctr = spectros.reset_index(drop=False).copy()
        spctr['time'] = (spctr['time'] * 1000).round(0).astype(int) #convert stimulus time to ms
        spectros = spctr.set_index(['stimulus','time'])
        spectros.to_csv(
            os.path.join(dirname, f'../build/{exp}/spectrograms.csv'), mode='w'
        )
        print(f" - Spectrogram creation complete for {exp}")
        
        if exp == 'nat8a':
            for dataset in ['alpha','beta']:
                raw_units = units_from_folders(
                    os.path.join(dirname, f"../datasets/{exp}-{dataset}-responses")
                )
                data = nat8a_alpha_preprocess(raw_units) if dataset=='alpha' else nat8a_beta_preprocess(raw_units)
                generate_response(data, params, exp, dataset)
        else:
            with open(os.path.join(dirname, f"../inputs/units/{exp}-recordings.yml")) as yamfile:
                datasets = yaml.safe_load(yamfile)
            for dataset, recordings in datasets.items():
                raw_units = units_from_recordings(
                    recordings,
                    input_loc=os.path.join(dirname, f"../datasets/{exp}-responses")
                )
                data = preprocess(raw_units)
                generate_response(data, params, exp, dataset)


def generate_response(data, params, exp, dataset):
    print(" - Aggregating unit responses.")
    responses = get_responses(data, params['r_start'], params['r_end'], params['t']/1000)
    resps = responses.reset_index()
    resps['time'] = (resps['time'] * 1000).round(0).astype(int) # convert response time to ms
    responses = resps.set_index(['stimulus', 'time']).copy()
    responses.sort_index(inplace=True) # Sorting necessary for IndexSlicing ahead

    resp_file = os.path.join(dirname, f'../build/{exp}/responses_{dataset}.h5')
    if not os.path.isfile(resp_file):
        responses.to_hdf(resp_file, key='Induction', mode='w')
        print(f" - Dataset {dataset} for {exp} responses written to file.")

    db_file = os.path.join(dirname, f'../build/{exp}/{dataset}_delemb_win{window}_basis{nbasis}.h5')
    if os.path.isfile(db_file):
        print(f" - Delay embedding already exists for {dataset} in {exp}.")
        return
    print(" - Performing delay embedding on responses.")
    db_tempfile = os.path.join(dirname, f'../build/{exp}/{dataset}_delemb_win{window}_basis{nbasis}_temp.h5')
    nunits = responses.shape[1]
    idxes = np.arange(nunits, 0, -30)[::-1]
    i1 = 0
    for i in idxes:
        print(f" -- chunking units {i1} to {i}.")
        rchunk = responses.iloc[:, i1:i].copy()
        columns = pd.MultiIndex.from_product([rchunk.columns, r_delays])
        de_chunk = rchunk.groupby(level='stimulus').apply(
            delay_embed,
            columns=columns,
            de_start=params['rde_start'],
            de_end=params['rde_end'],
            dt=params['t']
        ).copy()
        print(f" -- applying raised cosine basis.")
        db_chunk = de_chunk.T.groupby(level='unit').apply(
            apply_basis,
            basis=rbasis,
            columns=basis_columns
        ).copy()
        db_chunk = db_chunk.unstack(0).reorder_levels([1,0], axis=1).sort_index(axis=1)
        db_chunk.index.names = ['stimulus', 'time']
        db_chunk.columns.names = ['unit', 'basis']
        db_chunk.to_hdf(db_tempfile, key=f"chunk{i}", mode='a')
        i1 = i
    db_resps = []
    print(f" -- chunking complete, stitching chunks.")
    for i in idxes:
        chunk = pd.read_hdf(db_tempfile, key=f"chunk{i}", mode='r')
        db_resps.append(chunk)
    db_resps = pd.concat(db_resps, axis=1)
    db_resps.to_hdf(db_file, key='Induction', mode='w')
    os.remove(db_tempfile)
    print(f" - Stitching complete for {dataset} in {exp}.")

if __name__ == "__main__":
    main()
