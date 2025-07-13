import os
import glob
import json
import h5py
import ewave
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample
from gammatone.gtgram import gtgram, gtgram_strides
from gammatone.filters import erb_space
import scipy.stats as ss

### STIMULI FUNCTIONS
def get_stimuli(stim_names, spectrogram_params, input_loc = '../datasets/stimuli',
                t=0.001, compression=10.0, target_sr=48000, export=None):
    # Read all stimuli from neurobank in as oscillograms
    stimuli = []
    for stim_name in stim_names:
        stim_path = Path(f"{input_loc}/{stim_name}.wav")
        with ewave.open(stim_path, "r") as fp:
            sr = fp.sampling_rate
            samples = fp.read()
        if sr!=target_sr:  # resample to a millisecond-divisible sampling rate
            samples = resample(samples, int(len(samples)/sr*target_sr))
            sr = target_sr
        stimuli.append(
            {"stimulus": stim_name,
             "samples": samples,
             "sample_rate":  sr
            })
    stimuli = pd.DataFrame.from_records(stimuli).set_index('stimulus')
    # Compute Spectrograms, concatenating row-wise with each freq as a column
    spectros = pd.concat(
        {stim: compute_spectrogram(row, spectrogram_params, t, compression)
         for stim, row in stimuli.iterrows()},
        names=['stimulus','time']
    )
    # Convert values in time index to milisecond precision
    spectros = spectros.reset_index(level='time').round({'time':3}).set_index('time',append=True, drop=True)
    if export is not None:
        spectros.to_csv(export, mode='w')
    return spectros


def compute_spectrogram(row, spectrogram_params, t, compression):
    duration = row.samples.size / row.sample_rate
    # number of samples per desired timestep hop
    _, hop_samples, _ = gtgram_strides(row.sample_rate, spectrogram_params["window_time"], t, row.samples.size)
    # actual hop time
    hop_time = hop_samples / row.sample_rate
    columns = erb_space(spectrogram_params["f_min"], 
                        spectrogram_params["f_max"],
                        spectrogram_params["channels"])[::-1].round(0)
    gtgram_params = {k:v for k,v in spectrogram_params.items() if k!='f_max'}
    spectrogram = gtgram(row.samples, row.sample_rate, hop_time=t, **gtgram_params)
    _, nframes = spectrogram.shape
    spectrogram = np.log10(spectrogram + compression) - np.log10(compression)
    ## Checking for mismatch between sampling rate time (hop time) and response time (t)
    ## If get_stimuli() is used along with a proper sampling rate conversion (i.e. target_sr), this should succeed. 
    assert hop_time == t
    index = np.arange(0.0, nframes*hop_time, hop_time)
    return pd.DataFrame(spectrogram.T, columns=columns, index=index).rename_axis(index="time", columns="frequency")


### DATA FUNCTIONS
def units_from_recordings(recording_names, input_loc = '../datasets/units'):
    multi_records = []
    for r in recording_names:
        if r=='':
            continue
        # pprox files of the recording will be in the same folder as the arf file
        globber = str(Path(input_loc)) + f"/{r}*.pprox" 
        names = {Path(u).stem : Path(u) for u in glob.glob(globber)}
        print(f" -- Found {len(names)} units from recording {r}")
        units = {}
        for u, pprox in names.items():
            pprox_data = json.loads(pprox.read_text())
            units[u] = (
                pd.json_normalize(pprox_data["pprox"])
                .rename(columns={"stimulus.name": "stimulus"})
            )
            try:
                units[u].set_index("index", inplace=True)
            except KeyError:
                pass
        if units:
            multi_records.append(pd.concat(units, names=("unit", "trial")).sort_index())
    if len(multi_records)>0:
        return pd.concat(multi_records)
    else:
        return None


def units_from_folders(input_loc = '../datasets/units'):
    globber = str(Path(input_loc)) + "/*.pprox"
    names = {Path(u).stem : Path(u) for u in glob.glob(globber)}
    print(f" -- Found {len(names)} units from directory {input_loc}")
    units = {}
    for u, pprox in names.items():
        pprox_data = json.loads(pprox.read_text())
        units[u] = (
            pd.json_normalize(pprox_data["pprox"])
            .rename(columns={"stimulus.name": "stimulus"})
        )
        try:
            units[u].set_index("index", inplace=True)
        except KeyError:
            pass
    return pd.concat(units, names=('unit', 'trial'))


def get_unit_names(recording, dataset_loc):
    globber = str(Path(dataset_loc)) + f"/{recording}*.pprox"
    names = [Path(u).stem for u in glob.glob(globber)]
    return names


def preprocess(df):
    df[['stimulus_start','stimulus_end']] = pd.DataFrame(df['stimulus.interval'].to_list(), index=df.index).round(3)
    splits = df.stimulus.str.split('_', expand=True)
    df['motif'] = splits[1]
    df['type'] = splits[2]
    df['gap'] = splits[3].str.split('g', expand=True)[1].fillna(0)
    df['snr'] = splits[4].str.split('r', expand=True)[1].fillna(30)
    df = df.drop(columns=['offset', 'recording.entry','recording.start','recording.end','interval', 'stimulus.interval'])
    return df
     
def nat8a_beta_preprocess(df):
    df[['stimulus_start','stimulus_end']] = pd.DataFrame(df['stimulus.interval'].to_list(), index=df.index).round(3)
    splits = df.stimulus.str.split('_', expand=True)
    df['base_stim'] = splits[0]
    df['condition'] = splits[1]
    df = df.drop(columns=['offset', 'recording.entry','recording.start','recording.end','interval', 'stimulus.interval'])
    return df

def nat8a_alpha_preprocess(df):
    df = df.rename({
        'stim_on': 'stimulus_start',
        'stim_off': 'stimulus_end',
        'stimulus': 'motif',
        'event': 'events'
    }, axis=1)
    df['events'] = df.events.apply(lambda x: np.array(x)/1000)
    df['stimulus_end'] = df['stimulus_end'] / 1000
    df['stimulus_start'] = df['stimulus_start'] / 1000
    df['stimulus'] = df['motif'] + '_' + df['condition']
    df = df.drop(columns=['stim_uuid', 'category', 'units', 'trial'])
    return df


### REPSONSE FUNCTIONS
def get_responses(data, r_start=-0.5, r_end=0.5, t=0.001, instant=False):
    # Get start & end times for each stimulus, note that one stimuli may have multiple start&stop times
    # This is probably due to recording buffer differences? Best to implement some min() method here
    onoff = data[['stimulus','stimulus_start','stimulus_end']].reset_index(drop=True).drop_duplicates().set_index('stimulus')
    # For each stimuli, create an array of time bins of responses between stim_start+r_start and stim_end+r_end
    bins = data.reset_index().groupby('stimulus').first().apply(
        expand, axis=1, start=r_start, stop=r_end, bin_size=t
    ) 
    # Expand 'events' column of pprox data into rows based on stimuli time bins.
    # Resulting df has each stimuli's time bin as rows and each unit as a column.
    responses = []
    # Convert to instantaneous firing rate if instant=True 
    norm = t if instant else 1
    for s, sd in data.reset_index().groupby('stimulus'):
        b = bins.loc[s] # Get previously calculated time bin for stimuli
        # Convert per-stimulus responses
        # This results in a Series with meaningless index
        resp = sd.apply(point_process, axis=1, bins=b, norm=norm)
        
        df = pd.DataFrame(
            # resp.values is an array of arrays (since resp is a Series),
            # np.stack converts it to a 2d-array 
            np.stack(resp.values),
            # reusing the proper index
            index=pd.MultiIndex.from_frame(sd[['unit','trial']]),
            # number of bins = number of bin edges - 1
            # This means activity at time 't' is actually 
            # activity between time t and t+dt (good for a decoder)
            columns=b[:-1]
        )
        # Average response per unit over all trials/presentation of a stimulus
        # The pd.concat() simply adds 'stimulus' as an extra outer-most index level
        means = pd.concat([df.groupby(level='unit').mean().T], keys=[s], names=['stimulus'])
        responses.append(means) 
    responses = pd.concat(responses, axis=0)
    responses = responses.reset_index(level=1).round(
        {'level_1':3} # enforce rounding on "time" index
    ).rename(
        columns={'level_1': 'time'} # name column
    ).set_index(
        'time', append=True, drop=True # convert back to index
    )
    return responses


def anova_filter(data, r_start=-0.5, r_end=0.5, t=0.001):
    dat = data.reset_index()
    onoff = data[['stimulus','stimulus_start','stimulus_end']].reset_index(drop=True).drop_duplicates().set_index('stimulus')
    bins = dat.groupby('stimulus').first().apply(
        expand, axis=1, start=r_start, stop=r_end, bin_size=t
    )
    p_vals = {}
    # Calculate firing rate during silence and during stimulis per unit & stimulus
    # If the unit's response to ANY stimulus differs from silence, consider it auditory
    for u, ud in dat.groupby('unit'):
        u_silence = []
        u_esp = []
        for s, sd in ud.groupby('stimulus'):
            b = bins[s]
            resp = sd.apply(point_process, axis=1, bins=b, norm=None)
            r = pd.DataFrame(
                np.stack(resp.values),   
                index=pd.MultiIndex.from_frame(sd[['unit','trial']]),
                columns=b[:-1]
            ).T
            r.index.name='time'
            r.index = r.index.astype(float)
            silence = r[r.index<0].mean(axis=0).to_frame(name='silence').droplevel(1).reset_index()
            silence['trial'] = np.arange(len(silence))
            silence = silence.set_index(['unit', 'trial']).T
            u_silence.append(silence)
    
            esp = r[r.index.to_series().between(*onoff.loc[s].values).values].mean(axis=0).to_frame(name=s).droplevel(1).reset_index()
            esp['trial'] = np.arange(len(esp))
            esp = esp.set_index(['unit', 'trial']).T
            u_esp.append(esp)
    
        silence_samples = pd.concat(u_silence).mean().values
        esp_samples = pd.concat(u_esp).values
        f, p = ss.f_oneway(*np.vstack([esp_samples, silence_samples]))
        p_vals[u] = [f, p]
    
    results = pd.DataFrame.from_dict(p_vals, orient='index', columns=['F','p'])
    passed = results[results.p<0.05].index.tolist()
    return passed


def basis_set(lag_steps, linfac, nbasis, min_offset):
    
    def curve(center, domain, space):
        cos_input = np.clip((domain - center) * np.pi / space / 2, -np.pi, np.pi)
        return (np.cos(cos_input) + 1) / 2
    
    nonlinearity = lambda x: np.log(x + linfac + min_offset)
    first = nonlinearity(0)
    last = nonlinearity(len(lag_steps) * (1 - 1.5 / nbasis))
    peaks = np.linspace(first, last, nbasis)
    
    logdom = nonlinearity(lag_steps)
    peak_space = (last - first) / (nbasis - 1)
    basis = np.column_stack(
        [curve(c, logdom, peak_space) for c in peaks]
    )
    basis /= np.linalg.norm(basis, axis=0)
    return basis


def delay_embed(df, columns, de_start, de_end, dt):
    if de_end==0: # encoder (stimulus) delay-embedding (not used)
        index = pd.Index(df.index.get_level_values(1).to_list()[int(-de_start/dt):])
    else: # decoder (response) delay-embedding
        index = pd.Index(df.index.get_level_values(1).to_list()[int(-de_start/dt):int(-de_end/dt)])
    de_data = np.zeros( [len(index), len(columns)] )
    stimless = df.droplevel(0)
    for i, idx in enumerate(index):
        loc1 = idx+de_start
        loc2 = idx+de_end-dt
        de_data[i, :] = stimless.loc[loc1:loc2].unstack().values
    return pd.DataFrame(
        de_data,
        index=index,
        columns=columns
    )

def apply_basis(df, basis, columns):    
    return pd.DataFrame(
        np.dot(df.T, basis),
        index=df.columns,
        columns=columns
    )

def expand(df, start, stop, bin_size):
    # simple returns the ms-wide bins for each stimulus
    # to be used for np.histogram() in point_process()
    a = df.stimulus_start + start
    # Add bin_size to the end since np.histogram() 
    # requires right-most edge as well. 
    b = df.stimulus_end + stop + bin_size
    return np.arange(a, b, bin_size)

def point_process(row, bins, norm=None):
    # Set norm to dt to convert to instantaneous firing rate
    if norm is not None:
        return np.histogram(row.events, bins=bins)[0] / norm
    else:
        return np.histogram(row.events, bins=bins)[0]
