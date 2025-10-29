# Auditory Restoration

This repository contains analysis code to accompany Le et. al, "The zebra finch auditory cortex reconstructs occluded syllables in conspecific song" (preprint).

This README file describes the key steps in running the analysis, including links to data files hosted on public repositories. See the `docs` folder for information about installing dependencies, using a high performance cluster, and other topics. The instructions should work on Linux or any other POSIX-compatible operating system. Windows users will need to port batch scripts.

- On UVA's high performance cluster Rivanna, run `module list`.
- Run `module load miniforge` to change Python version from 3.6.8 to 3.11.6.
- Run `conda create -n r-4.3 r-base=4.3 r-essentials`.
- Run `conda activate r-4.3`.
- Run `R --version`.

## Datasets

The data for the analysis has been deposited as zip files in figshare datasets.

To download, verify, and unpack the dataset, do the following.
- Run `chmod +x scripts/fetch_datasets.sh`.
- Run `git update-index --chmod=+x scripts/fetch_datasets.sh`.
- Run `mkdir datasets`.
- Download `zebf-auditory-restoration-1.zip` from `https://figshare.com/ndownloader/files/55083911`.
- Add the archive to folder `datasets`.
- Run `unzip -o datasets/zebf-auditory-restoration-1.zip -d datasets`.
- Run `rsync -a datasets/zebf-auditory-restoration-1/ datasets/`.

## Setup and general notes

Temporary files are created in the `build` directory, while model results and statistics are store under `output`. To restart the analysis from the beginning, you can just clear these directory out.

The code is a combination of scripts and Jupyter notebooks. You will need to run the scripts first and then the notebooks.

If you're starting from a fresh repository, see `docs/installation.md` for instructions about how to set up your Python and R environments.

## Electrophysiology

The dataset is split into three experiments, 2 using natural song stimuli: `nat8a` and `nat8b`, and one using scrambled song motifs: `synth8b`. Each experiment contains the extracellular spike times (`.pprox` files) recorded from the auditory pallium under `dataset/{exp}-responses` and its corresponding stimuli set under `dataset/{exp}-stimuli`; `metadata` directory contains information about the vocalizers used for the study, the ephys-recorded birds, and recording sites. 

Further, the `inputs` directory contains information about how each experiment was organized, i.e. split between different cohorts and subjects. Metadata about the stimuli, including the critical intervals and naming convention for different stimuli conditions can be found under `inputs\stimuli\{exp}`.

### Initial preprocessing

Spike waveforms were saved during spike sorting by `group-kilo-spikes`, a console script created by installing our collection of custom Python code (https://github.com/melizalab/melizalab-tools). Each unit spiketimes are stored within a `.pprox` file alongside the stimulus presentation metadata. Run `./scripts/preprocess.py` to perform the data aggregating necessary for decoder analysis. The script will first convert cohort neural responses into time series data stored under `build/{exp}/responses-{dataset}.h5`, then further perform delay-embedding of the neural data with parameters specified in `inputs/parameters.yml`.

### Decoder Analysis

All the decoders found in the journal article can be fit using `scripts/decode.py`. This will load the data for each cohort/subject/recording, and use cross-validation to determine the optimal PLS hyperparameter, and then do train/test splits for each motif to compute predictions from the responses to illusion-inducing stimuli. Following, run `scripts/analyse.py` to calculate the Euclidean distances between pairs of stimulus condition, as well as the Restoration Index measure used in the paper.

To recreate figures from the publication, run the corresponding figure's notebook under `notebooks`. You should run `notebooks/R-models.ipynb` first to estimate the group means.