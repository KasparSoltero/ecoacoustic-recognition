- tested with python 3.11
- using uv:

`
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
`

for mps, run `export PYTORCH_ENABLE_MPS_FALLBACK=1`

---

This repo builds on the [bioacoustic data synthesic](https://github.com/KasparSoltero/bioacoustic-data-synthesis) framework. It involves two stages.

- stage 1 (isolator):

train-mask2former.py trains the mask2former model on the synthetic dataset to 'isolate' vocalisations. This is a single-clas mask segmenentation of the spectrogram to precisely crop all vocalisations.

- stage 2 (recogniser):

train-recogniser.py uses the trained isolator model to train a small MLP head with BirdNET to segment and classify vocalisations by species.


clone the [birdnet analyzer repo](https://birdnet-team.github.io/BirdNET-Analyzer/) and install it as a package i.e. `uv pip install -e path-to-cloned-repo`

the recogniser folder contains scripts for localisation and blind synchronisation using the recogniser model.

all parts of this framework are still undergoing testing and iteration. Please get in touch if you have any questions.