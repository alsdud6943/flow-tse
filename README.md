# FLOWTSE

FLOWTSE is a Flow Matching-based target speaker extraction (TSE) project.
Given a mixture and a short reference utterance from the target speaker, the
model extracts the target speaker's clean speech.

The main clean setup in this repository uses:

- `config/train_cfm.yaml` for training
- `config/test_cfm.yaml` for testing

## Setup

Create a new conda environment and install the dependencies:

```bash
conda env create -f environment.yaml
conda activate flowtse
```

## Train

Train the CFM-TSE model with:

```bash
python train.py --config-name=train_cfm
```

Before training, update the dataset paths in `config/train_cfm.yaml` to match
your local environment.

## Test

Evaluate a checkpoint with:

```bash
python test.py --config-name=test_cfm ckpt=/path/to/model.ckpt
```

Before testing, update the dataset paths in `config/test_cfm.yaml`.

Examples:

```bash
python test.py --config-name=test_cfm \
  ckpt=/path/to/model.ckpt \
  num_solver_steps=1 \
  ensemble=False
```

```bash
python test.py --config-name=test_cfm \
  ckpt=/path/to/model.ckpt \
  num_solver_steps=10 \
  ensemble=10
```

## Results

| Model | Solver Steps | Ensemble | PESQ | SI-SDR | ESTOI | DNSMOS (Signal) | DNSMOS (Overall) | Speaker Similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CFM-TSE | 1 | 1 | 1.99 | 12.58 | 0.82 | 3.42 | 2.92 | 0.84 |
| CFM-TSE | 10 | 10 | 2.62 | 15.28 | 0.88 | 3.50 | 3.16 | 0.89 |
