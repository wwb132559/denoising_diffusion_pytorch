# Stress-Field Transfer Experiments

This repository contains the implementation of the stress-field transfer method used in the paper, together with five comparison methods and the scripts used for timing analysis.

## What Is Included

- `denoising_diffusion_pytorch/guidance_diffusion_all step_trans.py`
  Main method in the paper.
- `denoising_diffusion_pytorch/timing_benchmark.py`
  DDIM timing and accuracy benchmark script.
- `compared methods/`
  Five comparison methods with unified `reproduce_for_stress.py` entry points.
- `EXPERIMENTS.md`
  Experiment mapping and command summary.
- `requirements.txt`
  Python dependencies.

## Data Availability

The original stress-field data used in the paper is not released in this repository because it involves sensitive aerospace-domain data.

To allow code inspection and public timing verification, the repository supports a synthetic benchmark mode. This mode generates random but structured tensors with the same public data interface as the real experiment:

- source-domain tensor shape: `(24, 1, 24, 88)`
- target-domain tensor shape: `(24, 1, 24, 88)`
- condition embedding shape: `(24, 64)`

Tensor meaning:

- dimension 1: loading condition index
- dimension 2: channel dimension
- dimension 3: radial direction
- dimension 4: axial direction

## Environment

```bash
pip install -r requirements.txt
```

Install a matching PyTorch build for your CUDA environment if needed.

## Main Method

Run the main method with synthetic benchmark data:

```bash
cd denoising_diffusion_pytorch
python "guidance_diffusion_all step_trans.py" --use-synthetic-data --skip-plots
```

If you have access to the private experiment tensors locally, the same script supports the real experiment format:

```bash
cd denoising_diffusion_pytorch
python "guidance_diffusion_all step_trans.py" --group-id 1 --skip-plots
```

The script follows the paper workflow:

1. offline pretraining on source-domain data
2. online fine-tuning on target-domain data
3. conditional sampling and metric reporting

## Timing Analysis

Public timing analysis can be run without the private dataset:

```bash
cd denoising_diffusion_pytorch
python timing_benchmark.py --use-synthetic-data
```

This writes:

- `denoising_diffusion_pytorch/results/ddim_latency_accuracy_table.csv`
- `denoising_diffusion_pytorch/results/ddim_latency_accuracy_table.md`

## Comparison Methods

Five comparison methods are included:

- `Co-kriging`
- `CSF-RBF`
- `QSCGAN`
- `VGCDM`
- `VQ_VAE`

Each method now supports the same experiment interface:

- `--group-id`
- `--use-synthetic-data`
- `--stage offline|online|both`
- `--do_sample`

Example:

```bash
cd "compared methods/QSCGAN"
python reproduce_for_stress.py --use-synthetic-data --stage both --do_sample
```

## Notes

- Raw paper data is intentionally excluded from the public release.
- The synthetic benchmark mode is intended for code execution checks and timing analysis, not for reproducing the paper accuracy numbers.
- The main method and the five comparison methods use the same two-stage experiment structure: source-domain stage and target-domain stage.
