# Experiment Notes

## Main Method

Script:

- `denoising_diffusion_pytorch/guidance_diffusion_all step_trans.py`

Stages:

1. offline pretraining on source-domain stress fields
2. online fine-tuning on target-domain stress fields
3. target-domain prediction and evaluation

## Public Timing Mode

Because the original data is not public, the repository provides synthetic benchmark data generation.

Synthetic benchmark command:

```bash
cd denoising_diffusion_pytorch
python "guidance_diffusion_all step_trans.py" --use-synthetic-data --skip-plots
python timing_benchmark.py --use-synthetic-data
```

Synthetic tensor interface:

- source-domain tensor: `(24, 1, 24, 88)`
- target-domain tensor: `(24, 1, 24, 88)`
- condition embedding: `(24, 64)`

## Private Data Interface

If a user has local access to the original tensors, the expected filenames are:

- `Data_<id>_trans.pt`
- `Data_<id>_t_trans.pt`
- `classes_emb_trans_<id>.64.pt`

The released scripts accept:

- `--group-id`
- `--source-data`
- `--target-data`
- `--classes-emb`

## Comparison Methods

The following comparison methods are provided with unified stress-field experiment scripts:

- `compared methods/Co-kriging/reproduce_for_stress.py`
- `compared methods/CSF-RBF/reproduce_for_stress.py`
- `compared methods/QSCGAN/reproduce_for_stress.py`
- `compared methods/VGCDM/reproduce_for_stress.py`
- `compared methods/VQ_VAE/reproduce_for_stress.py`

Each comparison script supports:

- two-stage execution with `--stage`
- experiment group selection with `--group-id`
- synthetic timing mode with `--use-synthetic-data`

## Checkpoints And Outputs

Main method outputs:

- `model-all step-trans-<tag>.pt`
- `predicted_group<id>_target.pt`

Comparison method outputs are written into the corresponding `results_stress/` directory under each method folder.
