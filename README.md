# A Diffusion Model with Key Node Feature Guidance for Digital Twin of Rocket Engine Frames under Variable Working Conditions

Official implementation for the paper **"A Diffusion Model with Key Node Feature Guidance for Digital Twin of Rocket Engine Frames under Variable Working Conditions"**.

![Overview of the training process of the proposed method](./picture/Overview%20of%20the%20training%20process%20of%20the%20proposed%20method.jpg)

## Introduction

We propose a diffusion model with key node feature guidance for digital twin modeling of rocket engine frames under variable working conditions. The proposed method introduces key node condition information into the diffusion process so that the generated stress field remains consistent with the working-condition-dependent structural response. By combining source-domain pretraining, target-domain fine-tuning, and conditional sampling, the method improves both prediction accuracy and transfer capability across different operating conditions.

To validate the effectiveness of the proposed method, we conduct two experimental studies under variable working conditions and compare the model with five representative baseline methods: Co-kriging, CSF-RBF, QSCGAN, VGCDM, and VQ-VAE. Experimental results show that the proposed method achieves superior performance in conditional stress-field prediction, demonstrating its effectiveness for rocket engine frame digital twin modeling.

## Method

The released implementation follows the workflow described in the paper:

1. Offline pretraining on source-domain stress-field data.
2. Online fine-tuning on target-domain stress-field data.
3. Conditional sampling and evaluation under target working conditions.

The main implementation is provided in:

- `denoising_diffusion_pytorch/guidance_diffusion_all step_trans.py`

The timing analysis script is provided in:

- `denoising_diffusion_pytorch/timing_benchmark.py`

## Repository Structure

- `denoising_diffusion_pytorch/guidance_diffusion_all step_trans.py`: main method used in the paper
- `denoising_diffusion_pytorch/timing_benchmark.py`: DDIM timing and accuracy benchmark
- `denoising_diffusion_pytorch/stress_data_utils.py`: shared data loader and synthetic benchmark generator
- `compared methods/`: five comparison methods with unified stress-field experiment scripts
- `EXPERIMENTS.md`: experiment mapping and command summary
- `requirements.txt`: Python dependencies

## Data Availability

The original stress-field data used in the paper is not released in this repository because it involves sensitive aerospace-domain data.

To support public code inspection and timing verification, this repository provides a **synthetic benchmark mode**. The synthetic data preserves the same tensor interface as the private experiment data:

- source-domain tensor shape: `(24, 1, 24, 88)`
- target-domain tensor shape: `(24, 1, 24, 88)`
- condition embedding shape: `(24, 64)`

Tensor meaning:

- dimension 1: loading condition index
- dimension 2: channel dimension
- dimension 3: radial direction
- dimension 4: axial direction

The synthetic benchmark is generated in `denoising_diffusion_pytorch/stress_data_utils.py`. It is designed only for:

1. verifying that the released code can run end-to-end
2. reproducing the public timing analysis pipeline
3. preserving the same input and output interface as the private experiments

It is **not** intended to reproduce the paper's final quantitative accuracy on the confidential aerospace dataset.

## Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

If you use GPU acceleration, install a PyTorch build compatible with your CUDA environment.

## Reproducing the Main Method

### Public Synthetic Benchmark

Run the full training and inference pipeline using synthetic benchmark data:

```bash
cd denoising_diffusion_pytorch
python "guidance_diffusion_all step_trans.py" --use-synthetic-data --skip-plots
```

### Private Experiment Data

If you have local access to the original tensor files, the same script supports the private experiment format:

```bash
cd denoising_diffusion_pytorch
python "guidance_diffusion_all step_trans.py" --group-id 1 --skip-plots
```

Expected private data naming format:

- `Data_<id>_trans.pt`
- `Data_<id>_t_trans.pt`
- `classes_emb_trans_<id>.64.pt`

## Timing Analysis

Public timing analysis can be performed without the private dataset:

```bash
cd denoising_diffusion_pytorch
python timing_benchmark.py --use-synthetic-data
```

This command writes:

- `denoising_diffusion_pytorch/results/ddim_latency_accuracy_table.csv`
- `denoising_diffusion_pytorch/results/ddim_latency_accuracy_table.md`

The timing script uses the same model construction and sampling path as the main method. When `--use-synthetic-data` is enabled, only the input tensors are replaced, so the released benchmark remains suitable for public verification of runtime efficiency.

## Comparison Methods

Five comparison methods are included in this repository:

- `Co-kriging`
- `CSF-RBF`
- `QSCGAN`
- `VGCDM`
- `VQ_VAE`

Each comparison method provides a unified `reproduce_for_stress.py` entry script and supports:

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
- Synthetic benchmark data is provided only for code execution checks and timing analysis.
- The main method and all comparison methods follow the same two-stage source-domain / target-domain experiment structure.

## Citation

If you use this repository in your research, please cite the corresponding paper.
