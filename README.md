# Diagonal Rescaling for Joint Weight-Activation Quantization

This repository contains code for optimization-based diagonal rescaling for post-training joint weight-activation quantization of large language models. The implementation focuses on alternating optimization of diagonal scales and quantized weights, together with efficient coordinate updates and active-set screening for the scale step.

The codebase includes:

- the alternating quantization method used in the paper
- evaluation scripts for perplexity, zero-shot accuracy
- launch scripts for experiments, with and without rotation preprocessing


## Main Files

- `SQ_alternating.py`
  Main entry point for the alternating quantization pipeline.

- `efficient_updates.py`
  Efficient Gram-form objective updates, coordinate search, and active-set alg.

- `Project/eval.py`
  Evaluation script for perplexity and zero-shot accuracy.

## Running Experiments

Experiments are launched through the provided sh wrappers:

`sh_scripts/quantize_qwen.sh`
`sh_scripts/quantize_qwen_w_rotation.sh`

