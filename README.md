# Compression Hacking: A Supplementary Perspective on Informatics Metric of Language Models from Geometric Distortion

## Introduction

This package is a supplementary package for the paper "Compression Hacking: A Supplementary Perspective on Informatics Metric of Language Models from Geometric Distortion". The package contains the implementation of the language models and the 3 evaluation metrics we proposed. You can easily use the package to evaluate the language models with them.

## Usage

First, you need to install the package. You can install it by running the following command:

```bash
git clone https://github.com/hggzjx/compression_hacking.git
cd compression_hacking
pip install .
pip install -e . # install in editable mode
```

Then, you can use the package to evaluate the language models. For example, you can use the following code to evaluate the language models:

```python
from evaluator import IFEvaluator
evaluator = IFEvaluator(model_path="facebook/opt-1.3b",sample_size=80,batch_size=4)
results = evaluator.evaluate(metric="semantic_cv")
print(results)
```

## Citation

If you use this package in your research, please cite our paper:

```bibtex
@article{zang2025compression,
  title={Compression Hacking: A Supplementary Perspective on Informatics Metric of Language Models from Geometric Distortion},
  author={Zang, Jianxiang and Ning, Meiling and Wei, Yongda and Dou, Shihan and Zhang, Jiazheng and Mo, Nijia and Li, Binhong and Gui, Tao and Zhang, Qi and Huang, Xuanjing},
  journal={arXiv preprint arXiv:2505.17793},
  year={2025}
}
