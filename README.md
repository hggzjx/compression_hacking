



<div align="center">
  <h1>Compression Hacking: A Supplementary Perspective on Informatics Metric of Language Models from Geometric Distortion</h1>
  <br />
  <span style="color:red">📢 <strong><i>If you are interested in our work, please star ⭐ our project.</i></strong></span>

  <h4>
    <a href="https://arxiv.org/abs/2505.17793"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:Compression-hacking&color=red&logo=arxiv"></a>
    <img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">
  </h4>
</div>


## 🌈 Introduction


Recently, the concept of "compression as intelligence" has provided a novel informatics metric perspective for language models (LMs), emphasizing that highly structured representations signify the intelligence level of LMs. However, from a geometric standpoint, the representation space of highly compressed LMs tends to degenerate into a highly anisotropic state, which hinders the LM's ability to comprehend instructions and directly impacts its performance. We found this compression-anisotropy synchronicity is essentially the **_Compression Hacking_** in LM representations, where **noise-dominated directions tend to create the illusion of high compression rates by sacrificing spatial uniformity**.
Based on this, we propose three refined compression metrics by incorporating geometric distortion analysis and integrate them into a self-evaluation pipeline. The refined metrics exhibit strong alignment with the LM's comprehensive capabilities, achieving Spearman correlation coefficients above 0.9, significantly outperforming both the original compression and other internal structure-based metrics. This confirms that compression hacking substantially enhances the informatics interpretation of LMs by incorporating geometric distortion of representations.

We packaged the evaluation pipeline into an **easily callable tool** as shown below, enabling users to conveniently evaluate models using our three proposed metrics.


## 😍 Usage

First, you need to install the package. You can install it by running the following command:

```bash
git clone https://github.com/hggzjx/compression_hacking.git
cd compression_hacking
# pip install .
pip install -e . # install in editable mode
```

Then, you can directly use the following evaluation script (Python) for assessment, or simply run the demo.py file.

```python
import os
from prettytable import PrettyTable
from inf_evaluator import InfEvaluator

# --- 1. Configuration Area ---
# Define the list of model paths to be evaluated
model_list = [
    "facebook/opt-13b",
    "facebook/opt-6.7b",
    "facebook/opt-1.3b"
]

# Define the list of metrics to be evaluated
metric_list = ["compression_se", "semantic_cv", "compression_revised"]

# Define the optimization direction for each metric: True means higher is better, False means lower is better
higher_is_better = {
    "compression_se": True,
    "semantic_cv": False,
    "compression_revised": True
}

# --- 2. Data Evaluation and Collection ---
results_data = {}
print("🚀 Starting evaluation for all models, please wait...")

for model_path in model_list:
    # Automatically extract a clean model name from the full path
    model_name = os.path.basename(model_path)
    print(f"Processing model: {model_name}...")
    
    # Initialize a dictionary to store all metric results for the current model
    results_data[model_name] = {}
    
    # Initialize the evaluator
    evaluator = InfEvaluator(model_path=model_path, sample_size=10, batch_size=16)
    
    for metric in metric_list:
        # Perform the evaluation and get the score
        score = evaluator.evaluate(metric=metric)[metric]
        results_data[model_name][metric] = score
        
print("✅ All model evaluations complete!\n")

# --- 3. Find the Best Model for Each Metric ---
best_models = {}
for metric in metric_list:
    # Extract all models and their scores for the current metric
    scores = [(model, results_data[model][metric]) for model in results_data]
    
    # Find the best model based on the metric's optimization direction
    if higher_is_better[metric]:
        # Find the maximum score
        best_model, _ = max(scores, key=lambda item: item[1])
    else:
        # Find the minimum score
        best_model, _ = min(scores, key=lambda item: item[1])
    best_models[metric] = best_model

# --- 4. Visualize Results Using a Table ---
# Create a table object
table = PrettyTable()
# Set the table headers
table.field_names = ["Model", *metric_list]
# Set column alignment
table.align["Model"] = "l"
for metric in metric_list:
    table.align[metric] = "c"

# Populate the table with data
for model_name, metrics in results_data.items():
    row = [model_name]
    for metric in metric_list:
        # Format the number to 4 decimal places
        value_str = f"{metrics[metric]:.4f}"
        # Add a crown emoji if it's the best model for the metric
        if model_name == best_models[metric]:
            value_str += " 👑"
        row.append(value_str)
    table.add_row(row)

print("📊 Detailed Model Performance Report")
print(table)
print("\nNotes:")
print("  - 👑: Indicates the best performing model for the metric.")
print("  - `compression_revised` and `compression_se`: Higher is better.")
print("  - `semantic_cv`: Lower is better.\n")

# --- 5. Generate the Final Summary Report ---
print("📝 Summary Report for Each Metric")
for metric in metric_list:
    # Sort models based on their performance for the current metric
    sorted_models = sorted(
        results_data.keys(),
        key=lambda m: results_data[m][metric],
        reverse=higher_is_better[metric] # Sort in descending/ascending order based on the optimization direction
    )
    
    comparison_str = " > ".join(sorted_models)
    best_model_name = best_models[metric]
    
    print(f"🔹 **{metric}**:")
    print(f"   - **Best Model**: **{best_model_name}**")
    print(f"   - **Performance Ranking**: {comparison_str}\n")
```



## 📝License
Distributed under the Apache-2.0 License. See LICENSE for more information.




## 📖Citation

if you find this work helpful, please cite it as:

```
@article{zang2025compression,
  title={Compression Hacking: A Supplementary Perspective on Informatics Metric of Language Models from Geometric Distortion},
  author={Zang, Jianxiang and Ning, Meiling and Wei, Yongda and Dou, Shihan and Zhang, Jiazheng and Mo, Nijia and Li, Binhong and Gui, Tao and Zhang, Qi and Huang, Xuanjing},
  journal={arXiv preprint arXiv:2505.17793},
  year={2025}
}
