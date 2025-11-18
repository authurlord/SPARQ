# ✨SPARQ: A Cost-Efficient Framework for Offline Table Question Answering via Adaptive Routing
SPARQ is an advanced framework that significantly reduces the cost of Offline Table Question Answering (TQA) tasks while maintaining high performance, achieved through an Adaptive Routing mechanism.


## 📄Paper and Full Version

This repository contains the official implementation of our paper, SPARQ: A Cost-Efficient Framework for Offline Table Question Answering via Adaptive Routing.
Full paper link: [SPARQ_full_version](sparq_full_ver.pdf).


## Setup

We recommend using Python 3.10 or higher to run SPARQ.

1. Create Environment: You can use `uv` or `conda` to create an isolated Python environment:
```Bash
# Conda example
conda create -n sparq_env python=3.10
conda activate sparq_env
```

2. Install dependencies:
```Bash
pip install -r requirements.txt
```

## Model Checkpoint & Datasets.

### Model checkpoints
To run the pipeline, you need four models: an LLM (Qwen series), an embedding model, and two fine-tuned components: Router E and Verifier Q.

<!-- ### Model Checkpoint
To guarantee the reproducibility of our code, please download our fine-tuned checkpoint for router E/verifier Q from [link](https://drive.google.com/file/d/1AMlhBFiaQsu3i_yJrLyMmOIpZoYYfepf/view?usp=drive_link). Then unzip it. E.g. in `model/router/wikitq` stored the router model for dataset `wikitq`, and `model/router/tab_fact` for dataset `tab_fact`. -->

The LLM model and embedding model can be download from Huggingface or ModelScope.
- [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Qwen3-30B-A3B-Instruct-2507-FP8](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8) or [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [bge-m3](https://huggingface.co/BAAI/bge-m3)

As for the fine-tuned models, please download checkpoints from [model_link](https://drive.google.com/file/d/1AMlhBFiaQsu3i_yJrLyMmOIpZoYYfepf/view?usp=drive_link).

After downloading the required models, place them in the `model/` directory or any location you prefer.

### Datasets Download
The required datasets will typically be downloaded automatically by the code execution.

We have preprocessed the results of the Query Rewriting step, which can be found in:
- Rewritten queries path: `schedule_pipeline/datasets/schedule_test/`

Note: The offline rewriting script will be uploaded later.


## Execution pipeline

We provide an execution script to run the end-to-end SPARQ routing inference pipeline.

Run the following command to start testing:

```Bash
./test_pipeline.sh
```
Parameter notes:
- Use the `first_n` parameter to limit the number of samples; `-1` processes the entire dataset.
- Update the model path variables in the script to match your local locations.


## 🙏Acknowledgements

This implementation is based on and has greatly benefited from the following excellent works. Thank you to the authors for releasing their code and research.
- H-STAR: LLM-driven Hybrid SQL-Text Adaptive Reasoning on Tables. [paper_link](https://arxiv.org/abs/2407.05952)
- TabSQLify: Enhancing Reasoning Capabilities of LLMs Through Table Decompositio. [paper_link](https://arxiv.org/abs/2404.10150)


