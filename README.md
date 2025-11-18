# SPARQ: A Cost-Efficient Framework for Offline Table Question Answering via Adaptive Routing

## Full Version

This is the official implementation of our paper SPARQ: A Cost-Efficient Framework for Offline Table Question Answering via Adaptive Routing. Full version of our paper is in [link](sparq_full_ver.pdf).

## Model Checkpoint

To guarantee the reproducibility of our code, please download our fine-tuned checkpoint for router E/verifier Q from [link](https://drive.google.com/file/d/1AMlhBFiaQsu3i_yJrLyMmOIpZoYYfepf/view?usp=drive_link). Then unzip it. E.g. in `model/router/wikitq` stored the router model for dataset `wikitq`, and `model/router/tab_fact` for dataset `tab_fact`.


### How-To-Run
1. run `pip install -r requirements.txt` to update openai to latest. It is suggested to create a dependent `sparq` environment.

2. Check `src/path.pth`, modify it to your own `SPARQ` path, and copy it to your conda site-packages path, e.g. `/home/wys/anaconda3/envs/sparq/lib/python3.9/site-packages/path.pth`. Then enter the path of your own `SPARQ` folder.

3. Enter conda environment `sparq`, then initialize the local vllm server with command `sh llm_server.sh`. We initialize the server with 2 RTX 4090 GPUs, with `flashinfer` backend. Default we use 4B model from [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507)/[HuggingFace](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), and 30B model from [ModelScope](https://modelscope.cn/models/Qwen/Qwen3-VL-30B-A3B-Instruct-FP8)/[HuggingFace](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507-FP8). Download model, and replace `MODEL_PATH` to your model path. If you want to initialize 30B model, please consider to modify `llm_server.sh` with additional args `--enable-expert-parallel`. The embedding model we use is bge-m3, which can be downloaded from [ModelScope](https://modelscope.cn/models/BAAI/bge-m3)/[HuggingFace](https://huggingface.co/BAAI/bge-m3).

4. Enter folder `schedule_pipeline`, and run `sh test_pipeline_api.sh` to start the pipeline, which conduct the framework automatically, including data-preprocess/route/check/conduct/evaluate. Replace `model_name` with your downloaded LLM above, `embedding_model_path` with the embedding model, `router_model_path` with `model/router/dataset_name`, and `check_model_path` with `model/check/dataset_name`.(dataset_name in 'wikitq','tab_fact'). All intermediate result is stored in `tmp_save_path`.

### Acknowledge

This implementation is based on [H-STAR: LLM-driven Hybrid SQL-Text Adaptive Reasoning on Tables](https://arxiv.org/abs/2407.05952). The work has also benefitted from [TabSQLify: Enhancing Reasoning Capabilities of LLMs Through Table Decomposition](https://arxiv.org/abs/2404.10150). Thanks to the author for releasing the code.


