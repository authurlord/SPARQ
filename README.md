# HybridRAG

## Tutorial
Pleace check `Tutorials` folder for basic RAG tutorials and operators. 

## Baselines

Curretly we are working on [H-STAR](https://github.com/nikhilsab/H-STAR). Modified code is in H-STAR folder

### How-To-Run
0. run `pip install -U openai` to update openai to latest. The default `openai==0.28.0` in `H-STAR` is imcapable with vllm and qwen. 

1. Check `src/path.pth`, modify it to your own `HybridRAG` path, and copy it to your conda site-packages path, e.g. `/home/wys/anaconda3/envs/hstar/lib/python3.9/site-packages/path.pth`. Then enter `H-STAR` folder

2. Change LLM: modify `llm_config.yaml` to your own LLM parameters. Note that you need to change `llm_config.yaml` in both `run_gpt.py` and `generation/generator_gpt.py`

3. Run `python run_gpt.py` to generate the results.

4. To run 50 test cases(TabFact contains a total of 11k), it consume 630k tokens. Times per quest is approx. 36s.

### How-To-Run Locally

1. init vLLM server: run 
```
CUDA_VISIBLE_DEVICES=3 vllm serve /home/wys/model/qwen-2.5-7B --api-key api-key-test-qwen-2.5-7B --dtype auto --port 8888 
```
or on 51.10 sever:
```
CUDA_VISIBLE_DEVICES=3 vllm serve /public/qwen-2.5-7B --api-key api-key-test-qwen-2.5-7B --dtype auto --port 8888
```
to init local LLM server. Check [vllm_openai_api](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#supported-apis) for more details.

2. modify `llm_config.yaml` to your own LLM parameters. e.g. the following:
```
model: /home/wys/model/qwen-2.5-7B
api_key: api-key-test-qwen-2.5-7B
base_url: http://0.0.0.0:8888/v1
```
3. run `run_gpt.py`. For offline loading , in single A100 GPU with 7B model, we have an average of `1432 tokens/s` prompt input throughput, and `118.5 tokens/s` prompt output throughput. The average time per case is `100.7s`.

4. add `parallel` num, e.g. 8-32,  can significantly improve the throughput to `4317/150`. However, denote it is still slower than offline batch inference. Maybe we can gather all quest together and run offline batch inference with vllm. (TBD)

### How-To-Evaluate

0. run `evaluate.ipynb`

1. You should also pay attention to `results` folder for the model input and output prompt. We can start with `model_gpt_qwen2.5_7B` for an case study. (TBD)

2. TabFact test is run in the default of small test set, in total of 2k. Check `H-STAR/scripts/model_gpt/col_sql.py` line 145 for detail. 