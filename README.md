# Overview of SemDisc
This repository contains the code for `Qualitative Join Discovery in Data Lakes using Examples (SIGMOD 2026)`.

SemDisc is an end-to-end join discovery system that supports a QbE (Query-by-Example) interface for join discovery. SemDisc is divided into two phases. In the offline processing stage- SemDisc builds the indexes including the join graph and the Inverted Join Path Index. In the online phase, SemDisc takes a query table as input and outputs a joined table by searching for the optimal join path from the data lake.

We have tested the code on a CHPC machine with a linux OS, 400 GB RAM, 40 2.75 GHz AMD EPYC 9454 CPU cores, and an NVIDIA A800 GPU. We will update the code to run on MacOS and Windows soon. The code ships with data lake and queries of DrugCentral for testing initial setups. You will need an OpenAI-compatible API key (OpenAI directly, or an OpenRouter key for free-tier testing) in order to run the code.

# Setup and Testing

## Local (Windows/WSL)
```
conda create --prefix ./semdiscenv python=3.9
conda activate ./semdiscenv
pip install -r requirements.txt
```
Create `./custom_lib/dotenv/.env`:
```
OPENAI_API_KEY=<your_key>
```
* OpenAI key (`sk-proj-...`) works as-is. For OpenRouter free-tier testing, use an OpenRouter key (`sk-or-v1-...`), set `base_url="https://openrouter.ai/api/v1"` on the `ChatOpenAI(...)` calls in `custom_lib/llm_interface.py`, and use a real model slug (e.g. `"meta-llama/llama-3.1-8b-instruct:free"`) — not the placeholder `"openrouter/free"`.
* `--pcount` defaults to 32 in `custom_lib/parallel.py`, which can overload a laptop — pass `--pcount 4` or `--parallel False` locally.
* PyCharm debugging: Run/Debug config with **Module name** `semdisc.stages.offline_stage`, args e.g. `--startstep 3 --endstep 12 --datalake drugcentral --pcount 2 --parallel False`. If the env is in WSL, add interpreter via **On WSL...** → `<repo>/semdiscenv/bin/python`. `--parallel False` is required for breakpoints to hit (debugger doesn't attach to `multiprocessing.Pool` workers).

## Server (CHPC / shared Linux)
```
conda create --prefix ./semdiscenv python=3.9
conda activate ./semdiscenv
pip install -r requirements.txt
```
Same `.env` setup. Higher `--pcount` is fine (originally run with ~40 cores) but leave headroom for other users. Use `tmux`/`screen` or the job scheduler for long runs so they survive disconnects.

## Testing
```
python -m semdisc.stages.offline_stage --startstep 3 --endstep 12 --datalake drugcentral
python -m semdisc.stages.online_stage --datalake drugcentral
```
Expected output:
```
joined table shape:  (685, 9)
Precision@5 for finding all examples: 1.0
```

## Flaky/free-tier LLM notes
Free-tier OpenRouter models intermittently return empty or `finish_reason: 'error'` responses. `call_gpt()` in `custom_lib/llm_interface.py` retries 5x with backoff before hard-exiting via `utils.crash_code` (`os._exit(1)`, not a normal exception).

If you hit `ValueError: No '[' found in the input string.` instead of a `crash_code` exit, that's a **poisoned cache entry** from before the retry logic existed — `llm_cache_load()` serves stale empty results without going through `call_gpt()`/retries. Fix:
```
rm -rf custom_lib/llm_outputs/cache
```

# Handling New Data Lakes
* Create new directory `./data/all_datalakes/<new_datalake_name>/uploaded_datasets` and put all data lake CSV files in that directory.
* Create new file `./data/all_datalakes/<new_datalake_name>/query_tables.json` and create as many query tables as you need with the similar structure of `./query_tables.json` file in root folder.
* Set hyperparameters in `./semdisc/stages/datalake_config.py`. Adjust hyperparameters if needed.
```
    "<new_datalake_name>":{
        'simhash_size': 18,
        'join_edge_threshold': 0.9,
        'semantic_type_similarity_threshold': 0.9,
        'number_of_hashes_for_minhash': 128,
        'diversity_multiplier_threshold': 0.4,
        'simple_path_max_length': 5,
        'normalize_embeddings': True,
        'diversity_enabled': True
    }
```
* Run `python -m semdisc.stages.offline_stage --startstep 2 --endstep 12 --datalake <new_datalake_name>`
* Run `python -m semdisc.stages.online_stage --datalake <new_datalake_name>` to find join paths for the query tables you specified.

