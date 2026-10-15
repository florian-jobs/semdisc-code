"""
Worker for beluga's SemDisc baseline (evaluation/baselines/semdisc_baseline.py). Runs with SemDisc's own
python (3.9) and doesn't import beluga. semdisc_baseline.py starts it as a subprocess in this directory
after copying the corpus to data/all_datalakes/<datalake>/uploaded_datasets.

The LLM client in custom_lib/llm_interface.py is set to beluga's provider, model and temperature.

Offline: step 3 of offline_stage.py (semantic types, via LLM or just column names), then offline_stage.main()
for steps 4-12. Steps 0-2 are replaced by the staging in semdisc_baseline.py.
Online: loop body of online_stage.py main() for one query table. Query columns are the join column and the
features of the base table. Changes: the base table's own columns are dropped, and candidates per query
column are capped by max_candidate_combinations (SemDisc tries all combinations), and the LLM's semantic type
answers are parsed leniently (parse_semantic_types), with the column name as fallback.

Usage: python beluga_worker.py --datalake <name> --query <json path> --output <json path> [--pcount N --parallel False]
       python beluga_worker.py --datalake <name> --query <json path> --build-only   (offline phase only, see
       scripts/build_semdisc_index.py in beluga)
"""

import argparse
import ast
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

SEMDISC_DIR = Path(__file__).resolve().parent
STAGE_COMPLETE_FILE = "beluga_stage_complete.json"  # written by SemDiscBaseline.stage_corpus
SEMANTIC_TYPES_COMPLETE_FILE = "beluga_semantic_types_complete.json"
OFFLINE_STAGE_COMPLETE_FILE = "beluga_offline_stage_complete.json"

# beluga's LLM providers: (base_url, api key env var)
LLM_PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY"),
}

# local OpenAI-compatible server (ollama, vllm, ...), set LOCAL_LLM_BASE_URL / LOCAL_LLM_API_KEY
LOCAL_LLM_PROVIDER = "local"
LOCAL_LLM_BASE_URL = "http://localhost:11434/v1"

sys.path.insert(0, str(SEMDISC_DIR))
os.chdir(SEMDISC_DIR)

import pandas as pd

from custom_lib import db, llm_interface, utils
from semdisc.algorithm import column_is_numeric, join_dataframes, online_query_table_processor, pairwise_simple_path_index
from semdisc.algorithm import semantic_hash_index, semantic_type_extractor
from semdisc.algorithm.query_builder.build_query import max_cardinality_for_row_eval
from semdisc.lib import file_path_manager as fpm
from semdisc.stages import datalake_config, offline_stage


def configure_llm(provider, model, temperature):
    """Point SemDisc's LLM client at beluga's provider, model and temperature."""

    if provider == LOCAL_LLM_PROVIDER:
        base_url = os.environ.get("LOCAL_LLM_BASE_URL") or LOCAL_LLM_BASE_URL
        api_key = os.environ.get("LOCAL_LLM_API_KEY") or "local"
    elif provider in LLM_PROVIDERS:
        base_url, api_key_variable = LLM_PROVIDERS[provider]
        api_key = os.environ.get(api_key_variable)
        if not api_key:
            raise RuntimeError(f"{api_key_variable} is not set (required for the LLM provider {provider!r}, as in beluga)")
    else:
        raise ValueError(f"Unsupported LLM provider {provider!r}: use one of {sorted(LLM_PROVIDERS) + [LOCAL_LLM_PROVIDER]}")
    # SemDisc checks OPENAI_API_KEY before every call (custom_lib/dotenvloader.py)
    os.environ["OPENAI_API_KEY"] = api_key

    # model comes from MODEL_4O, base_url is hardcoded to openrouter
    llm_interface.MODEL_4O = model
    chat_openai = llm_interface.ChatOpenAI

    def chat_openai_for_provider(**kwargs):
        kwargs["base_url"] = base_url
        kwargs["api_key"] = api_key
        kwargs["temperature"] = temperature  # SemDisc uses temperature=0 otherwise
        return chat_openai(**kwargs)

    llm_interface.ChatOpenAI = chat_openai_for_provider


# How the semantic type answers were parsed (see parse_semantic_types), reported after the build
semantic_type_parsing = {"strict": 0, "repaired": 0, "id_name_pairs": 0, "unusable": 0, "header_fallback_columns": 0}
ID_NAME_PAIR = re.compile(r"""["']id["']\s*:\s*["']?(.*?)["']?\s*,\s*["']name["']\s*:\s*["'](.*?)["']\s*}""", re.S)


def is_semantic_type_list(data):
    return isinstance(data, list) and all(isinstance(x, dict) and "id" in x and "name" in x for x in data)


def parse_semantic_types(text):
    """SemDisc's extract_json_from_string, but tolerant of almost-JSON answers of small models (e.g. Llama 3.1 8B:
    trailing commas, code fences, single quotes), which SemDisc's strict parser rejects, aborting the whole build.
    Returns [] if nothing usable is found (compute_refined_semantic_types then falls back to the column names)."""

    try:
        data = utils.extract_json_from_string(text)
        if is_semantic_type_list(data):
            semantic_type_parsing["strict"] += 1
            return [{"id": str(x["id"]), "name": str(x["name"])} for x in data]
    except ValueError:
        pass
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidate = re.sub(r",\s*([\]}])", r"\1", text[start:end + 1])
        for loads in (json.loads, ast.literal_eval):
            try:
                data = loads(candidate)
            except (ValueError, SyntaxError):
                continue
            if is_semantic_type_list(data):
                semantic_type_parsing["repaired"] += 1
                return [{"id": str(x["id"]), "name": str(x["name"])} for x in data]
    pairs = [{"id": id_.strip(), "name": name.strip()} for id_, name in ID_NAME_PAIR.findall(text)]
    if pairs:
        semantic_type_parsing["id_name_pairs"] += 1
        return pairs
    semantic_type_parsing["unusable"] += 1
    return []


def compute_refined_semantic_types(semantic_type_directory, all_dataframes):
    """SemDisc's compute_refined_semantic_types_for_missing_responses (each column gets the semantic type of the most
    similar id in the answer), but columns no id resembles at all, or tables without a usable answer, get their
    column name as semantic type (as with --semantic-type-source header) instead of crashing."""

    for table, dataframe in all_dataframes.items():
        answers = utils.file_load(f"{semantic_type_directory}/{table}.json")
        ids = [answer["id"] for answer in answers]
        refined_semantic_types = []
        for column in dataframe.columns:
            best_id = utils.best_match_difflib(strings=ids, query=column)
            if best_id is None:
                semantic_type_parsing["header_fallback_columns"] += 1
                refined_semantic_types.append({"column": column, "semantic_type": column, "id_by_gpt": None})
            else:
                refined_semantic_types.append({"column": column, "semantic_type": answers[ids.index(best_id)]["name"], "id_by_gpt": best_id})
        utils.file_dump(refined_semantic_types, f"{semantic_type_directory}/{table}.json")


class LenientSemanticTypeUtils:
    """custom_lib.utils with parse_semantic_types as extract_json_from_string, only for semantic_type_extractor
    (online_query_table_processor parses other answers with the original)."""

    extract_json_from_string = staticmethod(parse_semantic_types)

    def __getattr__(self, name):
        return getattr(utils, name)


def use_lenient_semantic_types():
    semantic_type_extractor.utils = LenientSemanticTypeUtils()
    semantic_type_extractor.compute_refined_semantic_types_for_missing_responses = compute_refined_semantic_types


def file_digest(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def is_up_to_date(marker_path, info):
    return marker_path.exists() and json.loads(marker_path.read_text(encoding="utf-8")) == info


def build_semantic_types(semantic_type_source):
    """Offline step 3: semantic types via LLM (cached), or just the column names for smoke tests."""

    all_dataframes = db.load_all_dataframes(fpm.get_uploaded_datasets_path())
    if not all_dataframes:
        raise ValueError(f"No tables staged in {fpm.get_uploaded_datasets_path()}")
    if semantic_type_source == "llm":
        utils.reset_directory(semantic_type_extractor.get_semantic_type_directory())
        semantic_type_extractor.get_and_save_semantic_types(all_dataframes=all_dataframes)
        print(f"Semantic type answers: {semantic_type_parsing}", flush=True)
        utils.file_dump(semantic_type_parsing, f"{fpm.get_datalake_path()}/beluga_semantic_type_parsing.json")
    else:
        semantic_type_extractor.save_all_semantic_types_single_file([
            {"table": table, "column": column, "semantic_type": column}
            for table, dataframe in all_dataframes.items() for column in dataframe.columns
        ])


def run_offline_stage(datalake):
    """Offline steps 4-12 (offline_stage.main() reads its args from sys.argv)."""

    argv = sys.argv
    sys.argv = [argv[0], "--datalake", datalake, "--startstep", "4", "--endstep", "12"] + [
        x for flag in ("--pcount", "--parallel") if flag in argv for x in (flag, argv[argv.index(flag) + 1])
    ]
    try:
        offline_stage.main()
    finally:
        sys.argv = argv


def build_index(datalake, query):
    """Offline phase, only reruns parts whose input changed."""

    datalake_dir = Path(fpm.get_datalake_path())
    stage_digest = file_digest(datalake_dir / STAGE_COMPLETE_FILE)

    semantic_types_path = datalake_dir / SEMANTIC_TYPES_COMPLETE_FILE
    semantic_types_info = {"stage": stage_digest, "semantic_type_source": query["semantic_type_source"]}
    if query["semantic_type_source"] == "llm":
        semantic_types_info.update({"llm_provider": query["llm_provider"], "llm_id": query["llm_id"], "llm_temperature": query["llm_temperature"]})
    if query["rebuild_index"] or not is_up_to_date(semantic_types_path, semantic_types_info):
        semantic_types_path.unlink(missing_ok=True)
        build_semantic_types(query["semantic_type_source"])
        semantic_types_path.write_text(json.dumps(semantic_types_info), encoding="utf-8")

    offline_stage_path = datalake_dir / OFFLINE_STAGE_COMPLETE_FILE
    offline_stage_info = {
        "stage": stage_digest,
        "semantic_types": file_digest(semantic_type_extractor.get_semantic_types_single_file_path()),
        "datalake_config": query["datalake_config"]
    }
    if query["rebuild_index"] or not is_up_to_date(offline_stage_path, offline_stage_info):
        offline_stage_path.unlink(missing_ok=True)
        run_offline_stage(datalake)
        offline_stage_path.write_text(json.dumps(offline_stage_info), encoding="utf-8")


def get_query_semantic_types(query, datalake_dir):
    """Describe the query columns with SemDisc's own semantic type prompt, or by their names."""

    columns = [query_column["column"] for query_column in query["columns"]]
    if query["semantic_type_source"] == "header":
        return {column: column for column in columns}

    query_table = f"beluga_query__{query['base_table']}"
    semantic_type_directory = f"{datalake_dir}/beluga_query_semantic_types"
    utils.create_directory(semantic_type_directory)
    # columns have different lengths after dropping NaNs, pad them
    query_df = pd.DataFrame({query_column["column"]: pd.Series(query_column["values"], dtype=object) for query_column in query["columns"]})
    semantic_type_extractor.call_gpt_for_semantic_types(all_dataframes={query_table: query_df}, semantic_type_directory=semantic_type_directory)
    # map LLM ids back to column names (like compute_refined_semantic_types_for_missing_responses)
    semantic_type_infos = utils.file_load(f"{semantic_type_directory}/{query_table}.json")
    ids = [str(x["id"]) for x in semantic_type_infos]
    semantic_types = dict()
    for column in columns:
        best_id = utils.best_match_difflib(strings=ids, query=column) if ids else None
        semantic_types[column] = str(semantic_type_infos[ids.index(best_id)]["name"]) if best_id is not None else column

    return semantic_types


def cap_candidate_columns(user_query_with_detected_columns, max_candidate_combinations):
    """Cap candidates per query column so the number of combinations stays below max_candidate_combinations."""

    counts = [len(x["top_columns_by_semantic_type"]) for x in user_query_with_detected_columns]
    if math.prod(counts) <= max_candidate_combinations:
        return
    cap = max(1, int(max_candidate_combinations ** (1 / len(counts))))
    print(f"Capping candidate columns per query column to {cap} (candidates: {counts})", flush=True)
    for query_column in user_query_with_detected_columns:
        query_column["top_columns_by_semantic_type"] = query_column["top_columns_by_semantic_type"][:cap]


def find_join_orders(query, semantic_types):
    """Online stage for one query table (loop body of online_stage.py main())."""

    all_dataframes = db.load_all_dataframes(fpm.get_uploaded_datasets_path())
    all_simhashes = semantic_hash_index.load_all_simhashes()
    all_semantic_type_infos = semantic_type_extractor.load_all_semantic_types_single_file()
    all_paths = pairwise_simple_path_index.load_all_simple_paths()
    table_index_map = pairwise_simple_path_index.load_table_index_map()
    table_to_path_index = pairwise_simple_path_index.load_table_to_path_index()
    all_join_orders = pairwise_simple_path_index.load_all_join_order()
    is_numeric_of_columns = column_is_numeric.load_column_is_numeric()
    hnsw_index = online_query_table_processor.load_semantic_type_hnsw_index()
    simhash_hyperplanes = semantic_hash_index.load_simhash_hyperplanes()

    column_value_indices, column_simhash_indices = online_query_table_processor.compute_column_value_and_simhash_indices_for_all_column(
        all_dataframes=all_dataframes,
        all_simhashes=all_simhashes
    )

    user_query = [
        {"semantic_type": semantic_types[query_column["column"]], "examples": query_column["examples"]}
        for query_column in query["columns"]
    ]
    example_row_count = len(user_query[0]["examples"])

    tables_without_path = pairwise_simple_path_index.compute_tables_without_paths(inverted_index=table_to_path_index, table_index_map=table_index_map)

    user_query_with_detected_columns = online_query_table_processor.get_most_relevant_columns_based_on_search_index_and_example_value_match_count(
        user_query=user_query,
        # faiss returns -1 if K > number of columns
        K_semantic_type=min(query["k_semantic_types"], hnsw_index.ntotal),
        search_index=hnsw_index,
        all_semantic_type_infos=all_semantic_type_infos,
        column_value_indices=column_value_indices,
        column_simhash_indices=column_simhash_indices,
        simhash_hyperplanes=simhash_hyperplanes,
        is_column_numeric=is_numeric_of_columns,
        tables_without_path=tables_without_path
    )

    # skip the base table if it's in the corpus
    exclude_tables = set(query["exclude_tables"])
    for query_column in user_query_with_detected_columns:
        query_column["top_columns_by_semantic_type"] = [x for x in query_column["top_columns_by_semantic_type"] if x["table"] not in exclude_tables]
    candidate_counts = {query_column["column"]: len(x["top_columns_by_semantic_type"]) for query_column, x in zip(query["columns"], user_query_with_detected_columns)}
    print(f"SemDisc candidate columns per query column: {candidate_counts}", flush=True)
    cap_candidate_columns(user_query_with_detected_columns, query["max_candidate_combinations"])

    candidate_path_infos = online_query_table_processor.get_candidate_paths(user_query_with_detected_columns)
    if len(candidate_path_infos) == 0:
        print("At least one query column has no candidate column in the data lake", flush=True)
        return []

    candidate_path_infos = pairwise_simple_path_index.sort_candidate_paths_based_on_row_matches(candidate_path_infos=candidate_path_infos, sort_enabled=True)

    best_join_orders = pairwise_simple_path_index.get_join_orders_from_candidate_paths_using_table_path_inverted_index(
        candidate_path_infos=candidate_path_infos,
        all_paths=all_paths,
        all_table_indices=table_index_map,
        table_to_path_index=table_to_path_index,
        all_join_orders=all_join_orders,
        K_top_paths=query["k_paths"],
    )

    enriched_best_join_orders = []
    for old_join_order_info in best_join_orders:
        join_order_info = utils.deep_copy(old_join_order_info)
        join_order = join_order_info["join_order"]
        detected_columns = join_order_info["detected_columns"]

        tables_ready_for_join = join_dataframes.prepare_tables_with_only_join_columns_with_nan_dropped_for_cardinality_estimation(
            join_order=join_order,
            all_dataframes=all_dataframes,
            all_simhashes=all_simhashes
        )
        table_to_example_match_index_map, _ = online_query_table_processor.get_table_row_indices_from_query_example(
            detected_columns=detected_columns,
            indices_validated_in_single_table_per_query_row=join_order_info["candidate_path_info"]["indices_validated_in_single_table_per_query_row"]
        )
        tables_ready_for_join = online_query_table_processor.trim_tables_ready_for_join_by_row_indices(
            tables_ready_for_join=tables_ready_for_join,
            table_to_example_match_index_map=table_to_example_match_index_map
        )
        cardinality_info = join_dataframes.calculate_true_join_cardinality(tables_ready_for_join=tables_ready_for_join, join_order=join_order)
        if cardinality_info["cardinality"] == 0 or cardinality_info["cardinality"] > max_cardinality_for_row_eval:
            continue

        joined_table = join_dataframes.perform_joins(tables_ready_for_join=tables_ready_for_join, join_order=join_order)["joined_table"]
        whole_example_row_find_count = online_query_table_processor.get_whole_query_row_find_count_by_equi_match(
            joined_table=joined_table,
            detected_columns=detected_columns,
            all_dataframes=all_dataframes,
            all_simhashes=all_simhashes,
            is_numeric_of_columns=is_numeric_of_columns,
            simhash_hyperplanes=simhash_hyperplanes,
        )

        enriched_best_join_orders.append({
            "path": list(join_order_info["path"]),
            "join_order": join_order,
            # matched lake column per query column
            "detected_columns": [
                {"query_column": query_column["column"], "table": detected_column["table"], "column": str(detected_column["column"])}
                for query_column, detected_column in zip(query["columns"], detected_columns)
            ],
            "whole_example_row_find_count": int(whole_example_row_find_count)
        })
        if whole_example_row_find_count == example_row_count:
            break

    # sort by number of example tuples found
    enriched_best_join_orders.sort(key=lambda x: -x["whole_example_row_find_count"])

    return enriched_best_join_orders


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datalake", required=True, help="Name of the data lake under data/all_datalakes")
    parser.add_argument("--query", required=True, help="Path to a JSON file with the query (see SemDiscBaseline.run)")
    parser.add_argument("--output", help="Where to write the ranked join orders, as JSON")
    parser.add_argument("--build-only", action="store_true", help="Only build the index (offline phase), without querying it")
    args, _ = parser.parse_known_args()  # --pcount/--parallel are read by custom_lib.parallel
    if not args.build_only and not args.output:
        parser.error("--output is required unless --build-only is set")

    with open(args.query, "r", encoding="utf-8") as file:
        query = json.load(file)

    fpm.startup(overwrite_datalake_name=args.datalake)
    datalake_dir = Path(fpm.get_datalake_path())
    datalake_config._config[args.datalake] = query["datalake_config"]

    if query["semantic_type_source"] == "llm":
        configure_llm(query["llm_provider"], query["llm_id"], query["llm_temperature"])
        use_lenient_semantic_types()

    build_index(args.datalake, query)

    if args.build_only:
        return

    semantic_types = get_query_semantic_types(query, datalake_dir)
    join_orders = find_join_orders(query, semantic_types)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump({"semantic_types": semantic_types, "join_orders": join_orders}, file)


if __name__ == "__main__":
    main()
