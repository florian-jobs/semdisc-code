"""
Worker for the SemDisc baseline of beluga (evaluation/baselines/semdisc_baseline.py in beluga), meant to be run with
SemDisc's own Python interpreter (SemDisc requires Python 3.9, beluga >= 3.10). It imports nothing from beluga.

Invoked as a subprocess by SemDiscBaseline.run_worker(), from this directory, since SemDisc resolves its data lakes and
LLM cache relative to the working directory. SemDiscBaseline has already staged the corpus under
data/all_datalakes/<datalake>/uploaded_datasets.

LLM: SemDisc's own client (custom_lib/llm_interface.py, an OpenAI-compatible LangChain ChatOpenAI hardcoded to
OpenRouter) is pointed at the same provider, model and temperature as beluga (llm_provider, llm_id, llm_temperature
in beluga's configuration), using the same API key environment variable as beluga.

1. Offline: SemDisc's offline stage (semdisc/stages/offline_stage.py) on the staged data lake. Step 3 (semantic types)
   is run here, with SemDisc's own function (LLM) or the column names (smoke tests); steps 4 to 12 are SemDisc's
   offline_stage.main() unchanged, with the hyperparameters of the data lake passed by the wrapper. Steps 0-2
   (sanitize) are replaced by the wrapper's staging. Each part is redone only when its input changes.
2. Online: SemDisc's online stage (semdisc/stages/online_stage.py, loop body of main()) for one query table: the
   query columns are the join column and the features of the base table, the example tuples are rows of it. SemDisc
   detects the lake columns matching every example of each query column, searches join paths connecting them and
   ranks them by the number of example tuples found in the joined table. Deviations: lake columns of excluded tables
   (the base table) are dropped, and the candidate columns per query column are capped when their Cartesian product
   exceeds max_candidate_combinations (SemDisc enumerates it in full, which is meant for queries of 2-5 columns).

Usage: python beluga_worker.py --datalake <name> --query <json path> --output <json path> [--pcount N --parallel False]
       python beluga_worker.py --datalake <name> --query <json path> --build-only   (offline phase only, see
       scripts/build_semdisc_index.py in beluga)
"""

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

SEMDISC_DIR = Path(__file__).resolve().parent
STAGE_COMPLETE_FILE = "beluga_stage_complete.json"  # written by SemDiscBaseline.stage_corpus
SEMANTIC_TYPES_COMPLETE_FILE = "beluga_semantic_types_complete.json"
OFFLINE_STAGE_COMPLETE_FILE = "beluga_offline_stage_complete.json"

# OpenAI-compatible endpoints of beluga's LLM providers, with the API key environment variable beluga reads for each
LLM_PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "google": ("https://generativelanguage.googleapis.com/v1beta/openai/", "GOOGLE_API_KEY"),
}

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

    if provider not in LLM_PROVIDERS:
        raise ValueError(f"Unsupported LLM provider {provider!r}: use one of {sorted(LLM_PROVIDERS)}")
    base_url, api_key_variable = LLM_PROVIDERS[provider]
    api_key = os.environ.get(api_key_variable)
    if not api_key:
        raise RuntimeError(f"{api_key_variable} is not set (required for the LLM provider {provider!r}, as in beluga)")
    # SemDisc checks OPENAI_API_KEY before every call (custom_lib/dotenvloader.py)
    os.environ["OPENAI_API_KEY"] = api_key

    # SemDisc reads the model from MODEL_4O at call time, and hardcodes OpenRouter's base_url in its ChatOpenAI calls
    llm_interface.MODEL_4O = model
    chat_openai = llm_interface.ChatOpenAI

    def chat_openai_for_provider(**kwargs):
        kwargs["base_url"] = base_url
        kwargs["api_key"] = api_key
        kwargs["temperature"] = temperature  # SemDisc hardcodes temperature=0 in its semantic type prompt
        return chat_openai(**kwargs)

    llm_interface.ChatOpenAI = chat_openai_for_provider


def file_digest(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def is_up_to_date(marker_path, info):
    return marker_path.exists() and json.loads(marker_path.read_text(encoding="utf-8")) == info


def build_semantic_types(semantic_type_source):
    """Step 3 of SemDisc's offline stage (one LLM call per table, cached by SemDisc under custom_lib/llm_outputs/cache),
    or the column names themselves (no LLM calls, for smoke tests)."""

    all_dataframes = db.load_all_dataframes(fpm.get_uploaded_datasets_path())
    if not all_dataframes:
        raise ValueError(f"No tables staged in {fpm.get_uploaded_datasets_path()}")
    if semantic_type_source == "llm":
        utils.reset_directory(semantic_type_extractor.get_semantic_type_directory())
        semantic_type_extractor.get_and_save_semantic_types(all_dataframes=all_dataframes)
    else:
        semantic_type_extractor.save_all_semantic_types_single_file([
            {"table": table, "column": column, "semantic_type": column}
            for table, dataframe in all_dataframes.items() for column in dataframe.columns
        ])


def run_offline_stage(datalake):
    """Steps 4 to 12 of SemDisc's offline stage, with its own main() (which reads its arguments from the command line)."""

    argv = sys.argv
    sys.argv = [argv[0], "--datalake", datalake, "--startstep", "4", "--endstep", "12"] + [
        x for flag in ("--pcount", "--parallel") if flag in argv for x in (flag, argv[argv.index(flag) + 1])
    ]
    try:
        offline_stage.main()
    finally:
        sys.argv = argv


def build_index(datalake, query):
    """Offline phase (see the module docstring), each part redone only when its input changes."""

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
    query_df = pd.DataFrame({query_column["column"]: query_column["values"] for query_column in query["columns"]})
    semantic_type_extractor.call_gpt_for_semantic_types(all_dataframes={query_table: query_df}, semantic_type_directory=semantic_type_directory)
    # Match the ids returned by the LLM to the column names, as in SemDisc's compute_refined_semantic_types_for_missing_responses
    semantic_type_infos = utils.file_load(f"{semantic_type_directory}/{query_table}.json")
    ids = [str(x["id"]) for x in semantic_type_infos]
    semantic_types = dict()
    for column in columns:
        best_id = utils.best_match_difflib(strings=ids, query=column) if ids else None
        semantic_types[column] = str(semantic_type_infos[ids.index(best_id)]["name"]) if best_id is not None else column

    return semantic_types


def cap_candidate_columns(user_query_with_detected_columns, max_candidate_combinations):
    """Keep the best candidate columns of each query column (most matched examples first, as sorted by SemDisc) so that
    the number of candidate paths (their Cartesian product) stays below max_candidate_combinations."""

    counts = [len(x["top_columns_by_semantic_type"]) for x in user_query_with_detected_columns]
    if math.prod(counts) <= max_candidate_combinations:
        return
    cap = max(1, int(max_candidate_combinations ** (1 / len(counts))))
    print(f"Capping candidate columns per query column to {cap} (candidates: {counts})", flush=True)
    for query_column in user_query_with_detected_columns:
        query_column["top_columns_by_semantic_type"] = query_column["top_columns_by_semantic_type"][:cap]


def find_join_orders(query, semantic_types):
    """SemDisc's online stage for one query table (loop body of semdisc/stages/online_stage.py's main())."""

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
        # FAISS pads the results with -1 when K exceeds the number of indexed columns
        K_semantic_type=min(query["k_semantic_types"], hnsw_index.ntotal),
        search_index=hnsw_index,
        all_semantic_type_infos=all_semantic_type_infos,
        column_value_indices=column_value_indices,
        column_simhash_indices=column_simhash_indices,
        simhash_hyperplanes=simhash_hyperplanes,
        is_column_numeric=is_numeric_of_columns,
        tables_without_path=tables_without_path
    )

    # The base table cannot retrieve itself if it belongs to the corpus
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
            # Lake column detected for each query column, in query order
            "detected_columns": [
                {"query_column": query_column["column"], "table": detected_column["table"], "column": str(detected_column["column"])}
                for query_column, detected_column in zip(query["columns"], detected_columns)
            ],
            "whole_example_row_find_count": int(whole_example_row_find_count)
        })
        if whole_example_row_find_count == example_row_count:
            break

    # Most example tuples found first (SemDisc's criterion), ties in SemDisc's order
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

    build_index(args.datalake, query)

    if args.build_only:
        return

    semantic_types = get_query_semantic_types(query, datalake_dir)
    join_orders = find_join_orders(query, semantic_types)

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump({"semantic_types": semantic_types, "join_orders": join_orders}, file)


if __name__ == "__main__":
    main()
