"""
eval_aaai_refusal.py

Evaluation script for refusal evaluation in model conversations.
This script checks model outputs for refusals according to policy.
"""

import os
import timeit
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import json
from utils_lang import *
from EvalChains.Refusal import refusal_chain
import time
import warnings
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

warnings.filterwarnings("ignore")

# Load experiment data
conv_dir = "./data/Conversations/"
eval_dir = "./data/EvalsRefusal/"
os.makedirs(eval_dir, exist_ok=True)

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "Change This"
)

client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint="Change This",
    azure_ad_token_provider=token_provider,
)

# Evaluate refusals in model outputs
def main_wrapper(line, eval_chains):
    ind, query, conv = line.split("\t")
    query = json.loads(query)
    conv = json.loads(conv)
    eval_chain = eval_chains["refusal"]
    # print("Eval Chain:", eval_chain)
    if skip_conversation_if_already_done(query, eval_file_name):
        return (None, None, None,None, None)

    start_time = timeit.default_timer()
    score, response, eval_conv_str = evaluate_aaai(conv, eval_chain, client)
    elapsed = timeit.default_timer() - start_time
    print("+++++ Time Elapsed (secs):", round(elapsed, 2))
    print(ind, score, response)
    # Add one second delay
    time.sleep(1)
    return (ind, eval_conv_str, score, response, query['policy'])


eval_chains = {
    "refusal": refusal_chain
}

# Output evaluation results
for dataset_name in os.listdir(conv_dir):
    if dataset_name.endswith(".tsv"):
        eval_file_name = eval_dir + dataset_name
        print("-" * 50)
        print(dataset_name)
        print("-" * 50)

        with open(
            eval_file_name, "a+", encoding="utf-8", errors="surrogatepass"
        ) as eval_file:

            with open(conv_dir + dataset_name) as f:
                lines = f.readlines()
                # print(lines[0])
                # print(main_wrapper(lines[0], eval_prompt))
                with ThreadPool(2) as pool:
                    for ind, eval_conv_str, score, response, query_policy in tqdm(
                        pool.imap_unordered(
                            lambda line: main_wrapper(line, eval_chains), lines
                        ),
                        total=len(lines),
                    ):
                        if ind is not None:
                            eval_file.write(
                                f"{ind}\t{score}\t{response}\t{query_policy}\t{eval_conv_str}\n"
                            )
