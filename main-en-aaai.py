"""
main-en-aaai.py

Main script for running multi-turn user simulation experiments in SAGE.
Handles user simulation, model interaction, and data collation.
"""

import os

os.environ["HF_HOME"] = "Change This"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import gc
import torch
from constants import *
from utils_lang import (
    load_human_prompt,
    converse,
    skip_conversation_if_already_done,
    init_conversation,
)
import json
from models import get_model_and_tokenizer
import timeit
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

import warnings

warnings.filterwarnings("ignore")

device = "cuda"
hf_token = "hAdd Your Token Here"

# Initialize user simulation
human_prompt = load_human_prompt(HUMAN_BOTS["5PT-Lang"])

model_list = [
    "Phi-3-mini-4k-instruct",
    'Llama-2-7b-chat-hf',
    'Llama-2-13b-chat-hf',
    'Mixtral-8x7B-Instruct-v0.1',
    'Mistral-7B-Instruct-v0.3',
    'gpt-4o',
    'Phi-3-medium-4k-instruct',
]

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), "Change This"
)

client = AzureOpenAI(
    api_version="2024-02-01",
    azure_endpoint="Change This",
    azure_ad_token_provider=token_provider
)

def main_wrapper(line):
    # print(conversation_file_name)
    line = line.strip()
    line = json.loads(line)
    if skip_conversation_if_already_done(line, conversation_file_name):
        return (None, None)

    start_time = timeit.default_timer()
    conversation_init = init_conversation(model_name, dataset_name)
    # code you want to evaluate
    conversation = converse(
        conversation=conversation_init,
        model=model,
        tokenizer=tokenizer,
        user_model=user_model,
        user_tokenizer=user_tokenizer,
        datarow=line,
        human_prompt=human_prompt,
        max_turns=line["max_turns"],
        device=device,
    )
    elapsed = timeit.default_timer() - start_time
    print("+++++ Time Elapsed (mins):", round(elapsed / 60, 2))

    if model_name != "gpt-4o":
        encodeds = tokenizer.apply_chat_template(conversation, return_tensors="pt")
        print("+++++ Num total tokens", len(encodeds[0]))
    
    return line, conversation


user_model, user_tokenizer = get_model_and_tokenizer("Mistral-7B-Instruct-v0.3", "auto")

# Run multi-turn conversations
for model_name in model_list:
    print("-" * 50)
    print(model_name)
    print("-" * 50)
    if model_name == "gpt-4o":
        model, tokenizer = model_name, client 
    else:
        model, tokenizer = get_model_and_tokenizer(model_name, "auto")
    input_dir = "data/Input/"
    for dataset_name in os.listdir(input_dir):
        if dataset_name.endswith(".tsv"):
            conversation_file_name = f"data/Conversations/{model_name}_{dataset_name}"
            with open(
                conversation_file_name, "a+", encoding="utf-8", errors="surrogatepass"
            ) as conversation_file:
                with open(input_dir + dataset_name) as f:
                    lines = f.readlines()
                    # print(lines)
                    with ThreadPool(5) as pool:
                        for line, conversation in tqdm(
                            pool.imap_unordered(main_wrapper, lines), total=len(lines)
                        ):
                            if line:
                                # Save the conversation
                                conversation_file.write(
                                    f"{line['index']}\t{json.dumps(line)}\t{json.dumps(conversation)}\n"
                                )

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()
