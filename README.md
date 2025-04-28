# SAGE: A Generic Framework for LLM Safety Evaluation

## Overview

SAGE is a framework for generating, simulating, and evaluating conversational data for language model safety research. It supports multi-turn and single-turn experiments, seed generation, user simulation, and automated evaluation of model outputs against safety policies.

---

## Setup

1. **Environment**  
   Create a conda environment and install dependencies:
   ```sh
   conda create -n sage python=3.10
   conda activate sage
   pip install -r requirements.txt
   ```

2. **Configuration**  
   - Add your HuggingFace token and Azure deployment details in the scripts you want to run (see `main-en-aaai.py`, `main-seeds.py`, etc.).
   - Adjust global settings in `constants.py` as needed.

---

## Directory Structure

```
.
├── constants.py                  # Global constants and configuration
├── eval_aaai.py                  # Main evaluation script for defect detection
├── eval_aaai_refusal.py          # Evaluation script for refusal detection
├── main-en-aaai.py               # Main script for multi-turn user simulation
├── main-seeds.py                 # Script for single-turn seed-based simulation
├── models.py                     # Model wrappers and utilities
├── process_seed_to_userbot_input.ipynb # Notebook to process seed data for userbot
├── utils_lang.py                 # Utility functions for language processing
├── data/
│   ├── CollatedData4oRun2.csv            # Data from GPT-4o re-run
│   ├── CollatedDataMain.csv              # Multi-turn experiment data
│   ├── CollatedDataSeedsExperiment.csv   # Single-turn seed experiment data
│   ├── Input/
│   │   ├── fin.tsv                       # Userbot input (finance)
│   │   ├── med.tsv                       # Userbot input (medical)
│   │   └── van.tsv                       # Userbot input (vanilla/general)
│   ├── SeedGenerationInput/
│   │   ├── master.txt                    # Seed generation input file
│   │   └── seedgeneration.md             # Seed generation prompt
│   └── SeedGenerationOut/
│       └── Fin_seeds.csv                 # Generated seeds output
├── EvalChains/
│   ├── Misinformation.py                 # Policy chain for misinformation
│   ├── Refusal.py                        # Policy chain for refusal
│   ├── SexualHarm.py                     # Policy chain for sexual harm
│   └── Violence.py                       # Policy chain for violence
├── HumanBots/
│   └── 5PT-Lang.md                       # User simulator prompt
└── README.md
```

---

## Data

We have collated the data from all the steps in the pipeline in the `/data` directory. The data is organized as follows:
- `CollatedDataMain.csv` - This contains the data from the multi-turn main experiment.
- `CollatedDataSeedsExperiment.csv` - This contains the data from the single-turn seed generation experiment.
- `CollatedData4oRun2.csv` - This contains the data from the re-run of the main experiment (only gpt-4o) for Inter-Run Variance evaluation.

---

## Data Organization

All data artifacts are stored in the `/data` directory:

- **CollatedDataMain.csv**: Multi-turn main experiment results.
- **CollatedDataSeedsExperiment.csv**: Single-turn seed generation experiment results.
- **CollatedData4oRun2.csv**: Results from GPT-4o re-run for inter-run variance.
- **Input/**: Userbot input files for different experiment settings.
- **SeedGenerationInput/**: Files and prompts for seed generation.
- **SeedGenerationOut/**: Output seeds generated for user simulation.

---

## Seed Generation

1. Input Data: `/data/SeedGenerationInput/` - This contains both the seed generation input file (each row is substituted in the prompt iteratively per generation) and prompt.
2. Output Data: `/data/SeedGenerationOutput/` - This contains the generated seeds. `./process_seed_to_userbot_input.ipynb` is used to generat the input to the userbot.

---

## User Simulator

1. Input Data: `/data/Input/` - This contains the userbot input files per setting (output of seed generation).
2. Scripts: `/main-en-aaai.py` - This is the main script that runs the userbot and carries the conversations. It uses the input data from `/data/Input/` and generates the output as `/data/Conversations`. The single-turn experiment is run using `/main-seeds.py` with the output in `/data/ConversationsSeeds`.
3. The User Simulator prompt is present in `/HumanBots/5PT-Lang.md`.

---

## Evaluator
1. Evals are run using `/eval_aaai.py` and `/eval_aaai_refusal.py` for Defect and Refusal respectively. The outputs are stored as `/data/Evals*` and `/data/EvalsRefusal*`.
2. Evaluator Chains with the policies are present in `/EvalChains/`.

---

## Utilities and Configuration

- **Global Settings**: Adjust `constants.py` for experiment-wide configuration.
- **Utility Functions**: Common helpers are in `utils_lang.py`.
- **Model Wrappers**: Model loading and inference logic is in `models.py`.

---

## Extending SAGE

- Add new evaluation policies by creating scripts in `/EvalChains/`.
- Add new user simulation prompts in `/HumanBots/`.
- Integrate new data sources by updating the input files in `/data/Input/` or `/data/SeedGenerationInput/`.

---