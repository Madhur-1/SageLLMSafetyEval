"""
constants.py

Global constants and configuration for the SAGE framework.
Modify these values to adjust experiment settings and other global options.
"""

HUMAN_RETRIES_PER_TURN = 3  # Number of retries allowed per turn for the human simulator

HUMAN_BOTS = {
    "5PT-Lang": "HumanBots/5PT-Lang.md",  # Path to the user simulator prompt
}