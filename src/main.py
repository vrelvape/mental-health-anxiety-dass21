"""
Main orchestration module for the Mental Health & Anxiety – DASS-21 project.

This module will eventually:
- initialise directories,
- load and clean the raw data,
- run descriptive analyses and visualisations,
- train and compare machine learning models,
- export figures and tables to the `results/` folder.

For now, it only contains a minimal `main()` function so that the project
has a clear entry point that already runs without errors.
"""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def main() -> None:
    """
    Entry point for the full analysis pipeline.

    At this stage, this function is a placeholder. It will be extended
    step by step to:

    - configure logging,
    - run the data loading and cleaning pipeline,
    - trigger the different analysis and modelling steps.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
    )

    logger.info("Mental Health & Anxiety – DASS-21 pipeline initialised.")
    logger.info("Main pipeline logic will be added in subsequent commits.")