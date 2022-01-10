"""ML pipeline

Author: Dan Sun
Date: 2022-01-07
"""
import argparse
import logging
import src.basic_cleaning as bc
import src.model_training as mt
import src.model_inference as mi


def execute_pipeline(args):
    """Execute machine learning pipeline
    """
    logging.basicConfig(level=logging.INFO)

    if (args.action == "combo" or args.action == "basic_cleaning"):
        logging.info("Start basic data cleaning ...")
        bc.execute()

    if (args.action == "combo" or args.action == "training"):
        logging.info("Model training procedure start ...")
        mt.execute()

    if (args.action == "combo" or args.action == "inference"):
        logging.info("Model inference procedure start ...")
        mi.execute()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="ML pipeline")

    parser.add_argument(
        "--action",
        type=str,
        choices=["basic_cleaning", "train_test_model", "inference", "combo"],
        default="combo",
        help="Pipeline action")

    args = parser.parse_args()

    execute_pipeline(args)
