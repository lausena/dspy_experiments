import dspy
from datasets import load_dataset
from dspy.datasets import DataLoader
from src.logger import logger
import random

def run_rag():
    logger.info('Starting RAG')
    llama70b = dspy.LM('meta-llama/Llama-3.3-70B-Instruct')

    dspy.configure(lm=llama70b)
    # kwargs = dict(fields=("claim", "supporting_facts", "hpqa_id", "num_hops"), input_keys=("claim",))
    # hover = DataLoader().from_huggingface(dataset_name="hover-nlp/hover", split="train", **kwargs)

