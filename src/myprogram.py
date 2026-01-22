#!/usr/bin/env python
import os
import string
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from trl import GRPOTrainer
from datasets import load_dataset
from trl.rewards import accuracy_reward
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import wandb
import random
import logging
import re
import torch

# Initialize logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler("nlp_run1.log")  # Output to a file
    ]
)
LOGGER = logging.getLogger(__name__)

# Initialize Weights & Biases
wandb.init(project="CSE447")

# MODEL NAME 
model_name = "Qwen/Qwen3-0.6B"

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    def __init__(self):
        self.trainer = None
    @classmethod
    def load_training_data(cls):
        # Shakespeare next character prediction dataset
        train_dataset = load_dataset("flwrlabs/shakespeare", split="train").shuffle(seed=42)
        return train_dataset['x'], train_dataset['y']

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, data, work_dir):
        train_data = self.load_training_data()
        self.trainer = GRPOTrainer(
            model=model_name,
            reward_funcs=accuracy_reward,
            train_dataset=train_data,
        )
        self.trainer.train()

    def run_pred(self, data):
        # inference
        preds = []
        self.trainer.model.eval()
        for inp in data:
            message = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": "Generate the next three most likely unicode characters to follow the input. The input is not necessarily in English, it may be in any human language."+inp}
            ]
            with torch.no_grad():
                response = self.trainer.model.generate(message, max_new_tokens=16384, temperature=0.6, top_p=0.7, top_k=50,)
                # identify text within the boxed region
                pred_text = re.search(r'\\boxed\{(.*?)\}', response[0]['generation']['content'], re.DOTALL).group(1)
                preds.append(pred_text)
        return preds

    def save(self, work_dir):
        # your code here
        # this particular model has nothing to save, but for demonstration purposes we will save a blank file
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            f.write('dummy save')

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            dummy_save = f.read()
        return MyModel()


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
