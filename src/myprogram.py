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
# wandb.init(project="CSE447")

# MODEL NAME 
model_name = "Qwen/Qwen3-0.6B"

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """
    @classmethod
    def __init__(cls, trainer, tokenizer=None):
        cls.trainer = None
        cls.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @classmethod
    def load_training_data(cls):
        # Shakespeare next character prediction dataset
        train_dataset = load_dataset("flwrlabs/shakespeare", split="train").shuffle(seed=42)
        # map to prompt solution pairs
        prompt = train_dataset['x']
        input_text = cls.tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        solution = train_dataset['y']
        return {'prompt': input_text, 'solution': solution}

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
        # self.trainer.train()

    @classmethod
    def run_pred(cls, data):
        # inference
        preds = []
        cls.trainer.model.eval()
        for inp in data:
            message = [
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\boxed{}."},
                {"role": "user", "content": "Generate the next three most likely unicode characters to follow the input. The input is not necessarily in English, it may be in any human language."+inp}
            ]
            with torch.no_grad():
                response = cls.trainer.model.generate(message, max_new_tokens=16384, temperature=0.6, top_p=0.7, top_k=50,)
                # identify text within the boxed region
                pred_text = re.search(r'\\boxed\{(.*?)\}', response[0]['generation']['content'], re.DOTALL).group(1)
                preds.append(pred_text)
        return preds

    def save(self, work_dir):
        # save model to work_dir from trainer
        self.trainer.model.save_pretrained(work_dir)
        self.tokenizer.save_pretrained(work_dir)

    def load(work_dir):
        # load from work_dir
        # Load a transformers model from the specified directory
        model = AutoModelForCausalLM.from_pretrained(work_dir, device_map='auto', trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(work_dir, trust_remote_code=True)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=accuracy_reward,
        )
        LOGGER.info("Model and tokenizer loaded from {}".format(work_dir))
        ret = MyModel(trainer, tokenizer)
        return ret

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
