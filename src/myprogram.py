#!/usr/bin/env python
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# Set seed for reproducibility
random.seed(0)
torch.manual_seed(0)

class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.data = [self.char_to_idx[c] for c in text]
        self.seq_len = seq_len
        self.num_samples = len(self.data) - seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[-1], dtype=torch.long)
        return x, y

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        # We only care about the last output for next-char prediction
        last_output = output[:, -1, :] 
        logits = self.fc(last_output)
        return logits

class MyModel:
    def __init__(self, vocab_size=None, char_to_idx=None, idx_to_char=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hyperparameters
        self.seq_len = 20
        self.embedding_dim = 64
        self.hidden_dim = 128
        self.batch_size = 64
        self.epochs = 5
        self.lr = 0.001

        if vocab_size:
            self.model = SimpleLSTM(vocab_size, self.embedding_dim, self.hidden_dim).to(self.device)
            self.char_to_idx = char_to_idx
            self.idx_to_char = idx_to_char
        else:
            self.model = None

    @classmethod
    def load_training_data(cls):
        # Using a small built-in corpus for demonstration. 
        # In a real scenario, use the dataset provided or a larger corpus.
        # Here we attempt to download Shakespeare if available, otherwise use dummy text.
        try:
            from datasets import load_dataset
            dataset = load_dataset('tiny_shakespeare')
            text = dataset['train']['text'][0]
            # Reduce size for speed in this demo if needed
            text = text[:100000] 
        except Exception:
            print("Could not load external dataset, falling back to dummy text.")
            text = "Hello world! This is a test dataset for character prediction. " * 500
        
        return text

    @classmethod
    def load_test_data(cls, fname):
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # remove newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, text, work_dir):
        # Prepare data
        dataset = CharDataset(text, self.seq_len)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model = SimpleLSTM(dataset.vocab_size, self.embedding_dim, self.hidden_dim).to(self.device)
        self.char_to_idx = dataset.char_to_idx
        self.idx_to_char = dataset.idx_to_char
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        self.model.train()
        print(f"Starting training on {self.device}...")
        for epoch in range(self.epochs):
            total_loss = 0
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{self.epochs}, Loss: {total_loss/len(dataloader):.4f}")

    def save(self, work_dir):
        # Save model state and vocabulary maps
        checkpoint = {
            'model_state': self.model.state_dict(),
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': len(self.char_to_idx)
        }
        torch.save(checkpoint, os.path.join(work_dir, 'model.checkpoint'))

    @classmethod
    def load(cls, work_dir):
        path = os.path.join(work_dir, 'model.checkpoint')
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found at {path}")
            
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            char_to_idx=checkpoint['char_to_idx'],
            idx_to_char=checkpoint['idx_to_char']
        )
        model.model.load_state_dict(checkpoint['model_state'])
        model.model.eval()
        return model

    def run_pred(self, data):
        self.model.eval()
        preds = []
        for inp in data:
            # Preprocess input: last seq_len chars, map to indices
            # Handle unknown characters by skipping or mapping to a known char?
            # Ideally model handles UNK, here we just filter for known chars.
            valid_inp = [c for c in inp if c in self.char_to_idx]
            
            if len(valid_inp) == 0:
                # Fallback if no valid context
                # Predict top 3 frequent chars (usually ' ', 'e', 't')
                preds.append(" et") 
                continue

            # Take last self.seq_len chars
            snippet = valid_inp[-self.seq_len:]
            x_indices = [self.char_to_idx[c] for c in snippet]
            x_tensor = torch.tensor([x_indices], dtype=torch.long).to(self.device)
            
            with torch.no_grad():
                logits = self.model(x_tensor)
                # Get top 3 indices
                top_k = torch.topk(logits, 3, dim=1)
                indices = top_k.indices[0].tolist()
                
            pred_chars = "".join([self.idx_to_char[i] for i in indices])
            preds.append(pred_chars)
            
        return preds

if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instantiating model')
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