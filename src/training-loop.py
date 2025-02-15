import torch
from torchtext.datasets import IMDb
from torchtext.data.utils import get_tokenizer

# Load the IMDb dataset
train_iter = IMDb(split='train')

# Define a tokenizer (we'll use a simple whitespace tokenizer for now)
tokenizer = get_tokenizer('basic_english')

# Create a vocabulary (mapping words to indices)
from torchtext.vocab import build_vocab_from_iterator
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"]) # set unknown token

# Convert text to numerical data
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: 1 if x == 'pos' else 0 # 1 for positive, 0 for negative