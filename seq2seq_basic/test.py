import spacy
import torch
import torchtext
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k

spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

def tokenizer_german(text):
    return [token.text.lower() for token in spacy_ger.tokenizer(text)]

def tokenizer_english(text):
    return [token.text.lower() for token in spacy_eng.tokenizer(text)]

def preprocess_sentence(sentence, tokenizer):
    return ['<sos>'] + tokenizer(sentence) + ['<eos>']

# Load the Multi30k dataset
train_data, valid_data, test_data = Multi30k(split=('train', 'valid', 'test'), language_pair=('de', 'en'))


# Preprocess the sentences and assign fields
for example in train_data:
    example = (preprocess_sentence(example[0], tokenizer_german), preprocess_sentence(example[1], tokenizer_english))


print("Sentences preprocessed..")

# Build the vocabulary
src_vocab = torchtext.vocab.Vocab()
trg_vocab = torchtext.vocab.Vocab()

src_vocab.build_vocab(train_data, min_freq=2, tokenizer=tokenizer_german)
trg_vocab.build_vocab(train_data, min_freq=2, tokenizer=tokenizer_english)

print("Vocab built..")

# Define collate function for batching
def collate_fn(batch):
    src_batch = [torch.tensor(example.src, dtype=torch.long) for example in batch]
    trg_batch = [torch.tensor(example.trg, dtype=torch.long) for example in batch]
    return src_batch, trg_batch

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=collate_fn)
valid_loader = DataLoader(valid_data, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=collate_fn)
