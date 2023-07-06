import spacy
import torch
import torchtext
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
from collections import Counter

spacy_ger = spacy.load("de_core_news_sm")
spacy_eng = spacy.load("en_core_web_sm")

def tokenizer_german(text):
    return [token.text.lower() for token in spacy_ger.tokenizer(text)]

def tokenizer_english(text):
    return [token.text.lower() for token in spacy_eng.tokenizer(text)]

def preprocess_sentence(sentence, tokenizer):
    return ['<sos>'] + tokenizer(sentence) + ['<eos>']

# Define a custom collate function to handle padding and batching
def collate_fn(batch):
    src_batch, trg_batch = zip(*batch)
    src_lengths = [len(src) for src in src_batch]
    trg_lengths = [len(trg) for trg in trg_batch]

    # Pad sequences to the maximum length in the batch
    src_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(src) for src in src_batch], padding_value=de_pad_idx)
    trg_padded = torch.nn.utils.rnn.pad_sequence([torch.tensor(trg) for trg in trg_batch], padding_value=en_pad_idx)

    return src_padded, trg_padded, src_lengths, trg_lengths

# Load the Multi30k dataset
train_data, valid_data, test_data = Multi30k(split=('train', 'valid', 'test'), language_pair=('de', 'en'))
combined_data = train_data + valid_data + test_data

# Define special tokens
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

# Build vocabulary
german_vocab = torchtext.vocab.build_vocab_from_iterator((tokenizer_german(sentence) for sentence, _ in combined_data), specials=[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN], min_freq=2, max_tokens=8000)
english_vocab = torchtext.vocab.build_vocab_from_iterator((tokenizer_english(sentence) for _, sentence in combined_data), specials=[PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN], min_freq=2, max_tokens=6000)
print(f"Vocabulary size: German={len(german_vocab)}, English={len(english_vocab)}")

# Get integer indices of the special tokens
de_pad_idx = german_vocab[PAD_TOKEN]
de_sos_idx = german_vocab[SOS_TOKEN]
de_eos_idx = german_vocab[EOS_TOKEN]
de_unk_idx = german_vocab[UNK_TOKEN]
print(f"GERMAN | Index of <pad>: {de_pad_idx}, <sos>: {de_sos_idx}, <eos>: {de_eos_idx}, <unk>: {de_unk_idx}")

en_pad_idx = english_vocab[PAD_TOKEN]
en_sos_idx = english_vocab[SOS_TOKEN]
en_eos_idx = english_vocab[EOS_TOKEN]
en_unk_idx = english_vocab[UNK_TOKEN]
print(f"ENGLISH | Index of <pad>: {de_pad_idx}, <sos>: {de_sos_idx}, <eos>: {de_eos_idx}, <unk>: {de_unk_idx}")

# Preprocess the sentences (prefix and suffix sos, and eos tokens)
train_data_processed = [(preprocess_sentence(example[0], tokenizer_german), preprocess_sentence(example[1], tokenizer_english)) for example in train_data]
valid_data_processed = [(preprocess_sentence(example[0], tokenizer_german), preprocess_sentence(example[1], tokenizer_english)) for example in valid_data]
test_data_processed = [(preprocess_sentence(example[0], tokenizer_german), preprocess_sentence(example[1], tokenizer_english)) for example in test_data]


# train_data_indices = [([german_vocab[token] for token in src], [english_vocab[token] for token in trg]) for src, trg in train_data_processed]

# Replace words with integer indices in the preprocessed data
train_data_indices = [
    ([german_vocab[token] if token in german_vocab else de_unk_idx for token in src], 
     [english_vocab[token] if token in english_vocab else en_unk_idx for token in trg]) 
    for src, trg in train_data_processed
]
valid_data_indices = [
    ([german_vocab[token] if token in german_vocab else de_unk_idx for token in src], 
     [english_vocab[token] if token in english_vocab else en_unk_idx for token in trg]) 
    for src, trg in valid_data_processed
]
test_data_indices = [
    ([german_vocab[token] if token in german_vocab else de_unk_idx for token in src], 
     [english_vocab[token] if token in english_vocab else en_unk_idx for token in trg]) 
    for src, trg in test_data_processed
]

# Batch size
batch_size = 16

# Create DataLoader
train_iterator = DataLoader(train_data_indices, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
valid_iterator = DataLoader(valid_data_indices, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_iterator = DataLoader(test_data_indices, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)


# for src, trg, src_lengths, trg_lengths in train_iterator:
#     break
