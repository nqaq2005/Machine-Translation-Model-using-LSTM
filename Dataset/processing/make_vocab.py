from datasets import load_dataset
import re

def processing_english(text):
    text = text.lower()
    pattern = r"[A-Za-z]+(?:-[A-Za-z]+)*|'s|\d+|[^\w\s]"

    return re.findall(pattern, text)

def processing_vietnamese(text):
    text = text.lower()
    pattern = r"\w+(?:-\w+)*|'\w+|[^\w\s]"

    return re.findall(pattern, text, flags=re.UNICODE)

def tokenize(rows):
    vi_words = []
    en_words = []

    for vi,en in zip(rows['Vietnamese'], rows['English']):

        vi = processing_vietnamese(vi)
        en = processing_english(en)

        vi_words.append(vi)
        en_words.append(en)

    return {
        "vi_words": vi_words,
        "en_words": en_words
    }

def make_vocab(path_data):
    dataset = load_dataset(path_data, split='train', streaming=True)
    dataset = dataset.map(tokenize, batched=True, batch_size=100000,
                          remove_columns=['Vietnamese', 'English', 'From'])

    #### English
    word2idx_en = {
        "<PAD>"   : 0,
        "<Vi2En>" : 1,
        "<EOS_EN>": 2,
        "<UNK_EN>": 3
    }
    idx2word_en = ["<PAD>", "<Vi2En>", "<EOS_EN>", "<UNK_EN>"]

    #### VietNamese
    word2idx_vi = {
        "<PAD>"   : 0,
        "<En2Vi>" : 1,
        "<EOS_VI>": 2,
        "<UNK_VI>": 3
    }
    idx2word_vi = ["<PAD>", "<En2Vi>", "<EOS_VI>", "<UNK_VI>"]

    next_idx_en = 4
    next_idx_vi = 4

    for row in dataset:
        en_words = row['en_words']
        vi_words = row['vi_words']

        for word in en_words:
            if word not in word2idx_en:
                word2idx_en[word] = next_idx_en
                idx2word_en.append(word)
                next_idx_en += 1

        for word in vi_words:
            if word not in word2idx_vi:
                word2idx_vi[word] = next_idx_vi
                idx2word_vi.append(word)
                next_idx_vi += 1

    return word2idx_en, idx2word_en, word2idx_vi, idx2word_vi