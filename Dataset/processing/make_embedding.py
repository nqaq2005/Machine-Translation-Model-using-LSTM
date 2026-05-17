import fasttext
from make_vocab import make_vocab
import torch
import torch.nn as nn

def make_embedding(path_data, vi_fasttext_path, en_fasttext_path):
    ft_vi = fasttext.load_model(vi_fasttext_path)
    ft_en = fasttext.load_model(en_fasttext_path)

    word2idx_en, idx2word_en, word2idx_vi, idx2word_vi = make_vocab(path_data)

    len_en_words = len(word2idx_en)
    len_vi_words = len(word2idx_vi)

    vi_embed = torch.zeros(size=(len_vi_words, 300))
    en_embed = torch.zeros(size=(len_en_words, 300))

    for key, value in word2idx_vi.items():
        if value < 4:
            vector_special_token = torch.randn(300) * 0.1
            vi_embed[value,:]=vector_special_token

        else:
            vector_word = ft_vi.get_word_vector(key)
            vi_embed[value,:] = vector_word

    for key, value in word2idx_en.items():
        if value < 4:
            vector_special_token = torch.randn(300) * 0.1
            en_embed[value,:]=vector_special_token

        else:
            vector_word = ft_en.get_word_vector(key)
            en_embed[value,:] = vector_word

    vi_embed = nn.Embedding.from_pretrained(vi_embed)
    en_embed = nn.Embedding.from_pretrained(en_embed)

    vietnamese = {
        "word2idx": word2idx_vi,
        "idx2word": idx2word_vi,
        "embedding": vi_embed.weight.data
    }
    english ={
        "word2idx": word2idx_en,
        "idx2word": idx2word_en,
        "embedding": en_embed.weight.data
    }

    torch.save(vietnamese, 'vi_vocab_embeddings.pt')
    torch.save(english, 'en_vocab_embeddings.pt')

    print("Vocabulary saved successfully!")
