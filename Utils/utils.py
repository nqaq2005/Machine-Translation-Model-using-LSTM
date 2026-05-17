import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence

from MachineTranslation.Dataset.loaders.streaming_dataset import streaming_VI_EN_translation
from MachineTranslation.Dataset.loaders.static_dataset import static_VI_EN_translation
from MachineTranslation.Model.gru import Seq2Seq, GRU_encoder, GRU_decoder, OutputLayer

"""Load word2idx, idx2word, Word_Embedding"""
def load_vocab_and_embeddings(path):
    load_dict = torch.load(path)
    embedding = nn.Embedding.from_pretrained(load_dict['embedding'], freeze=False, padding_idx=0)

    def freeze_hook(grad):
        grad_clone = grad.clone()
        grad_clone[4:] = 0
        return grad_clone

    embedding.weight.register_hook(freeze_hook)

    return load_dict['word2idx'], load_dict['idx2word'], embedding

"""Load model"""
def load_models(input_dim: int, hidden_dim: int, num_layers: int, device,
            vi_embedd:nn.Embedding, en_embedd: nn.Embedding, vi_vocab_size: int, en_vocab_size: int,
                dropout:int = 0.1):

    encoder = GRU_encoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, device=device,
                          dropout=dropout, vi_embedding=vi_embedd,en_embedding=en_embedd).to(device)
    decoder = GRU_decoder(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=num_layers, device=device,
                          dropout=dropout, vi_embedding=vi_embedd,en_embedding=en_embedd,).to(device)

    outputLayer_vi = OutputLayer(hidden_dim=hidden_dim, vocab_size=vi_vocab_size).to(device)
    outputLayer_en = OutputLayer(hidden_dim=hidden_dim, vocab_size=en_vocab_size).to(device)

    seq2seq = Seq2Seq(encoder=encoder, decoder=decoder,
                      OutputLayer_vi=outputLayer_vi, OutputLayer_en=outputLayer_en).to(device)

    return encoder, decoder, outputLayer_vi, outputLayer_en, seq2seq

"""Collate function"""
def translation_collate_fn(batch):
    PAD = 0

    encode_vi2en     = []
    src_decode_vi2en = []
    tgt_decode_vi2en = []

    encode_en2vi     = []
    src_decode_en2vi = []
    tgt_decode_en2vi = []

    for sample in batch:
        encode_vi2en.append(sample['vi_en']['encode_vi2en'])
        src_decode_vi2en.append(sample['vi_en']['src_decode_vi2en'])
        tgt_decode_vi2en.append(sample['vi_en']['tgt_decode_vi2en'])

        encode_en2vi.append(sample['en_vi']['encode_en2vi'])
        src_decode_en2vi.append(sample['en_vi']['src_decode_en2vi'])
        tgt_decode_en2vi.append(sample['en_vi']['tgt_decode_en2vi'])

    lengths_vi2en = torch.tensor([len(seq) for seq in encode_vi2en], dtype=torch.int64)
    lengths_en2vi = torch.tensor([len(seq) for seq in encode_en2vi], dtype=torch.int64)

    padding_encode_vi2en     = pad_sequence(encode_vi2en,     padding_value=PAD, batch_first=True)
    padding_src_decode_vi2en = pad_sequence(src_decode_vi2en, padding_value=PAD, batch_first=True)
    padding_tgt_decode_vi2en = pad_sequence(tgt_decode_vi2en, padding_value=PAD, batch_first=True)

    padding_encode_en2vi     = pad_sequence(encode_en2vi,     padding_value=PAD, batch_first=True)
    padding_src_decode_en2vi = pad_sequence(src_decode_en2vi, padding_value=PAD, batch_first=True)
    padding_tgt_decode_en2vi = pad_sequence(tgt_decode_en2vi, padding_value=PAD, batch_first=True)

    return {
        "vi_en": {
            "encode_vi2en"    : padding_encode_vi2en,
            "lengths_vi2en"   : lengths_vi2en,
            "src_decode_vi2en": padding_src_decode_vi2en,
            "tgt_decode_vi2en": padding_tgt_decode_vi2en
        },
        "en_vi": {
            "encode_en2vi"    : padding_encode_en2vi,
            "lengths_en2vi"   : lengths_en2vi,
            "src_decode_en2vi": padding_src_decode_en2vi,
            "tgt_decode_en2vi": padding_tgt_decode_en2vi
        }
    }

"""Load dataset"""
def load_datasets(path_dataset: str, word2id_vi: dict, word2id_en: dict, buffer_size: int,
                  streaming:bool):
    if streaming:
        train_dataset = streaming_VI_EN_translation(path_dataset=path_dataset, word2id_vi=word2id_vi,
                                          word2id_en=word2id_en, buffer_size=buffer_size, split='train')
        valid_dataset = streaming_VI_EN_translation(path_dataset=path_dataset, word2id_vi=word2id_vi,
                                          word2id_en=word2id_en, buffer_size=buffer_size, split='validation')

    else:
        train_dataset = static_VI_EN_translation(path_dataset=path_dataset, word2id_vi=word2id_vi,
                                                    word2id_en=word2id_en, split='train')
        valid_dataset = static_VI_EN_translation(path_dataset=path_dataset, word2id_vi=word2id_vi,
                                                    word2id_en=word2id_en, split='validation')

    return train_dataset, valid_dataset

"""Load optimizer and loss_fn"""
def configure_optimizers(model, lr:int, ignore_idx: int, label_smoothing:int):
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    loss_fn   = nn.CrossEntropyLoss(ignore_index=ignore_idx, label_smoothing=label_smoothing)

    return optimizer, scheduler, loss_fn

"""Load setup"""
def setup_experiment(epochs: int, vi_vocab: list, en_vocab: list, accumulation_steps:  int,
                     run_name="runs/gru_translation"):

    device            = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    writer            = SummaryWriter(run_name)
    vi_vocab_size     = len(vi_vocab)
    en_vocab_size     = len(en_vocab)
    idx_special_token = [0, 1, 2, 3]
    ignore_idx        = 0

    return device, epochs, writer, vi_vocab_size, en_vocab_size, idx_special_token, ignore_idx, accumulation_steps