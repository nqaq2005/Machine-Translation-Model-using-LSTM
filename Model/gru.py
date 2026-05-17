import torch.nn as nn
import torch
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from MachineTranslation.Model.attention_mechanism import attention

class GRU_encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, device,
                 dropout:float, vi_embedding: nn.Embedding, en_embedding: nn.Embedding):

        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, device=device, dropout=dropout, batch_first=True)
        self.vi_embedding = vi_embedding
        self.en_embedding = en_embedding

    def forward(self, Input: torch.tensor, lengths: torch.tensor, vi2en: bool):
        # Input: batch x seq
        if vi2en:
            embedded = self.vi_embedding(Input)#-> bach x seq x dim
        else:
            embedded = self.en_embedding(Input)#-> bach x seq x dim

        packed_embedded = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        packed_output, h_n = self.gru(packed_embedded)
        # packed_output: batch x seq x hidden_dim
        # h_n:    num_layer x batch x hidden_dim
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        return output, h_n


class GRU_decoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, device,
                 dropout:float, vi_embedding: nn.Embedding, en_embedding: nn.Embedding):

        super().__init__()
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim,
                          num_layers=num_layers, device=device, dropout=dropout, batch_first=True)
        self.vi_embedding = vi_embedding
        self.en_embedding = en_embedding

    def forward(self, Input: torch.tensor, h_0: torch.tensor, vi2en: bool):
        # h_0: num_layer x batch x hidden_dim
        # Input: batch x seq

        if vi2en:
            Input = self.en_embedding(Input)#-> batch x seq x dim
        else:
            Input = self.vi_embedding(Input)#-> batch x seq x dim

        output, h_n = self.gru(Input, h_0)
        # output: batch x seq x hidden_dim
        # h_n:    num_layer x batch x hidden_dim

        return output, h_n

class OutputLayer(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()

        # --- BLOCK 1: Initial Bottleneck ---
        # Compresses the concatenated [ht, ct] vector from 2x down to 1x
        self.layer1 = nn.Linear(in_features=hidden_dim * 2, out_features=hidden_dim, bias=False)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(p=dropout)

        # --- BLOCK 2: Additional Deep Layer ---
        # Maintains the hidden_dim size but adds an extra step of non-linear reasoning
        self.layer2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim, bias=False)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(p=dropout)

        # --- BLOCK 3: Final Classification ---
        # Projects the deeply processed features into the vocabulary space
        self.classifier = nn.Linear(in_features=hidden_dim, out_features=vocab_size, bias=True)

    def forward(self, x):
        # x expected shape: (batch_size, hidden_dim * 2)

        # Block 1
        x = self.layer1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.drop1(x)  # -> (batch_size, hidden_dim)

        # Block 2
        x = self.layer2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.drop2(x)  # -> (batch_size, hidden_dim)

        # Classification
        logits = self.classifier(x)  # -> (batch_size, vocab_size)

        return logits

class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module,
                 OutputLayer_vi: nn.Module, OutputLayer_en: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.outputlayer_vi = OutputLayer_vi
        self.outputlayer_en = OutputLayer_en

    def forward(self, src: torch.tensor, src_lengths: torch.tensor,
                target: torch.tensor, vi2en: bool, pad_mask: torch.tensor, teacher_forcing_ratio=0.5):
        #src: batch x seq
        #target: batch x seq
        batch_size = src.size(0)
        seq_len = target.size(1)

        output_encoder, h_n_encoder = self.encoder(src, src_lengths, vi2en)
        # output_encoder: batch x seq x hidden_dim
        # h_n_encoder:    num_layer x batch x hidden_dim

        decoder_input = target[:, 0].unsqueeze(1)  # shape: batch x 1
        h_n_decoder = h_n_encoder

        outputs = []

        for t in range(seq_len):
            output_decoder, h_n_decoder = self.decoder(decoder_input, h_n_decoder, vi2en)

            ht = output_decoder[:,0,:] #-> batch, hidden_dim
            ct = attention(output_encoder=output_encoder, output_decoder=ht, mask=pad_mask)
            # ct: batch x dim

            #combine
            combine = torch.cat([ht, ct], dim=1)

            if vi2en:
                out = self.outputlayer_en(combine)
            else:
                out = self.outputlayer_vi(combine)
            #out: batch x vocab

            teacher_force = random.random() < teacher_forcing_ratio

            # Get the highest predicted token from the model
            top1 = out.argmax(dim=1).unsqueeze(1)  # shape: batch x 1

            # Determine the input for the next time step
            if t < seq_len - 1:  # Prevent out of bounds on the last step
                if teacher_force:
                    decoder_input = target[:, t + 1].unsqueeze(1)  # Use ground truth
                else:
                    decoder_input = top1  # Use model's own prediction

            outputs.append(out)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, vocab)
        outputs = outputs.permute(0, 2, 1) # (batch, vocab, seq)

        return outputs