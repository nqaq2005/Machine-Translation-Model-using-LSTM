import torch

def attention_scores(output_encoder: torch.tensor, output_decoder: torch.tensor):
    # output_encoder: batch x seq x dim
    # hidden_state_decoder: batch x dim
    hidden_state_decoder = output_decoder.unsqueeze(-1)  # ->  batch x dim x 1
    return torch.bmm(output_encoder, hidden_state_decoder).squeeze(-1)  # -> batch x seq


def attention_weight(attention_scores: torch.tensor, mask: torch.tensor):
    min_value = torch.finfo(attention_scores.dtype).min
    weight = attention_scores.masked_fill(mask, value=min_value)
    return torch.softmax(weight, dim=1)


def attention_output(attention_weight: torch.tensor, output_encoder: torch.tensor):
    # attention_weight: batch x seq
    attention_weight = attention_weight.unsqueeze(-1)  # -> batch x seq x 1
    ct = output_encoder * attention_weight  # -> batch x seq x dim
    ct = torch.sum(ct, dim=1)  # -> batch x dim
    return ct


def attention(output_encoder, output_decoder, mask):
    score   = attention_scores(output_encoder, output_decoder)
    weights = attention_weight(score, mask)
    output  = attention_output(weights, output_encoder)
    return output
