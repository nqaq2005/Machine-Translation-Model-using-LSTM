import evaluate
import torch

class Metric:
    def __init__(self, idx2word_vi: list, idx2word_en: list, idx_special_token: list):
        self.list_metric = ["sacrebleu", "chrf"]

        self.bleu_metric = evaluate.load("sacrebleu")
        self.chrf_metric = evaluate.load("chrf")

        self.idx2word_vi = idx2word_vi
        self.idx2word_en = idx2word_en
        self.idx_special_token = idx_special_token

        self.reset()

    def reset(self):
        """Clears the accumulated sentences for a new epoch."""
        self.pred_en_strings = []  # Predictions for Vi -> En
        self.pred_vi_strings = []  # Predictions for En -> Vi
        self.tgt_en_strings = []  # Ground truth for Vi -> En
        self.tgt_vi_strings = []  # Ground truth for En -> Vi

    def add_batch(self, outputs_vi2en: torch.tensor, outputs_en2vi: torch.tensor,
                  tgt_vi2en: torch.tensor, tgt_en2vi: torch.tensor):

        token_en_ids = outputs_vi2en.argmax(dim=1)  # batch x seq
        token_vi_ids = outputs_en2vi.argmax(dim=1)  # batch x seq

        # 1. Decode Predictions
        words_en_pred = [
            [self.idx2word_en[idx.item()] for idx in sentence if idx.item() not in self.idx_special_token]
            for sentence in token_en_ids
        ]
        words_vi_pred = [
            [self.idx2word_vi[idx.item()] for idx in sentence if idx.item() not in self.idx_special_token]
            for sentence in token_vi_ids
        ]


        words_en_tgt = [
            [self.idx2word_en[idx.item()] for idx in sentence if idx.item() not in self.idx_special_token]
            for sentence in tgt_vi2en
        ]
        words_vi_tgt = [
            [self.idx2word_vi[idx.item()] for idx in sentence if idx.item() not in self.idx_special_token]
            for sentence in tgt_en2vi
        ]

        self.pred_en_strings.extend([" ".join(words) for words in words_en_pred])
        self.pred_vi_strings.extend([" ".join(words) for words in words_vi_pred])

        self.tgt_en_strings.extend([[" ".join(words)] for words in words_en_tgt])
        self.tgt_vi_strings.extend([[" ".join(words)] for words in words_vi_tgt])

    def compute_all(self):

        # Calculate BLEU
        bleu_score_en = self.bleu_metric.compute(predictions=self.pred_en_strings, references=self.tgt_en_strings)[
            'score']
        bleu_score_vi = self.bleu_metric.compute(predictions=self.pred_vi_strings, references=self.tgt_vi_strings)[
            'score']

        # Calculate chrF
        chrf_score_en = self.chrf_metric.compute(predictions=self.pred_en_strings, references=self.tgt_en_strings)[
            'score']
        chrf_score_vi = self.chrf_metric.compute(predictions=self.pred_vi_strings, references=self.tgt_vi_strings)[
            'score']

        # Clear the memory for the next evaluation phase
        self.reset()

        return bleu_score_vi, bleu_score_en, chrf_score_vi, chrf_score_en