from torch.utils.data import Dataset
from datasets import load_dataset
import torch

from MachineTranslation.Dataset.processing.make_vocab import processing_english, processing_vietnamese


class static_VI_EN_translation(Dataset):
    def __init__(self, path_dataset: str, word2id_vi: dict, word2id_en: dict, split: str = 'train'):
        self.data = load_dataset(path_dataset, split=split)

        self.word2id_vi = word2id_vi
        self.word2id_en = word2id_en

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]

        vi_words = processing_vietnamese(row['Vietnamese'])
        en_words = processing_english(row['English'])

        ids_vi = torch.tensor([self.word2id_vi.get(w, 3) for w in vi_words], dtype=torch.long)
        ids_en = torch.tensor([self.word2id_en.get(w, 3) for w in en_words], dtype=torch.long)

        bos_en = torch.tensor([self.word2id_en["<Vi2En>"]], dtype=torch.long)
        eos_en = torch.tensor([self.word2id_en["<EOS_EN>"]], dtype=torch.long)

        bos_vi = torch.tensor([self.word2id_vi["<En2Vi>"]], dtype=torch.long)
        eos_vi = torch.tensor([self.word2id_vi["<EOS_VI>"]], dtype=torch.long)

        return {
            "vi_en": {
                "encode_vi2en": ids_vi,
                "src_decode_vi2en": torch.cat([bos_en, ids_en]),
                "tgt_decode_vi2en": torch.cat([ids_en, eos_en])
            },
            "en_vi": {
                "encode_en2vi": ids_en,
                "src_decode_en2vi": torch.cat([bos_vi, ids_vi]),
                "tgt_decode_en2vi": torch.cat([ids_vi, eos_vi])
            }
        }