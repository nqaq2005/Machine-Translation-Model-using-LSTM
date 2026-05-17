# 🌐 Bidirectional Neural Machine Translation (Vietnamese <-> English)

A robust, memory-efficient Sequence-to-Sequence (Seq2Seq) Machine Translation model built from scratch in PyTorch. This project translates between Vietnamese and English using a Gated Recurrent Unit (GRU) architecture enhanced with an Attention mechanism.

## ✨ Key Features

This project implements several advanced deep learning techniques to ensure training stability, speed, and memory efficiency:

* **Optimized RNN Processing:** Utilizes PyTorch's `pack_padded_sequence` and `pad_packed_sequence` to skip computations on `<PAD>` tokens, preserving state integrity and speeding up training.
* **Streaming Datasets:** Integrates Hugging Face's `IterableDataset` with custom buffer-based shuffling (`set_epoch`), allowing the model to train on massive translation datasets without exceeding RAM limits.
* **Smart Embedding Fine-tuning:** Leverages pre-trained FastText word vectors. Uses custom **PyTorch Autograd Hooks** to freeze the semantic space of the vocabulary while exclusively fine-tuning special operational tokens (`<Vi2En>`, `<En2Vi>`, `<EOS>`, `<UNK>`).
* **Bidirectional Training:** Capable of learning both Vietnamese-to-English and English-to-Vietnamese translation pathways simultaneously within the same batch.
* **Dynamic Learning Rate:** Implements `ReduceLROnPlateau` to automatically adjust the learning rate based on validation loss plateaus.
* **Automated Evaluation:** Real-time logging of Loss, BLEU, and chrF scores to Tensorboard using the Hugging Face `evaluate` library.

## 📂 Project Structure

```text
MachineTranslation/
├── Dataset/
│   ├── loaders/
│   │   └── dataset.py               # IterableDataset with buffer shuffling
│   └── processing/
│       ├── make_embedding.py        # FastText vector extraction & saving
│       └── make_vocab.py            # Tokenization and vocabulary building
├── Model/
│   ├── attention_mechanism.py       # Custom Attention scoring and weighting
│   └── gru.py                       # Encoder, Decoder, OutputLayer, and Seq2Seq wrapper
├── Training/
│   └── train_pipeline.py            # Main training loop, validation, and checkpointing
└── Utils/
    ├── LabelSmoothingCrossEntropy.py # Custom Loss function (Alternative)
    ├── Metrics.py                   # BLEU and chrF score calculation
    ├── Parser.py                    # Command-line argument configuration
    └── utils.py                     # Collate functions, dataset loading, and setup