import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description="Machine Translation Training Pipeline (GRU Seq-to-Seq)")

    # --- File Paths ---
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the translation dataset')
    parser.add_argument('--vi_vocab_path', type=str, default='vi_vocab_embeddings.pt',
                        help='Path to the Vietnamese vocab/embeddings dictionary')
    parser.add_argument('--en_vocab_path', type=str, default='en_vocab_embeddings.pt',
                        help='Path to the English vocab/embeddings dictionary')
    parser.add_argument('--run_name', type=str, default='runs/gru_translation',
                        help='Tensorboard run directory')

    # --- Model Architecture ---
    parser.add_argument('--input_dim', type=int, default=300,
                        help='Dimension of the FastText embeddings (default 300)')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension for the GRU layers')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers in the GRU encoder/decoder')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')

    # --- Training Hyperparameters ---
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate for Adam optimizer')
    parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5,
                        help='Probability of using teacher forcing during decoding')
    parser.add_argument('--label_smoothing', type=float, default=0.1,
                        help='Label smoothing value for the loss function (default: 0.1)')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                        help='Maximum norm for gradient clipping (default: 1.0)')
    parser.add_argument('--buffer_size', type=int, default=10000,
                        help='Buffer size for shuffling the streaming dataset')
    parser.add_argument('--accumulation_steps', type=int, default=32,
                        help='Number of accumulation steps')
    parser.add_argument('--streaming', type=bool,  default=False,
                        help='Streaming dataset or not')

    return parser.parse_args()
