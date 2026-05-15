from tqdm import tqdm
import os

from MachineTranslation.Utils.Parser import get_args_parser
from  MachineTranslation.Utils.utils import *
from MachineTranslation.Utils.Metrics import Metric

def trainer():
    args = get_args_parser()

    vi_word2idx, vi_idx2word, vi_embedd = load_vocab_and_embeddings(path=args.vi_vocab_path)
    en_word2idx, en_idx2word, en_embedd = load_vocab_and_embeddings(path=args.en_vocab_path)

    train_dataloader, valid_dataloader = load_datasets(path_dataset=args.dataset_path, word2id_en=en_word2idx,
                                                       word2id_vi=vi_word2idx, batch_size=args.batch_size,
                                                       buffer_size=args.buffer_size)

    device, epochs, writer, vi_vocab_size, en_vocab_size, idx_special_token, ignore_idx \
        = setup_experiment(epochs=args.epochs, vi_vocab=vi_idx2word, en_vocab=en_idx2word, run_name=args.run_name)

    encoder, decoder, outputLayer_vi, outputLayer_en, seq2seq \
        = load_models(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers,
                      device=device, vi_embedd=vi_embedd, en_embedd=en_embedd,
                      vi_vocab_size=vi_vocab_size, en_vocab_size=en_vocab_size, dropout=args.dropout)

    optimizer, scheduler, loss_fn \
        = configure_optimizers(model=seq2seq, lr=args.lr, ignore_idx=ignore_idx, label_smoothing=args.label_smoothing)

    metric = Metric(idx2word_vi=vi_idx2word, idx2word_en=en_idx2word, idx_special_token=idx_special_token)

    global_train_step = 1
    global_valid_step  = 1
    best_crhf = 0
    best_bleu = 0

    for epoch in range(epochs):
        train_dataloader.dataset.set_epoch(epoch)
        progress_bar = tqdm(train_dataloader)
        seq2seq.train()

        for iter, datapoint in enumerate(progress_bar):
            vi_en:dict = datapoint['vi_en']
            en_vi:dict = datapoint['en_vi']

            encode_vi2en = vi_en['encode_vi2en'].to(device)
            lengths_vi2en = vi_en['lengths_vi2en']  # CPU
            src_decode_vi2en = vi_en['src_decode_vi2en'].to(device)
            tgt_decode_vi2en = vi_en['tgt_decode_vi2en'].to(device)

            encode_en2vi = en_vi['encode_en2vi'].to(device)
            lengths_en2vi = en_vi['lengths_en2vi']  # CPU
            src_decode_en2vi = en_vi['src_decode_en2vi'].to(device)
            tgt_decode_en2vi = en_vi['tgt_decode_en2vi'].to(device)

            outputs_vi2en = seq2seq(encode_vi2en, lengths_vi2en, src_decode_vi2en, True,
                                    teacher_forcing_ratio=args.teacher_forcing_ratio)
            outputs_en2vi = seq2seq(encode_en2vi, lengths_en2vi, src_decode_en2vi, False,
                                    teacher_forcing_ratio=args.teacher_forcing_ratio)

            loss_vi2en = loss_fn(outputs_vi2en, tgt_decode_vi2en)
            loss_en2vi = loss_fn(outputs_en2vi, tgt_decode_en2vi)

            total_loss = loss_vi2en+loss_en2vi

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), max_norm=args.clip_grad_norm)
            optimizer.step()

            writer.add_scalar("Loss_vi2en/train", loss_vi2en.item(), global_train_step)
            writer.add_scalar("Loss_en2vi/train", loss_en2vi.item(), global_train_step)
            writer.add_scalar("Total_loss/train", total_loss.item(), global_train_step)

            global_train_step += 1

            progress_bar.set_description(f"""Epoch: {epoch+1}. Iteration: {iter}.
                                             Loss_vi2en: {loss_vi2en}. Loss_en2vi: {loss_en2vi}. Total_loss: {total_loss}""")

        seq2seq.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0

        with torch.inference_mode():
            progress_bar = tqdm(valid_dataloader)
            for iter, datapoint in enumerate(progress_bar):
                vi_en: dict = datapoint['vi_en']
                en_vi: dict = datapoint['en_vi']

                encode_vi2en = vi_en['encode_vi2en'].to(device)
                lengths_vi2en = vi_en['lengths_vi2en']  # CPU
                src_decode_vi2en = vi_en['src_decode_vi2en'].to(device)
                tgt_decode_vi2en = vi_en['tgt_decode_vi2en'].to(device)

                encode_en2vi = en_vi['encode_en2vi'].to(device)
                lengths_en2vi = en_vi['lengths_en2vi']  # CPU
                src_decode_en2vi = en_vi['src_decode_en2vi'].to(device)
                tgt_decode_en2vi = en_vi['tgt_decode_en2vi'].to(device)
                num_val_batches += 1

                outputs_vi2en = seq2seq(encode_vi2en, lengths_vi2en, src_decode_vi2en,
                                        True, teacher_forcing_ratio=0.0) #batch x seq x vocab
                outputs_en2vi = seq2seq(encode_en2vi, lengths_en2vi, src_decode_en2vi,
                                        False, teacher_forcing_ratio=0.0) #batch x seq x vocab

                loss_vi2en = loss_fn(outputs_vi2en, tgt_decode_vi2en)
                loss_en2vi = loss_fn(outputs_en2vi, tgt_decode_en2vi)
                total_loss = loss_vi2en + loss_en2vi
                epoch_val_loss += total_loss.item()

                metric.add_batch(outputs_vi2en=outputs_vi2en, outputs_en2vi=outputs_en2vi,
                                       tgt_vi2en=tgt_decode_vi2en, tgt_en2vi=tgt_decode_en2vi)

                writer.add_scalar("Loss_vi2en/valid", loss_vi2en.item(), global_valid_step)
                writer.add_scalar("Loss_en2vi/valid", loss_en2vi.item(), global_valid_step)
                writer.add_scalar("Total_loss/valid", total_loss.item(), global_valid_step)


                global_valid_step += 1

                progress_bar.set_description(f""" Epoch: {epoch + 1}. Iteration: {iter}.
                                                  Loss_vi2en: {loss_vi2en}. Loss_en2vi: {loss_en2vi}. Total_loss: {total_loss}.""")

        bleu_score_vi, bleu_score_en, chrf_score_vi, chrf_score_en = metric.compute_all()

        writer.add_scalar("Total BLEU Vietnamese", bleu_score_vi, epoch)
        writer.add_scalar("Total BLEU English",    bleu_score_en, epoch)
        writer.add_scalar("Total CHRF Vietnamese", chrf_score_vi, epoch)
        writer.add_scalar("Total CHRF English",    chrf_score_en, epoch)

        total_bleu = (bleu_score_vi+bleu_score_en)/2
        total_chrf = (chrf_score_vi+chrf_score_en)/2

        is_best = False
        if total_chrf > best_crhf:
            best_crhf = total_chrf
            # We also update best_bleu just to know what the BLEU was on the best chrF epoch
            best_bleu = total_bleu
            is_best = True

        checkpoints = {
            "epoch"         : epoch,
            "encoder"       : encoder.state_dict(),
            "decoder"       : decoder.state_dict(),
            "outputlayer_vi": outputLayer_vi.state_dict(),
            "outputlayer_en": outputLayer_en.state_dict(),
            "optimizer"     : optimizer.state_dict(),
            "scheduler"     : scheduler.state_dict(),
            "best_chrf"     : best_crhf,
            "best_bleu"     : best_bleu
        }

        torch.save(checkpoints,  os.path.join(args.run_name, "checkpoints_latest.pt"))

        if is_best:
            print(f"🌟 New best model found at Epoch {epoch}! chrF: {best_crhf:.2f} (BLEU: {best_bleu:.2f})")
            torch.save(checkpoints, os.path.join(args.run_name, "checkpoints_best.pt"))

        avg_val_loss = epoch_val_loss / num_val_batches
        scheduler.step(avg_val_loss)

if __name__ == '__main__':
    trainer()