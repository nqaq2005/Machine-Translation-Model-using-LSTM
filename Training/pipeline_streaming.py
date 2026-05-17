import os
from torch.utils.data import DataLoader

from MachineTranslation.Utils.Parser import get_args_parser
from  MachineTranslation.Utils.utils import *
from MachineTranslation.Utils.Metrics import Metric

def trainer():
    args = get_args_parser()

    vi_word2idx, vi_idx2word, vi_embedd = load_vocab_and_embeddings(path=args.vi_vocab_path)
    en_word2idx, en_idx2word, en_embedd = load_vocab_and_embeddings(path=args.en_vocab_path)

    train_dataset, valid_dataset = load_datasets(path_dataset=args.dataset_path, word2id_en=en_word2idx,
                                                       word2id_vi=vi_word2idx, batch_size=args.batch_size,
                                                       buffer_size=args.buffer_size, streaming=args.streaming)

    train_dataloader = DataLoader(dataset=train_dataset, collate_fn=translation_collate_fn)
    valid_dataloader = DataLoader(dataset=valid_dataset, collate_fn=translation_collate_fn)

    device, epochs, writer, vi_vocab_size, en_vocab_size, idx_special_token, ignore_idx, accumulation_steps \
        = setup_experiment(epochs=args.epochs, vi_vocab=vi_idx2word, en_vocab=en_idx2word, run_name=args.run_name,
                           accumulation_steps=args.accumulation_steps)

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

    scaler = torch.cuda.amp.GradScaler()
    PAD_IDX = 0

    for epoch in range(epochs):
        steps_since_update = 0
        seq2seq.train()

        for datapoint in train_dataloader:
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

            mask_vi2en = (encode_vi2en == PAD_IDX)
            mask_en2vi = (encode_en2vi == PAD_IDX)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs_vi2en = seq2seq(encode_vi2en, lengths_vi2en, src_decode_vi2en, True,
                                        mask_vi2en, teacher_forcing_ratio=args.teacher_forcing_ratio)

                outputs_en2vi = seq2seq(encode_en2vi, lengths_en2vi, src_decode_en2vi, False,
                                        mask_en2vi, teacher_forcing_ratio=args.teacher_forcing_ratio)

                # Tính Loss
                loss_vi2en = loss_fn(outputs_vi2en, tgt_decode_vi2en)
                loss_en2vi = loss_fn(outputs_en2vi, tgt_decode_en2vi)
                total_loss = (loss_vi2en + loss_en2vi) / accumulation_steps

                display_loss_vi = loss_vi2en.item()
                display_loss_en = loss_en2vi.item()
                display_loss_total = total_loss.item() * accumulation_steps  # Nhân lại để log đúng giá trị thật

            scaler.scale(total_loss).backward()
            steps_since_update += 1

            # Backward pass & Optimize
            if (steps_since_update == accumulation_steps):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), max_norm=args.clip_grad_norm)

                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()
                steps_since_update = 0

            writer.add_scalar("Loss_vi2en/train", display_loss_vi, global_train_step)
            writer.add_scalar("Loss_en2vi/train", display_loss_en, global_train_step)
            writer.add_scalar("Total_loss/train", display_loss_total, global_train_step)

            global_train_step += 1

        # Vét gradient cuối epoch (Nếu dataset bị lẻ)
        if steps_since_update > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(seq2seq.parameters(), max_norm=args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        seq2seq.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0

        with torch.inference_mode():
            for datapoint in valid_dataloader:
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

                mask_vi2en_val = (encode_vi2en == PAD_IDX)
                mask_en2vi_val = (encode_en2vi == PAD_IDX)

                outputs_vi2en = seq2seq(encode_vi2en, lengths_vi2en, src_decode_vi2en, True,
                                        mask_vi2en_val, teacher_forcing_ratio=0.0)
                outputs_en2vi = seq2seq(encode_en2vi, lengths_en2vi, src_decode_en2vi, False,
                                        mask_en2vi_val, teacher_forcing_ratio=0.0)

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