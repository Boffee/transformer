import tensorflow as tf
import numpy as np
import os
from data_loader import *
from layers import *
from layers_experimental import *
from model_builder import *
from model_trainer import *


def train(src_path,
          tgt_path,
          src_vocab_path,
          tgt_vocab_path,
          logdir,
          maxlen=32,
          batch_size=64,
          num_oov_buckets=1):
    """Train model"""
    print("Loading vocab...")
    src2idx, idx2src = load_vocab(src_vocab_path)
    tgt2idx, idx2tgt = load_vocab(tgt_vocab_path)
    src_vocab_size = len(src2idx) + num_oov_buckets
    tgt_vocab_size = len(tgt2idx) + num_oov_buckets

    src2idx_table, idx2src_table = create_vocab_tables(src_vocab_path, num_oov_buckets=num_oov_buckets)
    tgt2idx_table, idx2tgt_table = create_vocab_tables(tgt_vocab_path, num_oov_buckets=num_oov_buckets)

    (x, y, queue_init) = load_data_queue(src_path, tgt_path, src2idx_table, tgt2idx_table, maxlen=maxlen, batch_size=batch_size)
    print("Building model...")
    logits, preds = build_model(x, src_vocab_size, tgt_vocab_size)
    print("Creating metrics...")
    _, mean_loss, _ = compute_metrics(y, logits, preds)
    print("Generating train op...")
    train_op, global_step = create_train_op(mean_loss)

    print("Starting training...")
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=900, local_init_op=tf.group(queue_init, tf.tables_initializer())) #init_fn=lambda sess: sess.run(queue_init))

    fetches = [x, y, preds, global_step, mean_loss, train_op]
    fetch_keys = ['x', 'y', 'preds', 'global_step', 'mean_loss', 'train_op']
    
    with sv.managed_session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        while not sv.should_stop():
            fetch_results = sess.run(fetches)
            fetch_dict = dict(zip(fetch_keys, fetch_results))
            if fetch_dict['global_step'] % 200 == 0:
                print("=============================================================")
                print("step {}, total loss: {}".format(fetch_dict['global_step'], fetch_dict['mean_loss']))
                print("=============================================================")
                print("-- source: ", convert_seq2sent(fetch_dict['x'][0], idx2src))
                print("-- target: ", convert_seq2sent(fetch_dict['y'][0], idx2tgt))
                print("-- prediction: ", convert_seq2sent(fetch_dict['preds'][0], idx2tgt))


def convert_seq2sent(seq, idx2tok):
    return " ".join([idx2tok[idx] for idx in seq])


def build_model(inputs, src_vocab_size, tgt_vocab_size):
    """build model graph"""
    src_embed = embed(inputs, src_vocab_size,
                      scope="src_embed", embedding_depth=256)
    aggr1 = aggregation_layer(src_embed, scope="aggr1", output_depth=256)
    aggr2 = aggregation_layer(aggr1, scope="aggr2", output_depth=256)
    disp1 = dispersion_layer(aggr2, scope="disp1", output_depth=256)
    disp2 = dispersion_layer(disp1, scope="disp2", output_depth=256)

    logits, preds = project(disp2, tgt_vocab_size)

    return logits, preds


def compute_metrics(y, logits, preds, add_summary=True):
    """Compute loss and accuracy"""
    valid_tgt_pos = get_valid_pos(y)
    loss, mean_loss = compute_loss(y, logits, valid_tgt_pos)
    accuracy = compute_accuracy(y, preds, valid_tgt_pos)

    if add_summary:
        tf.summary.scalar("mean_loss", mean_loss)
        tf.summary.scalar("accuracy", accuracy)

    return loss, mean_loss, accuracy


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--srcfile', help="path to source dataset", type=str, required=True)
    parser.add_argument('--tgtfile', help="path to target dataset", type=str, required=True)
    parser.add_argument('--srcvocabfile', help="path to source vocab", type=str, required=True)
    parser.add_argument('--tgtvocabfile', help="path to target vocab", type=str, required=True)
    parser.add_argument('--logdir', help="log and checkpoint directory", type=str, required=True)
    parser.add_argument('--dataroot', help="root directory of data files", type=str, default="")
    parser.add_argument('--maxlen', help="maximum number of tokens in a sentence", type=int, default=32)
    parser.add_argument('--batchsize', help="number of samples per batch", type=int, default=128)
    parser.add_argument('--numoov', help="number of oov buckets", type=int, default=1)

    args = parser.parse_args()

    src_path = os.path.join(args.dataroot, args.srcfile)
    tgt_path = os.path.join(args.dataroot, args.tgtfile)
    src_vocab_path = os.path.join(args.dataroot, args.srcvocabfile)
    tgt_vocab_path = os.path.join(args.dataroot, args.tgtvocabfile)

    train(src_path,
          tgt_path,
          src_vocab_path,
          tgt_vocab_path,
          args.logdir,
          maxlen=args.maxlen,
          batch_size=args.batchsize,
          num_oov_buckets=args.numoov)
