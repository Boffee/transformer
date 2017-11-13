import tensorflow as tf
import os

START_TOK = u"<s>"
END_TOK = u"</s>"
PAD_TOK = u"<pad>"

def load_vocab(vocab_path):
    """loads vocab to index and index to vocab mapping."""
    with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
        vocab = [line.split()[0] for line in vocab_file]
    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
    idx2tok = {idx: tok for idx, tok in enumerate(vocab)}
    return tok2idx, idx2tok 

def create_vocab_tables(vocab_path, num_oov_buckets=1):
    """creates tensorflow vocab lookup table."""
    tok2idx_table = tf.contrib.lookup.index_table_from_file(vocab_path, num_oov_buckets=num_oov_buckets)
    idx2tok_table = tf.contrib.lookup.index_to_string_table_from_file(vocab_path)
    return tok2idx_table, idx2tok_table

def load_dataset(src_path, 
                 tgt_path,
                 src_tok2idx_table,
                 tgt_tok2idx_table,
                 maxlen=80,
                 num_parallel_calls=4,
                 batch_size=64):
    output_buffer_size = batch_size*100
    src = tf.data.TextLineDataset(src_path)
    tgt = tf.data.TextLineDataset(tgt_path)

    src_tgt = tf.data.Dataset.zip((src, tgt))

    src_tgt = src_tgt.map(
        lambda src, tgt: (tf.string_split([src + " " + END_TOK]).values, 
                        tf.string_split([tgt + " " + END_TOK]).values),
        num_parallel_calls=num_parallel_calls)

    src_tgt = src_tgt.filter(
        lambda src, tgt: tf.less_equal(tf.size(src), maxlen))
    
    src_tgt = src_tgt.filter(
        lambda src, tgt: tf.less_equal(tf.size(tgt), maxlen))
    
    src_tgt = src_tgt.map(
        lambda src, tgt: (tf.to_int32(src_tok2idx_table.lookup(src)),
                          tf.to_int32(tgt_tok2idx_table.lookup(tgt))),
        num_parallel_calls=num_parallel_calls)
    
    src_tgt = src_tgt.shuffle(output_buffer_size)

    batched_src_tgt = src_tgt.padded_batch(batch_size, padded_shapes=(maxlen,maxlen))

    return batched_src_tgt


def load_data_queue(src_path,
                    tgt_path,
                    src_tok2idx_table,
                    tgt_tok2idx_table,
                    maxlen=80,
                    num_parallel_calls=4,
                    batch_size=64):
    
    batched_src_tgt = load_dataset(src_path,
                                   tgt_path,
                                   src_tok2idx_table,
                                   tgt_tok2idx_table,
                                   maxlen=maxlen,
                                   num_parallel_calls=num_parallel_calls,
                                   batch_size=batch_size)

    batch_iterator = batched_src_tgt.make_initializable_iterator()
    (src, tgt) = batch_iterator.get_next()
    
    return src, tgt, batch_iterator.initializer

