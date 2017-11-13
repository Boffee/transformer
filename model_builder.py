import tensorflow as tf
from utils import *
from layers import *

def create_model(src,
                 tgt,
                 src_vocab_size,
                 tgt_vocab_size,
                 hidden_depth=512,
                 src_embed_scope="src_embed",
                 tgt_embed_scope="tgt_embed",
                 enc_scope="enc", 
                 dec_scope="dec"):
    """
    create the transformer model.
    Args:
        src: tensor of shape (batch_size, max_sequence_length) containing the index representation of the source sequences
        tgt: tensor of shape (batch_size, max_sequence_length) containing the index representation of the target sequences
        src_vocab_size: int, number of unique tokens in the source dataset
        tgt_vocab_size: int, number of unique tokens in the target dataset
        hidden_depth: int, size of hidden layers
        src_embed_scope: str, source embedding variable scope
        tgt_embed_scope: str, target embedding variable scope
        enc_scope: str, encoder variable scope
        dec_scope: str, decoder variable scope
    Returns:
        tensor of shape (batch_size, max_sequence_length, tgt_vocab_size)
    """
    tgt_shifted = tf.concat([tf.ones([1, get_shape(tgt)[-1]]), tgt[:,:-1]],  -1, name="tgt_shifted")

    src_embedded = embed(src, src_vocab_size, embedding_depth=hidden_depth, scope=src_embed_scope)
    tgt_embedded = embed(tgt_shifted, tgt_vocab_size, embedding_depth=hidden_depth, scope=tgt_embed_scope)

    encoder_outputs = encode(src_embedded, scope=enc_scope)
    decoder_outputs = decode(tgt_embedded, encoder_outputs, scope=dec_scope)

    logits = project(decoder_outputs, tgt_vocab_size)

    return logits

def embed(sequences,
          vocab_size,
          embedding_depth=512,
          scope="embed",
          reuse=None):
    """
    Embed the sequences and add positional encoding
    Args:
        sequences: int tensor of shape (batch_size, max_sequence_length) mapped from token to index to embed.
        vocab_size: int, embedding lookup size. Number of unique tokens that needs to be embedded.
        embedding_depth: int, output embedding space size.
        scope: string, variable scope.
        reuse: Boolean, whether to reuse the weights of the previous layers by the same name.
    Returns:
        tensor of shape (batch_size, max_sequence_length, embedding_depth)
    """
    with tf.variable_scope(scope, reuse=reuse):
        embedded = embedding(sequences, vocab_size,
                             embedding_depth=embedding_depth)
        max_sequence_length = get_shape(sequences)[-1]
        pe = positional_encoding(max_sequence_length, encoding_depth=embedding_depth)
        return embedded + pe

def encode(src_embedded,
           padding_mask=None,
           self_masking_range=None,
           hidden_depth=512,
           num_heads=8,
           num_layers=6,
           num_devs=1,
           scope="enc",
           resuse=None):
    """
    Encode the embeded sequences for the decoder
    Args:
        src_embedded: tensor of shape (batch_size, max_sequence_length, embedding_depth) to encode.
        self_masking_range: tuple specifying the self-attention masking range relative to the position of the query.
            `None` maskes from the beginning and to the end for the first and second value respective.
            (1, None) maskes all future tokens from the querying position.
            Swap the first and second value to mask everything outside of the range
        hidden_depth: int, size of encoder hidden layers
        num_heads: int, number of heads for multihead attention
        num_devs: int, number of devices distribute the attentions heads on
        scope: string, variable scope.
        reuse: Boolean, wheter to reuse the weights of the previous layers by the same name
    Returns:
        tensor of shape (batch_size, max_sequence_length, hidden_depth)
    """
    with tf.variable_scope(scope):
        self_attention_sublayer_fn = lambda queries : multihead_attention(
            queries, queries, hidden_depth=hidden_depth, num_heads=num_heads, num_devs=num_devs, self_masking_range=self_masking_range)    
        feedfoward_sublayer_fn = lambda attentions : feedfoward(attentions, hidden_depth=hidden_depth)

        tranformer_sublayers = [self_attention_sublayer_fn, feedfoward_sublayer_fn]
        src_encoded = stack_layers(src_embedded, tranformer_sublayers, num_layers=num_layers, layer_scope_prefix="transformer")
    
    return src_encoded

def decode(src_encoded,
           tgt_embedded,
           self_masking_range=(1,None),
           hidden_depth=512,
           num_heads=8,
           num_layers=6,
           num_devs=1,
           scope="enc",
           resuse=None):
    """
    Decode the encoded sequences
    Args:
        src_encoded: tensor of shape (batch_size, max_sequence_length, encoder_hidden_depth) to decode
        tgt_embedded: tensor of shape (batch_size, max_sequence_length, embedding_depth) to decode
        self_masking_range: tuple specifying the attention masking range relative to the position of each query.
            `None` maskes from the beginning and to the end for the first and second value respective.
            (1, None) maskes all future tokens from the querying position.
            Swap the first and second value to mask everything outside of the range
        hidden_depth: int, size of encoder hidden layers
        num_heads: int, number of heads for multihead attention
        num_devs: int, number of devices distribute the attentions heads on
        scope: string, variable scope.
        reuse: Boolean, whether to reuse the weights of the previous layers by the same name.
    Returns:
        tensor of shape (batch_size, max_sequence_length, hidden_depth)
    """
    with tf.variable_scope(scope):
        self_attention_sublayer_fn = lambda queries : multihead_attention(
            queries, queries, hidden_depth=hidden_depth, num_heads=num_heads, num_devs=num_devs, self_masking_range=self_masking_range)
        memory_attention_sublayer_fn = lambda queries : multihead_attention(
            queries, src_encoded, hidden_depth=hidden_depth, num_heads=num_heads, num_devs=num_devs)
        feedfoward_sublayer_fn = lambda attentions : feedfoward(attentions, hidden_depth=hidden_depth)
        
        tranformer_sublayers = [self_attention_sublayer_fn, memory_attention_sublayer_fn, feedfoward_sublayer_fn]
        src_decoded = stack_layers(tgt_embedded, tranformer_sublayers, num_layers=num_layers, layer_scope_prefix="transformer")

    return src_decoded

def project(src_decoded, tgt_size, scope="project", reuse=None):
    """
    Project (linear) the hidden decoder output states to the target data space
    Args:
        src_decoded: tensor of shape (batch_size, max_sequence_length, decoder_hidden_depth) to project
        tgt_size: int, size of the target data space
    Returns:
        logit tensor of shape (batch_size, max_sequence_length, tgt_size)
        predicted sequence int tensor of shape (batch_size, max_sequence_length)
    """
    with tf.variable_scope(scope, reuse=reuse):
        logits = tf.layers.dense(src_decoded, tgt_size)
        pred_sequences = tf.to_int32(tf.argmax(logits, axis=-1))
        
    return logits, pred_sequences

