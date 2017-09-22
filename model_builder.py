import tensorflow as tf
from utils import *
from layers import *

def create_model(src,
                 tgt,
                 src_vocab_size,
                 tgt_vocab_size,
                 hidden_size=512,
                 src_embed_scope="src_embed",
                 tgt_embed_scope="tgt_embed",
                 enc_scope="enc", 
                 dec_scope="dec"):

    tgt_shifted = tf.concat([tf.ones([1, get_shape(tgt)[-1]]), tgt[:,:-1]],  -1, name="shift_tgt")

    src_embedded = embed(src, src_vocab_size, embed_size=hidden_size, scope=src_embed_scope)
    tgt_embedded = embed(tgt_shifted, tgt_vocab_size, embed_size=hidden_size, scope=tgt_embed_scope)

    encoder_outputs = encode(src_embedded, scope=enc_scope)
    decoder_outputs = decode(tgt_embedded, encoder_outputs, scope=dec_scope)

    logits = project(decoder_outputs, tgt_vocab_size)

    return logits

def embed(sequences,
          vocab_size,
          embed_size=512,
          scope="embed",
          reuse=None):
    """
    Embed the sequences and add positional encoding
    Args:
        sequences: int tensor of shape (batch_size, max_time_step) mapped from token to index to embed.
        vocab_size: int, embedding lookup size. Number of unique tokens that needs to be embedded.
        embed_size: int, output embedding space size.
        scope: string, variable scope.
        reuse: Boolean, wheter to reuse the weights of the previous layers by the same name.
    Returns:
        tensor of shape (batch_size, max_time_step, embed_size)
    """
    with tf.variable_scope(scope, reuse=reuse):
        embedded = embedding(sequences, vocab_size, embed_size=512)
        max_time_step = get_shape(sequences)[-1]
        pe = positional_encoding(max_time_step, encoding_size=embed_size)
        return embedded + pe

def encode(src_embedded,
           masking_range=None,
           hidden_size=512,
           num_heads=8,
           num_layers=6,
           num_devs=1,
           scope="enc",
           resuse=None):
    """
    Encode the embeded sequences for the decoder
    Args:
        src_embedded: tensor of shape (batch_size, max_time_step, embed_size) to encode.
        masking_range: tuple specifying the attention masking range relative to the position of the query.
            `None` maskes from the beginning and to the end for the first and second value respective.
            (1, None) maskes all future tokens from the querying position.
            Swap the first and second value to mask everything outside of the range
        hidden_size: int, size of encoder hidden layers
        num_heads: int, number of heads for multihead attention
        num_devs: int, number of devices distribute the attentions heads on
        scope: string, variable scope.
        reuse: Boolean, wheter to reuse the weights of the previous layers by the same name
    Returns:
        tensor of shape (batch_size, max_time_step, hidden_size)
    """
    with tf.variable_scope(scope):
        self_attention_sublayer_fn = lambda queries : multihead_attention(
            queries, queries, hidden_size=hidden_size, num_heads=num_heads, num_devs=num_devs, masking_range=masking_range)    
        feedfoward_sublayer_fn = lambda attentions : feedfoward(attentions, hidden_size=hidden_size)

        tranformer_sublayers = [self_attention_sublayer_fn, feedfoward_sublayer_fn]
        src_encoded = stack_layers(src_embedded, tranformer_sublayers, num_layers=num_layers, layer_scope_prefix="transformer")
    
    return src_encoded

def decode(src_encoded,
           tgt_embedded,
           masking_range=(1,None),
           hidden_size=512,
           num_heads=8,
           num_layers=6,
           num_devs=1,
           scope="enc",
           resuse=None):
    """
    Decode the encoded sequences
    Args:
        src_encoded: tensor of shape (batch_size, max_time_step, encoder_hidden_size) to decode
        tgt_embedded: tensor of shape (batch_size, max_time_step, embed_size) to decode
        masking_range: tuple specifying the attention masking range relative to the position of each query.
            `None` maskes from the beginning and to the end for the first and second value respective.
            (1, None) maskes all future tokens from the querying position.
            Swap the first and second value to mask everything outside of the range
        hidden_size: int, size of encoder hidden layers
        num_heads: int, number of heads for multihead attention
        num_devs: int, number of devices distribute the attentions heads on
        scope: string, variable scope.
        reuse: Boolean, whether to reuse the weights of the previous layers by the same name.
    Returns:
        tensor of shape (batch_size, max_time_step, hidden_size)
    """
    with tf.variable_scope(scope):
        self_attention_sublayer_fn = lambda queries : multihead_attention(
            queries, queries, hidden_size=hidden_size, num_heads=num_heads, num_devs=num_devs, masking_range=masking_range)
        memory_attention_sublayer_fn = lambda queries : multihead_attention(
            queries, src_encoded, hidden_size=hidden_size, num_heads=num_heads, num_devs=num_devs)
        feedfoward_sublayer_fn = lambda attentions : feedfoward(attentions, hidden_size=hidden_size)
        
        tranformer_sublayers = [self_attention_sublayer_fn, memory_attention_sublayer_fn, feedfoward_sublayer_fn]
        src_decoded = stack_layers(tgt_embedded, tranformer_sublayers, num_layers=num_layers, layer_scope_prefix="transformer")

    return src_decoded

def project(src_decoded, tgt_size, scope="project", reuse=None):
    """
    Project (linear) the hidden decoder output states to the target data space
    Args:
        src_decoded: tensor of shape (batch_size, max_time_step, decoder_hidden_size) to project
        tgt_size: int, size of the target data space
    Returns:
        logit tensor of shape (batch_size, max_time_step, tgt_size)
        predicted sequence int tensor of shape (batch_size, max_time_step)
    """
    with tf.variable_scope(scope, reuse=reuse):
        logits = tf.layers.dense(src_decoded, tgt_size)
        pred_sequences = tf.to_int32(tf.argmax(logits, axis=-1))
        
    return logits, pred_sequences