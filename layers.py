import tensorflow as tf
from utils import *

def positional_encoding(max_time_step, encoding_size=512):
    """
    Positional encoding in https://arxiv.org/pdf/1706.03762.pdf
    Each dimension in the encoding represents a sinusoidal progression of different wavelength
    from the beginning to the end of a sequence.
    Since any segment of the sinusoid can be represented by an offset of another segment, 
    the encoding is ensentially converting absolute position [0,1,2,3,...] in the sequence to 
    relative positions [...,-2,-1,0,1,2,..], where long and short wavelengths encode long and short 
    relative position ranges respective.
    Args:
        max_time_step: int, max length of input sequence
        encoding_size: int, output encoding size
    Returns:
        tensor of shape (batch_size, encoding_size)
    """
    with tf.variable_scope("positional_encoding"):
        pos = tf.expand_dims(tf.range(max_time_step), 0)
        i = tf.expand_dims(tf.range(encoding_size), 1)
        radians = tf.matmul(pos, 10000**(i/encoding_size))
        pe = tf.where(tf.equal(i%2, 0), tf.sin(radians), tf.cos(radians))
    return pe


def embedding(sequences, vocab_size, embed_size=512):
    """
    Map the index representation of sequences into a dense vector space.
    Args:
        sequences: int tensor of shape (batch_size, max_time_step) mapped from token to index to embed.
        vocab_size: int, embedding lookup size. Number of unique tokens that needs to be embedded.
        embed_size: int, output embedding space size.
    Returns:
        tensor of shape (batch_size, max_time_step, embed_size)
    """
    with tf.variable_scope("embedding"):
        embedding_kernel = tf.get_variable("kernel", shape=[vocab_size, embed_size])
        sequences_embedded = tf.gather(embedding_kernel, sequences)
    return sequences

def feedforward(inputs,
                hidden_depth=2048,
                output_depth=512,
                strides=1,
                kernel_size=1,
                padding="valid",
                use_residual=True,
                use_layer_norm=True):
    """
    Two layer feedforward (nonlinear mapping) using 1D convolutions (equivalent to dense layers if stride and kernel size = 1)
    Args:
        hidden_depth: int, number of convolutional units to use for the hidden layer
        output_depth: int, number of convolutional units to use for the output layer
        strides: int, stride width of the sliding window. (number of time steps gapped between each window)
        kernel_size: int, width of each window. (number of time steps covered by each window)
        use_residual: Boolean, whether to use residual connection between the outputs and the queries.
        use_layer_norm: Boolean, whether to layer normalize the outputs.
    Returns:

    """
    intermediates = tf.layers.conv1d(inputs, hidden_depth, strides=strides, kernel_size=kernel_size, padding=padding, activation=tf.nn.relu)
    outputs = tf.layers.conv1d(intermediates, output_depth, strides=strides, kernel_size=kernel_size, padding=padding)

    if use_residual:
        outputs = residual_connection(outputs, inputs)

    if use_layer_norm:
        outputs = layer_normalization(outputs)

    return outputs

def multihead_attention(queries,
                        keys,
                        values,
                        biases=None,
                        masking_range=(1, None),
                        attention_depth=512,
                        num_heads=8,
                        num_devs=1,
                        use_residual=True,
                        use_layer_norm=True):
    """
    partition the input space into multiple subspaces, and run an independent attention on each subspace.
    This allows the model to attend to multiple independent positions, each conditioned on different
    non-overlapping dimensions, instead a single attention weighed across all values.
    Additionaly, since the size of vector spaces increases exponential with each additional dimension 
    (curse of dimensionality) partitioning a space into N subspaces reduces the size of the original space
    from size(space) to N*pow(size(space),1/N).
    Args:
        queries: tensor of shape (batch_size, queries_length, query_depth) containing the queries for memories.
        keys: tensor of shape (batch_size, keys_length, key_depth) containing the keys to the memories.
        values: tensor of shape (batch_size, values_length, value_depth) containing the memories to attend to.
        biases: tensor of shape (batch_size, queries_length, values_length) containing biases to the attention weight logits.
        attention_depth: int, number of dimensions in the attention space
        num_heads: int, number of subspaces to partition into.
        num_devs: int, number of devices distribute the attentions heads on.
        use_residual: Boolean, whether to use residual connection between the outputs and the queries.
        use_layer_norm: Boolean, whether to layer normalize the outputs.
    Returns:
        tensor of shape (batch_size, queries_length, attention_depth)
    """
    with tf.variable_scope("multihead_attention"):
        queries_partitions = tf.split(queries, num_heads, axis=-1)
        values_partitions = tf.split(values, num_heads, axis=-1)

        attention_heads = []
        for index, (queries_partition, values_partition) in enumerate(zip(queries_partitions, values_partitions)):
            with tf.device("/gpu:{}".format(index%num_devs)):
                attention_head = single_attention_head(queries_partition, values_partition, masking_range, attention_depth/num_heads)
                attention_heads.append(attention_head)
        
        attentions = tf.concat(attention_heads, axis=-1)
        outputs = tf.layers.conv1d(attentions, attention_depth, 1, 0)

        if use_residual:
            outputs = residual_connection(outputs, queries)

        if use_layer_norm:
            outputs = layer_normalization(outputs)
    
    return outputs


def single_attention_head(queries,
                          keys,
                          values,
                          attention_depth,
                          biases=None):
    """
    project the input space into the attention space and attend to the values using the scaled dot product attention.
    Args:
        queries: tensor of shape (batch_size, queries_length, query_depth) containing the queries for memories.
        keys: tensor of shape (batch_size, keys_length, key_depth) containing the keys to the memories.
        values: tensor of shape (batch_size, values_length, value_depth) containing the memories to attend to.
        biases: tensor of shape (batch_size, queries_length, values_length) containing biases to the attention weight logits.
    Returns:
        tensor of shape (batch_size, queries_length, attention_depth)
    """
    with tf.variable_scope("attention_head"):
        queries_projected = tf.layers.dense(queries, attention_depth)
        keys_projected = tf.layers.dense(keys, attention_depth)
        values_projected = tf.layers.dense(values, attention_depth)
        outputs = scaled_dot_product_attention(queries_projected, keys_projected, values_projected, biases)

    return outputs


def scaled_dot_product_attention(queries,
                                 keys,
                                 values,
                                 biases):
    """
    Dot product attention scaled by the depth of the keys.
    Args:
        queries: tensor of shape (batch_size, queries_length, query_depth) containing the queries for memories.
        keys: tensor of shape (batch_size, keys_length, key_depth) containing the keys to the memories.
        values: tensor of shape (batch_size, values_length, value_depth) containing the memories to attend to.
        biases: tensor of shape (batch_size, queries_length, values_length) containing biases to the attention weight logits.
    Returns:
        tensor of shape (batch_size, queries_length, values_length)
    """
    assert get_shape(queries)[-1] == get_shape(keys)[-1]
    assert get_shape(biases)[-2:] == [get_shape(queries)[-2], get_shape(keys)[-2]]
    with tf.variable_scope("scaled_dot_product_attention"):
        logits = tf.matmul(queries, keys, transpose_b=True)
        logits /= tf.to_float(get_shape(logits)[-1])**0.5
        if biases:
            logits += biases

        weights = tf.nn.softmax(logits)
        outputs = tf.matmul(weights, values)

    return outputs


def attention_mask(masking_range,
                   queries_length,
                   keys_length):
    """
    Create a binary mask, where masked=1 and unmasked=0, with the specified range relative to the
    position of each query.
    Args:
        masking_range: tuple of size 2, attention masking range relative to the position of each query.
            `None` maskes from the beginning and to the end for the first and second value respective.
            (1, None) maskes all future tokens from the querying position.
            Swap the first and second value to mask everything outside of the range
        queries_length: int, number of queries
        keys_length: int, number of keys
    Returns:
        tensor of shape (1, queries_length, values_length)
    """
    with tf.variable_scope("attention_mask"):
        mask_template = tf.ones([queries_length, keys_length])
        if not masking_range:
            mask = 1 - mask_template
        else:
            lower, upper = masking_range
            lower = max(lower-1, -queries_length) if lower else -queries_length
            upper = min(upper, queries_length) if upper else queries_length

            lower_mask = lower_matrix(mask_template, lower)
            upper_mask = upper_matrix(mask_template, upper)

            if lower <= upper:
                mask = 1 - lower_mask - upper_mask
            else:
                mask = abs(lower_mask - upper_mask)

            mask = tf.expand_dims(mask, [0])

    return mask


def layer_normalization(inputs,
                        subtensor_axes=[-1],
                        epsilon=1e-9):
    """
    layer normalization in https://arxiv.org/abs/1607.06450
    normalize the inputs by re-centering and re-scaling each subtensor defined by subtensor axes by the mean
    and standard deviation of the values in the subtensor. Since layer normalization does not need to normalize 
    against a batch, it is suitable for sequence models where the length of the input is not fixed and for
    large models which cannot use sufficiently large batches.tensor unit in the specified subspace
    Args:
        inputs: tensor with rank >= 2 and shape (batch_size, ...).
        subtensor_axes: list of int representing the axes of the subtensor.
    Returns:
        tensor of the same shape as inputs.
    """
    with tf.variable_scope("layer_normalization"):
        mean = tf.reduce_mean(inputs, axis=subtensor_axes, keep_dim=True)
        variance = tf.reduce_mean((inputs-mean)**2, axis=subtensor_axes, keep_dim=True)
        
        inputs_shape = tf.get_shape(inputs)
        subtensor_shape = [inputs_shape[i] if i in subtensor_axes else 1 for i in range(len(inputs_shape))]
        bias = tf.get_variable("bias", shape=subtensor_shape)
        gain = tf.get_variable("gain", shape=subtensor_shape)

        normalized = (1+gain) * (inputs-mean) / (variance+epsilon)**0.5 + bias

    return normalized

def residual_connection(current_states, previous_states, project=False):
    """
    Add the previous states to the current states. Optionally linearly project the previous states space
    onto the current states space.
    Args:
        current_states: tensor which has the same shape as previous_states for all dimensions except or
            including the last.
        previous_states: tensor which has the same shape as current_states for all dimensions except or
            including the last.
        project: Boolean, whether to linearly project previous_states the residual connection.
    Returns:
        tensor of the same shape as current_states
    """
    if not project:
        assert get_shape(current_states)[-1] == get_shape(previous_states)[-1]
        return current_states + previous_states
    else:
        return current_states + tf.layers.dense(previous_states, get_shape(current_states)[-1])
    

def lower_matrix(inputs, diag_offset):
    """
    Take the matrices represented by the last 2 dimensions of the input and preserve only the values below the
    offsetted diagonal.
    Args:
        inputs: tensor with rank >= 2
        diag_offset: int, positional offset from the center diagonal
    Returns:
        tensor of same shape as inputs
    """
    outputs = tf.matrix_band_part(input, -1, diag_offset)
    if diag_offset < 0:
        outputs -= tf.matrix_band_part(inputs, -diag_offset-1, -1)
    return outputs


def upper_matrix(inputs, diag_offset):
    """
    Take the matrices represented by the last 2 dimensions of the input and preserve only the values above the
    offsetted diagonal.
    Args:
        inputs: tensor with rank >= 2
        diag_offset: int, positional offset from the center diagonal
    Returns:
        tensor of same shape as inputs
    """
    outputs = tf.matrix_band_part(input, -diag_offset, -1)
    if diag_offset > 0:
        outputs -= tf.matrix_band_part(inputs, -1, diag_offset-1)
    return outputs