import tensorflow as tf

def positional_encoding(max_time_step, encoding_size=512):
    """
    positional encoding in https://arxiv.org/pdf/1706.03762.pdf
    Args:
        max_time_step: int, max length of input sequence
        encoding_size: int, output encoding size
    Returns:
        tensor of shape (batch_size, encoding_size)
    """
    pos = tf.expand_dims(tf.range(max_time_step), 0)
    i = tf.expand_dims(tf.range(encoding_size), 1)
    radians = tf.matmul(pos, 10000**(i/encoding_size))
    pe = tf.where(tf.equal(i%2, 0), tf.sin(radians), tf.cos(radians))
    return pe