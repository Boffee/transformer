import tensorflow as tf
import numpy as np

def scaled_dot_prod_attention(Q, 
                              K,
                              V,
                              attention_bias=0,
                              scope="scaled_dot_prod_attention",
                              reuse=None):
    """
    Dot-product attention scaled by sqrt(depth(K))
    Args:
        Q: Tensor of shape (N, t_q, d_qk), the queries.
        K: Tensor of shape (N, t_kv, d_qk), the keys for the values.
        V: Tensor of shape (N, t_kv, d_v), the values queried for.
        attention_bias: Tensor of shape (N, t_q, t_kv), the bias for the attention. Can be used to mask.
        scope: str, variable scope.
        reuse: bool, reuse variables.
    Return:
        Tensor of shape (N, t_q, d_v), the queried values.
    """
    d_q = Q.get_shape().as_list()[-1]
    d_k = K.get_shape().as_list()[-1]
    t_k = K.get_shape().as_list()[-2]
    t_v = V.get_shape().as_list()[-2]
    
    assert d_q == d_k, "Query depth {} does not match with key depth {}.".format(d_q, d_k)
    assert t_k == t_v, "Key steps {} does not match value steps {}.".format(t_k, t_v)
    
    with tf.variable_scope(scope, reuse=reuse):
        # (N, t_q, t_v)
        attention = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
        attention = attention/(K.get_shape().as_list()[-1]**.5)
        attention = tf.nn.softmax(attention + attention_bias, axis=-1)

        # (N, t_q, d_v)
        output = tf.matmul(attention, V)
    
    return output

def single_head_attention(Q, 
                          K,
                          V,
                          attention_bias=0,
                          attention_depth=64,
                          scope="single_head_attention",
                          reuse=None):
    """
    Linear projection layer followed by scaled dot product attention.
    Args:
        Q: Tensor of shape (N, t_q, d_qk), the queries.
        K: Tensor of shape (N, t_kv, d_qk), the keys for the values.
        V: Tensor of shape (N, t_kv, d_v), the values queried for.
        attention_bias: Tensor of shape (N, t_q, t_kv), the bias for the attention. Can be used to mask.
        attention_depth: int, depth of projection and output.
        scope: str, variable scope.
        reuse: bool, reuse variables.
    Return:
        Tensor of shape (N, t_q, attention_depth), the queried values.
    """
    with tf.variable_scope(scope, reuse=reuse):
        Q_p = tf.layers.dense(Q, attention_depth)
        K_p = tf.layers.dense(K, attention_depth)
        V_p = tf.layers.dense(V, attention_depth)
        output = scaled_dot_prod_attention(Q_p, K_p, V_p, attention_bias)
        
    return output

def multi_head_attention(Q,
                         K,
                         V, 
                         attention_bias=0,
                         attention_depth=512,
                         num_heads=8,
                         gpus=None,
                         scope="multi_head_attention",
                         reuse=None):
    """
    Partition Q, K, and V into multiple heads and run single head attention on each. Concat
    the outputs from the heads and linearly project the output.
    Args:
        Q: Tensor of shape (N, t_q, d_qk), the queries.
        K: Tensor of shape (N, t_kv, d_qk), the keys for the values.
        V: Tensor of shape (N, t_kv, d_v), the values queried for.
        attention_bias: Tensor of shape (N, t_q, t_kv), the bias for the attention. Used as mask.
        attention_depth: int, depth of projection and output.
        num_heads: int, number of attention heads.
        gpus: list of strings, gpus to distribute the heads on.
        scope: str, variable scope.
        reuse: bool, reuse variables.
    Return:
        Tensor of shape (N, t_q, attention_depth), the queried values.
    """
    d_q = Q.get_shape().as_list()[-1]
    d_k = K.get_shape().as_list()[-1]
    d_v = V.get_shape().as_list()[-1]
    
    assert d_q%num_heads == 0, "Query depth {} is not divisible by {} heads.".format(d_q, num_heads)
    assert d_k%num_heads == 0, "Key depth {} is not divisible by {} heads.".format(d_k, num_heads)
    assert d_v%num_heads == 0, "Value depth {} is not divisible by {} heads.".format(d_v, num_heads)
    assert attention_depth%num_heads == 0, "Attention depth {} is not divisible by {} heads.".format(attention_depth, num_heads)

    head_depth = attention_depth/num_heads
    
    with tf.variable_scope(scope, reuse=reuse):
        Q_s = tf.split(Q, num_heads, axis=-1)
        K_s = tf.split(K, num_heads, axis=-1)
        V_s = tf.split(V, num_heads, axis=-1)
        
        heads = []
        for i, (Q_i, K_i, V_i) in enumerate(zip(Q_s, K_s, V_s)):
            if gpus:
                with tf.device(gpus[i%len(gpus)]):
                    head = single_head_attention(Q_i, K_i, V_i, attention_bias, attention_depth=head_depth, scope="head_{}".format(i))
            else:
                head = single_head_attention(Q_i, K_i, V_i, attention_bias, attention_depth=head_depth, scope="head_{}".format(i))

            heads.append(head)
        
        output = tf.concat(heads, axis=-1)
        output = tf.layers.dense(output, attention_depth)
        
    return output

def feed_forward(X,
                 hidden_dim=2048,
                 output_dim=512,
                 axis=-1,
                 activation=tf.nn.relu,
                 scope="feed_forward",
                 reuse=None):
    """
    Linear projection -> activation -> linear projection on specified axis
    Args:
        X: N-D tensor, input to feed forward network.
        hidden_dim: int, dim of axis hidden layer.
        output_dim: int, dim of axis output layer.
        axis: int, axis to apply transformations to.
        scope: str, variable scope.
        reuse: bool, reuse variables.
    Return:
        N-D tensor with same shape as X, except for the specified axis,
        which has output_dim values.
    """
    
    with tf.variable_scope(scope, reuse=reuse):
        hidden = tf.layers.dense(X, hidden_dim, activation=activation)
        output = tf.layers.dense(hidden, output_dim)
        
    return output

def layer_norm(X,
               norm_axes=[-1],
               params_axes=[-1],
               scope="layer_norm",
               reuse=None):
    """
    https://arxiv.org/pdf/1607.06450.pdf
    Args:
        X: X: N-D tensor, input tensor to normalize.
        norm_axes: list of int, axes to compute moments for normalization.
        scope: str, variable scope.
        reuse: bool, reuse variables.
    Return:
        Normalized tensor of same shape as X.
    """
    X_shape = X.get_shape().as_list()
    params_shape = [X_shape[axis] for axis in params_axes]
    
    with tf.variable_scope(scope, reuse=reuse):
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        mean, var = tf.nn.moments(X, norm_axes)
        var_epsilon = 1e-12
        output = tf.nn.batch_normalization(X, mean, var, beta, gamma, var_epsilon)
        
    return output

def attention_bias(Q,
                   K,
                   scope="attention_bias",
                   reuse=None):
    """
    Bias for scaled dot-product attention to mask decoder self-attention to only attend to preceding words.
    Args:
        Q: Tensor of shape (N, t_q, d_qk), the queries.
        K: Tensor of shape (N, t_kv, d_qk), the keys for the values.
        scope: str, variable scope.
        reuse: bool, reuse variables.
    Return:
        Tensor of shape (t_q, t_kv) with the diagonal and above set to -1e9
    """
    with tf.variable_scope(scope, reuse=reuse):
        t_q = Q.get_shape().as_list()[-2]
        t_k = K.get_shape().as_list()[-2]
        output = -1e9 * (1 - tf.matrix_band_part(tf.ones((t_q, t_k)), -1, 0))
    
    return output

def positional_encoding(num_positions, 
                        embedding_depth,
                        scope="positional_encoding",
                        resuse=None):
    """
    Sinusoidal positional encoding for the absolute and relative position of items in the sequence.
    Args:
        num_positions: int, number of positions.
        embedding_depth: int, number values in the encoding for each position.
        scope: str, variable scope.
        reuse: bool, reuse variables.
    Return:
        Tensor of shape (N, num_positions, embedding_depth)
    """
    with tf.variable_scope(scope, resuse=reuse):
        pos = tf.expand_dims(tf.range(num_positions), 0)
        dim = 1/10000**tf.expand_dims(2*tf.range(embedding_depth)/embedding_depth, 1)
        stripe[::2,:] = np.pi/2
        pos_enc = tf.sin(tf.matmul(dim, pos) + stripe)
        
    return pos_enc

def residual(curr_layer,
             prev_layer,
             project=False,
             axis=-1,
             scope="residual",
             reuse=None):
    """
    Add previous layer to current layer. If project, previous layer will be linearly projected to
    match the depth of the current layer.
    Args:
        curr_layer: n-D Tensor, current layer.
        prev_layer: n-D Tensor, previous layer.
        project: bool, flag to project previous layer.
        axis: int, axis to project.
        scope: str, variable scope.
        reuse: bool, reuse variables.
    Return:
        Tensor with same shape as curr_layer.
    """
    with tf.variable_scope(scope, reuse=reuse):
        if project:
            output = tf.layers.dense(prev_layer, curr_layer.get_shape().as_list()[axis]) + curr_layer
        else:
            output = prev_layer + curr_layer
    
    return output


def transformer_encoder(X,
                        attention_bias=0,
                        attention_depth=512,
                        ff_hidden_depth=2048,
                        num_heads=8,
                        gpus=None,
                        scope="transformer_encoder",
                        reuse=None):
    """
    Transformer encoder layer from https://arxiv.org/pdf/1706.03762.pdf.
    Args:
        X: Tensor of shape (N, max_steps, depth), input source sequences.
        attention_bias: Tensor of shape (N, t_q, t_kv), the bias for the attention. Used as mask.
        attention_depth: int, depth for attention layers.
        ff_hidden_depth: int, depth for feedforward hidden layer.
        num_heads: int, number of attention heads.
        gpus: list of strings, gpus to distribute the heads on.
        scope: str, variable scope.
        reuse: bool, reuse variables.
    Return:
        Tensor of shape (N, max_steps, attention_depth), encoded input.
    """
    with tf.variable_scope(scope, reuse=reuse):
        self_attention = multi_head_attention(X,
                                              X,
                                              X,
                                              attention_bias=attention_bias,
                                              attention_depth=attention_depth,
                                              num_heads=num_heads,
                                              gpus=gpus)
        addnorm_1 = layer_norm(residual(self_attention, X, scope="res_1"), scope="ln_1")
        
        ff = feed_forward(addnorm_2, hidden_dim=ff_hidden_depth, output_dim=attention_depth)
        addnorm_2 = layer_norm(residual(ff, addnorm_1, scope="res_2"), scope="ln_2")
        
    return addnorm_2


def transformer_decoder(Y,
                        encoding,
                        self_attention_bias=0,
                        encoding_attention_bias=0,
                        attention_depth=512,
                        ff_hidden_depth=2048,
                        num_heads=8,
                        gpus=None,
                        scope="transformer_decoder",
                        reuse=None):
    """
    Transformer encoder layer from https://arxiv.org/pdf/1706.03762.pdf.
    Args:
        Y: Tensor of shape (N, max_steps, depth), input target sequences.
        encoding: Tensor of shape (N, enc_max_steps, enc_depth), source sequence encodings.
        self_attention_bias: Tensor of shape (N, t_q, t_kv), the bias for self attention. Used as mask.
        encoding_attention_bias: Tensor of shape (N, t_q, t_kv), the bias for encoding attention. Used as mask.
        attention_depth: int, depth for attention layers.
        ff_hidden_depth: int, depth for feedforward hidden layer.
        num_heads: int, number of attention heads.
        gpus: list of strings, gpus to distribute the heads on.
        scope: str, variable scope.
        reuse: bool, reuse variables.
    Return:
        Tensor of shape (N, max_steps, attention_depth), encoded input.
    """
    with tf.variable_scope(scope, reuse=reuse):
        self_attention = multi_head_attention(Y,
                                              Y,
                                              Y,
                                              attention_bias=self_attention_bias,
                                              attention_depth=attention_depth,
                                              num_heads=num_heads,
                                              gpus=gpus)
        addnorm_1 = layer_norm(residual(self_attention, X, scope="res_1"), scope="ln_1")
        
        encoding_attention = multi_head_attention(addnorm_1,
                                                  encoding,
                                                  encoding,
                                                  attention_bias=encoding_attention_bias,
                                                  attention_depth=attention_depth,
                                                  num_heads=num_heads,
                                                  gpus=gpus)
        
        addnorm_2 = layer_norm(residual(encoding_attention, addnorm_1, scope="res_2"), scope="ln_2")

        ff = feed_forward(addnorm_2, hidden_dim=ff_hidden_depth, output_dim=attention_depth)
        addnorm_3 = layer_norm(residual(ff, addnorm_2, scope="res_3"), scope="ln_3")

    return addnorm_3

def embedding(sequences,
              vocab_size,
              embedding_depth,
              scope="embedding",
              reuse=None):
    """
    map sequences of indices to sequence of vectors.
    Args:
        sequences: int Tensor of shape (N, max_steps), input sequences of indices.
        vocab_size: int, number of index to vector mappings.
        embedding_depth: int, size of embedding vectors.
        scope: str, variable scope.
        reuse: bool, reuse variables.
    Returns:
        Tensor of shape (N, max_steps, embedding_depth)
    """
    with tf.variable_scope(scope, reuse=reuse):
        embedding_kernel = tf.get_variable("kernel", 
                                           shape=[vocab_size, embedding_depth], 
                                           initializer=tf.contrib.layers.xavier_initializer())
        embeddings = tf.nn.embedding_lookup(embedding_kernel, sequences)
    
    return embeddings