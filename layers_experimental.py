import tensorflow as tf
from utils import *
from layers import *

def aggregation(inputs,
                output_depth=512,
                scope="aggregation",
                reuse=None):
    """
    Aggregate (add) inputs along the time axis weighed by the covariance of the time steps.
    Args:
        inputs: tensor of shape (batch_size, time_steps, input_depth)
        output_depth: int, depth of the output
    Return:
        tensor of shape (batch_size, 1, input_depth)
    """
    inputs_shape = get_shape(inputs)
    assert len(inputs_shape) == 3
    
    with tf.variable_scope(scope, reuse=reuse):
        projection_1 = tf.layers.dense(inputs, output_depth, name="projection_1")
        projection_2 = tf.layers.dense(inputs, output_depth, name="projection_2")
        similarities = tf.matmul(projection_1, tf.matrix_diag(tf.reduce_mean(projection_1, axis=-2)))
        similarities_scaled = similarities / tf.to_float(inputs_shape[-1])**0.5
        
        weights = tf.nn.softmax(similarities_scaled)
        outputs = tf.nn.relu(tf.reduce_sum(projection_2 * weights, -2, keep_dims=True))

    return outputs


def dispersion(inputs,
               output_length=2,
               output_depth=512,
               scope="dispersion",
               reuse=None):
    output_list = []

    with tf.variable_scope(scope, reuse=reuse):
        for i in range(output_length):
            output_list.append(aggregation(inputs, output_depth=output_depth, scope="aggr_{}".format(i)))
            outputs = tf.concat(output_list, axis=-2)
        
    return outputs


def aggregation_layer(inputs,
                      output_depth=512,
                      aggregation_length=6,
                      aggregation_stride=2,
                      scope="aggregation_layer",
                      use_layer_norm=True):

    inputs_shape = get_shape(inputs)

    length = inputs_shape[1]
    pad_size = int((aggregation_length - aggregation_stride)/2)
    # padding = tf.zeros((tf.to_int32(inputs_shape[0]), pad_size, tf.to_int32(inputs_shape[2])))
    # inputs_padded = tf.concat((padding, inputs, padding), axis=1)

    with tf.variable_scope(scope):
        output_list = []
        for i in range(-pad_size, length - pad_size, aggregation_stride):
            start = max(0, i)
            end = min(length, i + aggregation_length)

            output_list.append(aggregation(
                inputs[:, start:end, :],
                output_depth=output_depth,
                reuse=tf.AUTO_REUSE))
        
        outputs = tf.concat(output_list, axis=-2)

        if use_layer_norm:
            outputs = layer_normalization(outputs)

    return outputs


def dispersion_layer(inputs,
                     output_depth=512,
                     dispersion_input_length=3,
                     dispersion_output_length=2,
                     dispersion_stride=1,
                     scope="dispersion_layer",
                     use_layer_norm=True):
    
    inputs_shape = get_shape(inputs)

    length = inputs_shape[1]
    pad_size = int((dispersion_input_length - dispersion_stride) / 2)
    # padding = tf.zeros((tf.to_int32(inputs_shape[0]), pad_size, tf.to_int32(inputs_shape[2])))
    # inputs_padded = tf.concat((padding, inputs, padding), axis=1)
    
    with tf.variable_scope(scope):
        output_list = []
        for i in range(-pad_size, length - pad_size, dispersion_stride):
            start = max(0, i)
            end = min(length, i + dispersion_input_length)

            output_list.append(dispersion(
                inputs[:, start:end, :],
                output_length=dispersion_output_length,
                output_depth=output_depth,
                reuse=tf.AUTO_REUSE))
        
        outputs = tf.concat(output_list, axis=-2)

        if use_layer_norm:
            outputs = layer_normalization(outputs)

    return outputs
