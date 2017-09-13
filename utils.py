import tensorflow as tf

def stack_layers(initial_state, sublayers, num_layers=1, layer_scope_prefix="layer"):
    """
    Stack the layer with the defined sublayer transition functions for num_layers times.
    All sublayers must only accept a single tensor as input and return a single tensor as output,
    and the output from the final sublayer must be same same shape as the input to the first sublayer
    if num_layers is greater than 1.
    Args:
        initial_state: tensor input to the first sublayer
        sublayers: sequential list of sublayer transition functions representing a single layer.
        num_layers: number of times to stack the sublayers.
    Returns:
        tensor outputed by the final sublayer
    """
    transition_state = initial_state
    for i in range(num_layers):
        with tf.variable_scope("{}_{}".format(layer_scope_prefix, i)):
            for sublayer in sublayers:
                transition_state = sublayer(transition_state)
    
    return transition_state

def get_shape(tensor):
  static_shape = tensor.shape.as_list()
  dynamic_shape = tf.unstack(tf.shape(tensor))
  dims = [s[1] if s[0] is None else s[0]
          for s in zip(static_shape, dynamic_shape)]
  return dims