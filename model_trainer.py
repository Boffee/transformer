import tensorflow as tf
from utils import *

def compute_loss(y, logits, valid_tgt_pos):
    labels = tf.one_hot(y, depth=get_shape(logits)[-1])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    mean_loss = tf.reduce_sum(loss*valid_tgt_pos) / (tf.reduce_sum(valid_tgt_pos))

    return loss, mean_loss

def compute_accuracy(y, preds, valid_tgt_pos):
    accuracy = tf.reduce_sum(tf.to_float(tf.equal(preds, y))*valid_tgt_pos)/ (tf.reduce_sum(valid_tgt_pos))
    return accuracy

def get_valid_pos(ids):
    return tf.to_float(tf.not_equal(ids, 0))

def create_train_op(mean_loss, learning_rate=0.0001, train_vars=None, max_gradient_norm=None):
    if not train_vars:
        train_vars = tf.trainable_variables()
        
    gradients = tf.gradients(mean_loss, train_vars, colocate_gradients_with_ops=True)
    
    if max_gradient_norm:
        gradients, global_clip_norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
        tf.summary.scalar("global_clip_norm", global_clip_norm)
    
    global_step = tf.Variable(0, name='global_step', trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)    
    train_op = optimizer.apply_gradients(zip(gradients, train_vars), global_step=global_step)

    return train_op, global_step
