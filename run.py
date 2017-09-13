import tensorflow as tf

def train(hp):
    # load train data
    src, tgt = load_train_data()

    # create model on data
    logits = create_model(src, tgt, training=True)

    # compute loss
    loss = compute_loss(tgt, logits)

    # minmize loss
    train_vars = tf.trainable_variables()
    train_op = miminize(loss, train_vars)
    
    return src, tgt, logits, loss, train_op

def eval(hp):
    # load eval data
    src, tgt = load_eval_data()

    # create eval model
    logits = create_model(src, tgt, training=False, beam_search=True)

    return logits

def summarize(tgt, preds):
    # predictions
    preds = tf.to_int32(tf.argmax(logits, axis=-1))
    
    # accuracy
    accuracy = compute_accuracy(tgt, preds)

    # perplexity
    perplexity = compute_perplexity(tgt, preds)

    # bleu
    bleu = compute_bleu(tgt, preds)

    return preds, accuracy, perplexity, bleu


with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
    if hp.retrain:
        pretrained_saver.restore(sess, tf.train.latest_checkpoint(os.path.join(hp.logdir, "pretrained")))
        init_untrained_variables = tf.global_variables_initializer() # tf.variables_initializer(untrained_variables)
        init_tables = tf.tables_initializer()
        sess.run([init_untrained_variables, init_tables])
    else:
        saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
        init_tables = tf.tables_initializer()
        sess.run([init_tables])
        
    merged_summary = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(hp.logdir, sess.graph)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    while not coord.should_stop():
        fetch_results = sess.run(fetches)
        fetch_dict = dict(zip(fetch_cols, fetch_results))
        step = fetch_dict["global_step"]
        if step % 50 == 0:
            step_summary = sess.run(merged_summary)
            train_writer.add_summary(step_summary, step)
            if step % 200 == 0:
                print("=============================================================")
                print("step {}, total loss: {}".format(fetch_dict["global_step"], fetch_dict["mean_loss"]))
                print("=============================================================")
                print("-- source: ", convert_ind2sent(fetch_dict["src"][0], idx2en))
                print("-- target: ", convert_ind2sent(fetch_dict["tgt"][0], idx2en))
                print("-- prediction: ", convert_ind2sent(fetch_dict["preds"][0], idx2en))
                if step % 3000 == 0:
                    saver.save(sess, hp.logdir + "/model_ckpt_{}".format(step))

    coord.request_stop()
    coord.join(threads)