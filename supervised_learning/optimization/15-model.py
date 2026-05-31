#!/usr/bin/env python3
"""Build, train, and save a TF1 model with optimization techniques."""

import numpy as np
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data
create_batch_norm_layer = __import__('14-batch_norm').create_batch_norm_layer
create_Adam_op = __import__('10-Adam').create_Adam_op
learning_rate_decay_tf = __import__(
    '12-learning_rate_decay').learning_rate_decay


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow.

    Returns:
    - save_path: path where model was saved
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    m = X_train.shape[0]

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    global_step = tf.Variable(0, trainable=False)
    alpha_t = learning_rate_decay_tf(alpha, decay_rate,
                                     global_step, 1)

    A = x
    for i, (nodes, activation) in enumerate(zip(layers, activations)):
        if i == 0:
            A = create_batch_norm_layer(A, nodes, activation)
        else:
            A = create_batch_norm_layer(A, nodes, activation)

    y_pred = A
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    train_op = tf.train.AdamOptimizer(
        learning_rate=alpha_t,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon
    ).minimize(loss, global_step=global_step)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    steps_per_epoch = int(np.ceil(m / batch_size))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs + 1):
            train_cost, train_acc = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train}
            )
            valid_cost, valid_acc = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid}
            )

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_acc))

            if epoch == epochs:
                break

            X_shuff, Y_shuff = shuffle_data(X_train, Y_train)

            for step in range(steps_per_epoch):
                start = step * batch_size
                end = min(start + batch_size, m)
                X_batch = X_shuff[start:end]
                Y_batch = Y_shuff[start:end]

                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if (step + 1) % 100 == 0 or step == steps_per_epoch - 1:
                    step_cost, step_acc = sess.run(
                        [loss, accuracy],
                        feed_dict={x: X_batch, y: Y_batch}
                    )
                    print("\tStep {}:".format(step + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_acc))

        return saver.save(sess, save_path)
