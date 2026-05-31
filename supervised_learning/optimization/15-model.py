#!/usr/bin/env python3
"""Build, train, and save a TF1 model with optimization techniques."""

import numpy as np
import tensorflow as tf


def shuffle_data(X, Y):
    """
    Shuffle two data matrices in the same order.

    Args:
        X: numpy.ndarray containing input data.
        Y: numpy.ndarray containing labels.

    Returns:
        The shuffled X and Y matrices.
    """
    perm = np.random.permutation(X.shape[0])
    return X[perm], Y[perm]


def create_batch_norm_layer(prev, n, activation):
    """
    Create a dense TensorFlow layer with batch normalization.

    Args:
        prev: activated output tensor from the previous layer.
        n: number of nodes in the layer.
        activation: activation function to apply after normalization.

    Returns:
        The activated output tensor for the layer.
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    dense = tf.layers.Dense(
        units=n,
        kernel_initializer=init,
        use_bias=False
    )
    Z = dense(prev)

    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)
    mean, var = tf.nn.moments(Z, axes=[0])
    Z_norm = tf.nn.batch_normalization(Z, mean, var, beta, gamma, 1e-8)

    if activation is None:
        return Z_norm
    return activation(Z_norm)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Create an inverse time learning rate decay operation.

    Args:
        alpha: original learning rate.
        decay_rate: rate at which the learning rate decays.
        global_step: TensorFlow variable tracking the training step.
        decay_step: number of steps before each decay.

    Returns:
        The decayed learning rate tensor.
    """
    return tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in tensorflow.

    Args:
        Data_train: tuple containing training inputs and labels.
        Data_valid: tuple containing validation inputs and labels.
        layers: list containing the number of nodes in each layer.
        activations: list containing activation functions for each layer.
        alpha: learning rate.
        beta1: Adam first moment weight.
        beta2: Adam second moment weight.
        epsilon: small number to avoid division by zero.
        decay_rate: learning rate decay rate.
        batch_size: number of data points in each mini-batch.
        epochs: number of passes through the training data.
        save_path: path where the model should be saved.

    Returns:
        The path where the model was saved.
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    m = X_train.shape[0]

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    global_step = tf.Variable(0, trainable=False)
    alpha_t = learning_rate_decay(alpha, decay_rate, global_step, 1)

    A = x
    for nodes, activation in zip(layers, activations):
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
    ).minimize(loss)
    step_op = global_step.assign_add(1)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    saver = tf.train.Saver()
    steps_per_epoch = m // batch_size
    if m % batch_size:
        steps_per_epoch += 1

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

                if (step + 1) % 100 == 0:
                    step_cost, step_acc = sess.run(
                        [loss, accuracy],
                        feed_dict={x: X_batch, y: Y_batch}
                    )
                    print("\tStep {}:".format(step + 1))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_acc))

            sess.run(step_op)

        return saver.save(sess, save_path)
