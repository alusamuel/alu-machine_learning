#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


class NST:
    """Neural Style Transfer class."""

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """Initialize NST with style and content images."""
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta
        self.model = self.load_model()
        self.generate_features()

    @staticmethod
    def scale_image(image):
        """Rescale image so max side is 512 and pixel values are in [0, 1]."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        if max(h, w) > 512:
            if h >= w:
                new_h = 512
                new_w = int(w * (512 / h))
            else:
                new_w = 512
                new_h = int(h * (512 / w))
        else:
            new_h, new_w = h, w

        image = tf.image.resize_images(
            image[np.newaxis, ...],
            [new_h, new_w],
            method=tf.image.ResizeMethod.BICUBIC,
            align_corners=False
        )
        image = image / 255.0
        return image

    def load_model(self):
        """Load VGG19 and create model returning style + content layer outputs."""
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        model = tf.keras.Model(inputs=vgg.input, outputs=style_outputs + [content_output])
        return model

    @staticmethod
    def gram_matrix(input_layer):
        """Compute Gram matrix for a feature map."""
        if not isinstance(input_layer, (tf.Tensor, tf.Variable)) or input_layer.shape.ndims != 4:
            raise TypeError("input_layer must be a tensor of rank 4")

        _, h, w, c = input_layer.shape
        features = tf.reshape(input_layer, [h * w, c])
        gram = tf.matmul(tf.transpose(features), features)
        gram = gram / tf.cast(h * w, tf.float32)
        return gram[tf.newaxis, ...]

    def generate_features(self):
        """Extract style Gram matrices and content feature."""
        vgg19 = tf.keras.applications.vgg19
        style_input = vgg19.preprocess_input(self.style_image * 255)
        content_input = vgg19.preprocess_input(self.content_image * 255)

        style_outputs = self.model(style_input)[:-1]
        self.gram_style_features = [self.gram_matrix(out) for out in style_outputs]

        content_outputs = self.model(content_input)
        self.content_feature = content_outputs[-1]

    def layer_style_cost(self, style_output, gram_target):
        """Compute style cost for a single layer."""
        if not isinstance(style_output, (tf.Tensor, tf.Variable)) or style_output.shape.ndims != 4:
            raise TypeError("style_output must be a tensor of rank 4")

        _, _, _, c = style_output.shape
        expected_shape = [1, c, c]
        if not isinstance(gram_target, (tf.Tensor, tf.Variable)) or list(gram_target.shape) != expected_shape:
            raise TypeError(f"gram_target must be a tensor of shape [1, {c}, {c}]")

        gram_style = self.gram_matrix(style_output)
        diff = gram_style - gram_target
        cost = tf.reduce_sum(tf.square(diff)) / (4 * tf.cast(c ** 2, tf.float32))
        return cost

    def style_cost(self, style_outputs):
        """Compute total style cost across all style layers."""
        if not isinstance(style_outputs, list) or len(style_outputs) != len(self.style_layers):
            raise TypeError(f"style_outputs must be a list with a length of {len(self.style_layers)}")

        num_layers = len(style_outputs)
        weights = [1.0 / num_layers] * num_layers

        total_cost = 0.0
        for w, out, gram in zip(weights, style_outputs, self.gram_style_features):
            total_cost += w * self.layer_style_cost(out, gram)
        return total_cost

    def content_cost(self, content_output):
        """Compute content cost."""
        if not isinstance(content_output, (tf.Tensor, tf.Variable)) or content_output.shape != self.content_feature.shape:
            raise TypeError(f"content_output must be a tensor of shape {self.content_feature.shape}")

        h, w, c = self.content_feature.shape[1:]
        diff = content_output - self.content_feature
        cost = tf.reduce_sum(tf.square(diff)) / (4 * tf.cast(h * w * c, tf.float32))
        return cost

    def total_cost(self, generated_image):
        """Compute total cost J = alpha*J_content + beta*J_style."""
        if not isinstance(generated_image, (tf.Tensor, tf.Variable)) or generated_image.shape != self.content_image.shape:
            raise TypeError(f"generated_image must be a tensor of shape {self.content_image.shape}")

        vgg19 = tf.keras.applications.vgg19
        preprocessed = vgg19.preprocess_input(generated_image * 255)
        outputs = self.model(preprocessed)

        style_outputs = outputs[:-1]
        content_output = outputs[-1]

        J_content = self.content_cost(content_output)
        J_style = self.style_cost(style_outputs)
        J_total = self.alpha * J_content + self.beta * J_style
        return J_total, J_content, J_style

    def compute_grads(self, generated_image):
        """Compute gradients of total cost w.r.t. generated image."""
        with tf.GradientTape() as tape:
            tape.watch(generated_image)
            J_total, J_content, J_style = self.total_cost(generated_image)
        grads = tape.gradient(J_total, generated_image)
        return grads, J_total, J_content, J_style

    def generate_image(self, iterations=1000, step=None, lr=0.01, beta1=0.9, beta2=0.99):
        """Generate NST image using Adam optimization."""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be positive")
        if step is not None and not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step is not None and (step <= 0 or step >= iterations):
            raise ValueError("step must be positive and less than iterations")
        if not isinstance(lr, (int, float)):
            raise TypeError("lr must be a number")
        if lr <= 0:
            raise ValueError("lr must be positive")
        if not isinstance(beta1, float):
            raise TypeError("beta1 must be a float")
        if not (0 <= beta1 <= 1):
            raise ValueError("beta1 must be in the range [0, 1]")
        if not isinstance(beta2, float):
            raise TypeError("beta2 must be a float")
        if not (0 <= beta2 <= 1):
            raise ValueError("beta2 must be in the range [0, 1]")

        generated_image = tf.contrib.eager.Variable(self.content_image)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)

        best_cost = float('inf')
        best_image = self.content_image

        for i in range(iterations + 1):
            grads, J_total, J_content, J_style = self.compute_grads(generated_image)
            optimizer.apply_gradients([(grads, generated_image)])
            generated_image.assign(tf.clip_by_value(generated_image, 0, 1))

            if J_total < best_cost:
                best_cost = J_total
                best_image = generated_image.numpy()

            if step is not None and (i % step == 0 or i == iterations):
                print(f"Cost at iteration {i}: {J_total.numpy()}, content {J_content.numpy()}, style {J_style.numpy()}")

        return best_image, best_cost