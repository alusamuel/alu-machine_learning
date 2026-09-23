# Neural Style Transfer

This directory builds the `NST` class step by step. It repaints a content
image in the style of another image, using a VGG19 network pre-trained on
ImageNet (with max pooling replaced by average pooling). Each file adds one
feature to the class from the file before it.

## Files

- `0-neural_style.py`: validates the inputs and scales both images so
  their largest side is 512 pixels and their pixel values are in `[0, 1]`.
- `1-neural_style.py`: `load_model` builds a Keras model that returns the
  style layer outputs followed by the content layer output.
- `2-neural_style.py`: `gram_matrix` computes the Gram matrix of a layer.
- `3-neural_style.py`: `generate_features` extracts the target style Gram
  matrices and the target content feature.
- `4-neural_style.py`: `layer_style_cost` computes the style cost of one
  layer.
- `5-neural_style.py`: `style_cost` averages the style cost over all style
  layers.
- `6-neural_style.py`: `content_cost` computes the content cost.
- `7-neural_style.py`: `total_cost` combines them as
  `J = alpha * J_content + beta * J_style`.
- `8-neural_style.py`: `compute_grads` returns the gradients of the total
  cost with respect to the generated image.
- `9-neural_style.py`: `generate_image` runs Adam gradient descent, starting
  from the content image, and keeps the image with the lowest cost.
- `10-neural_style.py`: adds a variational (total variation) cost, weighted
  by `var`, to reduce noise in the generated image.

## Style layers

- Style: `block1_conv1`, `block2_conv1`, `block3_conv1`, `block4_conv1`,
  `block5_conv1`
- Content: `block5_conv2`
