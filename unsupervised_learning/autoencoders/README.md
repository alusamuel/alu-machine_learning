# Autoencoders

This project implements several kinds of autoencoders using `tensorflow.keras` on the MNIST dataset [page:2]. The goal is to understand how autoencoders learn compressed latent representations and how different architectures (sparse, convolutional, variational) behave [page:2].

## Project structure

Repository layout for this project:

```text
alu-machine_learning/
└── unsupervised_learning/
    └── autoencoders/
        ├── 0-vanilla.py
        ├── 1-sparse.py
        ├── 2-convolutional.py
        ├── 3-variational.py
        ├── 0-main.py
        ├── 1-main.py
        ├── 2-main.py
        └── 3-main.py
```

The `*_main.py` files are provided by the project to train and visualize each model on MNIST [page:2].

## Requirements

According to the project instructions [page:2]:

- All code is run on **Ubuntu 16.04** with **python3.5**, `numpy 1.15`, `tensorflow 1.12` and `tensorflow.keras` [page:2].  
- Only `import tensorflow.keras as keras` (plus standard libs used in the mains, like `numpy`, `matplotlib`, and `tensorflow`) is allowed in your project files [page:2].  
- All files must:
  - start with `#!/usr/bin/env python3`  
  - be executable and end with a new line  
  - follow `pycodestyle` 2.4  
  - include module, class, and function docstrings [page:2].  
- A `README.md` at the **root of the project repository** is mandatory [page:2].

## Files

### 0-vanilla.py — Vanilla autoencoder

Implements:

```python
def autoencoder(input_dims, hidden_layers, latent_dims):
    ...
```

- `input_dims`: integer dimensionality of the flattened input (784 for MNIST) [page:2].  
- `hidden_layers`: list of encoder hidden layer sizes; the decoder mirrors this list in reverse [page:2].  
- `latent_dims`: integer size of the latent space (bottleneck) [page:2].  

The function returns `(encoder, decoder, auto)` where:

- `encoder`: maps inputs to the latent representation.  
- `decoder`: reconstructs inputs from latent vectors.  
- `auto`: full autoencoder `encoder ∘ decoder` compiled with **Adam** and **binary cross-entropy** [page:2].  

All layers use **ReLU** except the last decoder layer, which uses **sigmoid** [page:2]. `0-main.py` trains the model on MNIST and shows original vs reconstructed digits [page:2].

### 1-sparse.py — Sparse autoencoder

Implements:

```python
def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    ...
```

- Same arguments as the vanilla autoencoder plus `lambtha`, the **L1 regularization** parameter applied to the encoded output to enforce sparsity [page:2].  
- Returns `(encoder, decoder, auto)` analogous to the vanilla model [page:2].  

The autoencoder is compiled with Adam and binary cross-entropy and uses ReLU activations except for the last decoder layer (sigmoid) [page:2]. `1-main.py` trains the sparse autoencoder on MNIST and visualizes reconstructions and the effect of sparsity [page:2].

### 2-convolutional.py — Convolutional autoencoder

Implements:

```python
def autoencoder(input_dims, filters, latent_dims):
    ...
```

- `input_dims`: tuple `(H, W, C)` for the input image size, e.g. `(28, 28, 1)` for MNIST [page:2].  
- `filters`: list of filter counts for each convolutional layer in the encoder; the decoder reverses this list [page:2].  
- `latent_dims`: tuple for the spatial latent representation shape [page:2].  

Encoder:

- For each `f` in `filters`, applies `Conv2D(f, (3, 3), padding='same', activation='relu')` followed by `MaxPooling2D((2, 2), padding='same')` [page:2].  

Decoder:

- Mirrors the encoder using `Conv2D` + `UpSampling2D((2, 2))` for each layer except the last two [page:2].  
- The **second to last** convolution uses **valid** padding instead of same [page:2].  
- The **last** convolution uses `filters = input_dims[-1]`, kernel `(3, 3)`, `sigmoid` activation, and **no upsampling** [page:2].  

The model is compiled with Adam and binary cross-entropy [page:2]. `2-main.py` trains the convolutional autoencoder on MNIST images and shows original vs reconstructed digits [page:2].

### 3-variational.py — Variational autoencoder (VAE)

Implements:

```python
def autoencoder(input_dims, hidden_layers, latent_dims):
    ...
```

- `input_dims`: integer dimensionality of the input (784 for flattened MNIST) [page:2].  
- `hidden_layers`: list of encoder hidden layer sizes; the decoder reverses this list [page:2].  
- `latent_dims`: integer dimensionality of the latent space, typically small (e.g. 2) for visualization [page:2].  

Encoder outputs three tensors:

- `z`: sampled latent vector via the **reparameterization trick**.  
- `z_mean`: mean of the latent Gaussian [page:2].  
- `z_log_var`: log variance of the latent Gaussian [page:2].  

Decoder maps latent vectors back to reconstructed inputs with ReLU hidden layers and a final sigmoid output layer [page:2]. The VAE loss combines:

- Reconstruction loss: binary cross-entropy between inputs and reconstructions.  
- KL divergence: regularization term pushing the latent distribution towards a unit Gaussian [page:2].  

`3-main.py` trains the VAE on MNIST, prints means and standard deviations of the latent codes for sample digits, and visualizes both reconstructions and a 2D latent manifold of generated digits [page:2].

## How to run

From the `unsupervised_learning/autoencoders` directory:

```bash
# Vanilla autoencoder
./0-main.py

# Sparse autoencoder
./1-main.py

# Convolutional autoencoder
./2-main.py

# Variational autoencoder
./3-main.py
```

Each script downloads MNIST, normalizes the data, trains the corresponding model, prints summary statistics (e.g. loss, latent statistics), and displays original vs reconstructed images using `matplotlib` [page:2].