
# Deep Generative Models

A collection of hands-on implementations and experiments exploring deep generative models, from probabilistic diffusion processes to latent-variable modeling. This repository focuses on understanding the underlying mathematics and building intuition through PyTorch implementations and experiments with Hugging Face Diffusers.

## Notebooks

### 1. `DDPM_Forward_Diffusion_Process.ipynb`

Explores the forward diffusion process in Denoising Diffusion Probabilistic Models (DDPMs). Gaussian noise is progressively added to a data sample over a sequence of timesteps, gradually transforming it into noise. The notebook examines the noise schedule and the mathematical formulation behind this process, building an understanding of how diffusion models construct noisy training samples.

### 2. `DDPM_Reverse_Transition_Kernel.ipynb`

Explores the reverse transition kernel that enables DDPMs to generate new samples. Starting from a noisy sample, the model iteratively estimates and removes noise to recover structured data. The notebook focuses on the probabilistic formulation of reverse transitions and the denoising process that makes generation possible.

### 3. `VAE_training.ipynb`

Implements and trains a Variational Autoencoder (VAE), a generative model that learns a probabilistic latent representation of data. The notebook explores the encoder-decoder architecture, latent-space sampling, reconstruction loss, and KL divergence. It demonstrates how the Evidence Lower Bound (ELBO) objective balances reconstruction quality with regularization of the learned latent distribution.

### 4. `diffusers.ipynb`

Explores diffusion models using the Hugging Face Diffusers library. The notebook provides practical experience with the library's model components and pipelines, connecting the theoretical foundations of diffusion models with their implementation in a reusable framework. It serves as an introduction to working with diffusion-based generative models using pretrained components.

## Concepts Covered

- Denoising Diffusion Probabilistic Models (DDPMs)
- Forward and reverse diffusion processes
- Gaussian noise schedules and probabilistic transitions
- Variational Autoencoders (VAEs) and latent-variable modeling
- Reconstruction loss, KL divergence, and ELBO
- Practical experimentation with Hugging Face Diffusers

## Tech Stack

- Python
- PyTorch
- NumPy
- Matplotlib
- Hugging Face Diffusers
- Jupyter Notebook

## Getting Started

Clone the repository:

```bash
git clone https://github.com/themdanas/Deep_Generative_Models.git
cd Deep_Generative_Models
```

Install the dependencies:

```bash
pip install torch numpy matplotlib diffusers transformers accelerate jupyter
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open a notebook to explore its implementation and experiments. Individual notebooks may require additional dependencies or pretrained model downloads.

## Objective

The goal of this repository is to develop a deeper understanding of generative modeling by connecting mathematical foundations with practical implementations.

---

**Author:** [@themdanas](https://github.com/themdanas)

