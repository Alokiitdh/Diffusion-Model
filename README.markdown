# DDPM Anime Face Generator 🎨

*A PyTorch implementation of Denoising Diffusion Probabilistic Models for Anime Face Generation*

This project uses a **Denoising Diffusion Probabilistic Model (DDPM)** to generate high-quality anime faces from random noise. Built with PyTorch, it employs a **U-Net architecture** to iteratively denoise images, trained on an anime face dataset to produce vibrant, detailed results.

---

## 🚀 Features

* **Noise-to-Image Generation**: Transform random noise into anime faces.
* **U-Net Denoising**: Predicts and removes noise at each timestep.
* **Customizable Schedules**: Control the noise addition process.
* **Training Pipeline**: Train the model on anime face datasets.
* **Visualization**: View the denoising process and final outputs.

---

## 📜 Theory — How DDPM Works

Denoising Diffusion Probabilistic Models (DDPMs) are generative models that create images by learning to reverse a process of gradually adding noise. Here’s a breakdown of the approach:

### **Forward Process (Adding Noise)**

The process begins with a real image, such as an anime face, denoted as $x_0$. Over a series of $T$ steps, small amounts of Gaussian noise are added to the image:

- At each step $t$, the image $x_{t-1}$ is transformed into $x_t$ by adding noise scaled by a variance schedule $\beta_t$:

  $$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)$$

- After many steps, the image $x_T$ approximates isotropic Gaussian noise. A key property allows direct sampling at any timestep $t$:

  $$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1-\bar{\alpha}_t) I)$$

  where $${\alpha}_t = \prod_{s=1}^t (1 - \beta_s)$$
.

### **Reverse Process (Denoising)**

The goal is to reverse this process, starting from noise $x_T$ and reconstructing a clean image $x_0$. The reverse process is modeled as:

  $$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(t))$$

A U-Net predicts the mean $\mu_\theta$ by estimating the noise component $\epsilon_\theta(x_t, t)$ in the noisy image $x_t$ at timestep $t$. The variance $\Sigma_\theta(t)$ is typically fixed based on $\beta_t$.

### **Training**

The model is trained to predict the noise added at each timestep. The loss function is the mean squared error between the true noise $\epsilon$ and the predicted noise $\epsilon_\theta$:

  $$L = \mathbb{E}_{x_0, t, \epsilon} \left[ \|\epsilon - \epsilon_\theta(x_t, t)\|^2 \right]$$

This simplifies training, as predicting noise is more stable than directly reconstructing the image.

### **Sampling**

To generate an image:
- Start with random noise $x_T \sim \mathcal{N}(0, I)$.
- Iteratively denoise using the learned model:

  $$x_{t-1} = \frac{1}{\sqrt{1-\beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z$$

  where $z$ is random Gaussian noise, and $\sigma_t$ is derived from the variance schedule.

### **Why It Works**

DDPMs excel because the forward process is deterministic and simple, while the reverse process leverages the U-Net’s ability to learn complex patterns, enabling high-quality image generation with stable training.

---

## 📊 Dataset

* [Kaggle Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset)

---

## 🖼 Example Outputs

* **Noisy Start**: Random Gaussian noise as input.
* **Denoised Output**: Clean anime faces after iterative denoising.

---

## 🛠 Customization

* **Variance Schedule**: Adjust $\beta_t$ for different noise patterns.
* **U-Net Architecture**: Modify layers or channels for performance.
* **Training Parameters**: Tune learning rate and epochs.

---

## 📚 References

* Denoising Diffusion Probabilistic Models (Ho et al., 2020): [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)
* Improved Denoising Diffusion Probabilistic Models (Nichol & Dhariwal, 2021): [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)
* U-Net: Convolutional Networks for Biomedical Image Segmentation: [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

---

## 📄 License

MIT License — Free to use and modify.