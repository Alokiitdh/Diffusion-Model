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

Denoising Diffusion Probabilistic Models (DDPMs) are generative models that create images by learning to reverse a gradual noising process.

### **Forward Process (Adding Noise)**

The process starts with a real image. Over many steps, small amounts of Gaussian noise are added, controlled by a variance schedule. Eventually, the image becomes pure noise. Thanks to the process’s mathematical properties, noise can be applied in a single step for any point in the sequence.

### **Reverse Process (Denoising)**

The model’s task is to reverse the noising process—starting from pure noise and step-by-step reconstructing the original image. This is achieved using a U-Net that predicts the noise present at each step, allowing the model to recover a cleaner image at the previous step.

### **Training**

The model is trained to predict the exact noise added at each step. This is done by comparing the true noise with the predicted noise, and adjusting the model to minimize the difference. Predicting noise directly is more stable than reconstructing the entire image.

### **Sampling**

To generate an image, the process starts with random noise and repeatedly applies the learned denoising steps until a coherent image emerges.

### **Why It Works**

DDPMs are effective because the forward process is simple and well-defined, while the reverse process leverages the U-Net’s ability to learn detailed structures. This combination produces high-quality, stable image generation.

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