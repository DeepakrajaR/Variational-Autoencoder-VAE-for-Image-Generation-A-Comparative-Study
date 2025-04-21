# Variational Autoencoder (VAE) for Image Generation: A Comparative Study

## Project Overview

This repository contains a comprehensive implementation of Variational Autoencoders (VAEs) applied to two different image datasets: CIFAR-10 and Fashion-MNIST. The project demonstrates how dataset characteristics affect generative model behavior and performance through detailed analysis and visualization.

## Key Features

- Complete implementation of VAE models in TensorFlow/Keras
- Comparative analysis between color (CIFAR-10) and grayscale (Fashion-MNIST) datasets
- Exploration of latent space dimensionality and its impact on model performance
- Implementation of β-VAE for investigating disentangled representations
- Extensive visualizations of reconstructions, latent space organization, and model performance
- Detailed evaluation metrics and loss function analysis

## Datasets

- **CIFAR-10**: 60,000 32×32 color images across 10 classes (airplanes, cars, birds, cats, etc.)
- **Fashion-MNIST**: 70,000 28×28 grayscale images of fashion items across 10 classes (T-shirts, trousers, dresses, etc.)

## Key Findings

- Dataset complexity significantly impacts VAE performance, with Fashion-MNIST being easier to model than CIFAR-10
- Larger latent dimensions improve reconstruction quality but reduce compression efficiency
- The organization of the latent space reflects the inherent structure of the data
- β-VAE demonstrates the trade-off between reconstruction quality and latent space organization

## Technical Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Scikit-learn
- Pandas
- Seaborn

## Getting Started

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/vae-image-generation.git
   cd vae-image-generation
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Jupyter notebook:
   ```bash
   jupyter notebook VAE_Image_Generation_Comparative_Study.ipynb
   ```

## Structure

- `VAE_Image_Generation_Comparative_Study.ipynb`: Main Jupyter notebook with all code and analysis
- `requirements.txt`: Required Python packages
- `images/`: Directory containing saved visualizations and results
- `models/`: Directory for saving trained model weights (created during execution)

## Extensions and Exercises

The notebook includes three in-depth exercises that build upon the base implementation:

1. **Latent Space Exploration and Conditional Generation**: Implement techniques for controlling the generation process by manipulating the latent space
2. **Exploring the Effect of Latent Dimensionality**: Investigate how the dimensionality of the latent space affects model performance
3. **Implementing β-VAE for Disentangled Representations**: Modify the VAE implementation to include a β parameter controlling the KL divergence weight

## Results Preview

The implementation achieves successful image generation and reconstruction on both datasets, with better performance on Fashion-MNIST due to its simpler structure. Visualizations demonstrate the trade-offs between latent dimensionality, reconstruction quality, and latent space organization.

## Future Work

Potential improvements and extensions include:
- Implementing more advanced architectures (deeper networks, residual connections)
- Exploring alternative approaches to latent space modeling (normalizing flows, hierarchical structures)
- Implementing advanced training strategies (KL annealing, perceptual loss functions)

## References

- Kingma, D. P., & Welling, M. (2014). Auto-encoding variational Bayes. *International Conference on Learning Representations (ICLR)*.
- Higgins, I., et al. (2017). β-VAE: Learning basic visual concepts with a constrained variational framework. *ICLR*.
- Krizhevsky, A., & Hinton, G. (2009). Learning multiple layers of features from tiny images. *Technical report, University of Toronto*.
- Xiao, H., Rasul, K., & Vollgraf, R. (2017). Fashion-MNIST: A novel image dataset for benchmarking machine learning algorithms. *arXiv preprint arXiv:1708.07747*.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
