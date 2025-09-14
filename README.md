# AAI595-Final-Project

This project implements adversarial training for image classification on the CIFAR-100 dataset using PyTorch Lightning. The goal is to train models that are robust to adversarial attacks, specifically using the TRADES loss and Projected Gradient Descent (PGD) adversarial examples. The code supports multi-GPU training and includes an interactive demo.

## Features

- **TRADES Loss**: Balances clean accuracy and adversarial robustness.
- **PGD Attack**: Generates adversarial examples during training and evaluation.
- **Multi-GPU Training**: Efficient distributed training with PyTorch Lightning.
- **Class-Balanced Loss**: Dynamically adjusts class weights for CIFAR-100.
- **Progressive Training**: Gradually increases adversarial strength during training.
- **Robustness Evaluation**: Consistent evaluation on clean and adversarial samples.
- **Streamlit Demo**: Interactive web app to visualize model predictions and adversarial attacks.
