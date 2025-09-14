# AAI595-Final-Project

This project provides a comprehensive exploration into the vulnerability of deep neural networks to adversarial attacks and details the implementation of a robust defense mechanism through adversarial training. Using the CIFAR-100 dataset, we first demonstrate how imperceptible perturbations can deceive a standard ResNet-18 classifier. We then build a more resilient model using the TRADES (TRadeoff-inspired Adversarial Defense via Surrogate-loss) framework, quantitatively proving its enhanced security.

The entire project is implemented in PyTorch and PyTorch Lightning, culminating in an interactive Streamlit application that allows for real-time visualization of these concepts.

## Features

- **TRADES Loss**: Implements the TRadeoff-inspired Adversarial Defense via Surrogate-loss to effectively balance model performance on clean images and robustness against adversarial examples.

- **PGD Attack Engine**: A robust implementation of the Projected Gradient Descent (PGD) attack capable of generating both untargeted (cause any misclassification) and targeted (force a specific incorrect class) adversarial examples.

- **Multi-GPU Training**: Built with PyTorch Lightning to support efficient, distributed training. The experiments for this project were conducted on two NVIDIA RTX GPUs at Stevens Institute of Technology.

- **Class-Balanced Loss**: Dynamically adjusts class weights during training to handle the class imbalance in the CIFAR-100 dataset, improving overall model performance.

- **Comprehensive Evaluation**: Provides a detailed quantitative analysis of model performance through key metrics:

    - **Clean Accuracy**: Standard accuracy on the original, unperturbed test set.
    - **Attack Success Rate**: The percentage of adversarial examples that successfully fool the model. A lower success rate signifies higher robustness.

- **Interactive Demo**: A web application built with Streamlit that allows for real-time, visual comparison of the baseline and robust models against user-uploaded images and live PGD attacks.

## Workflow and Results

The project follows a structured set of experiments to clearly demonstrate the problem of adversarial vulnerability and the effectiveness of our proposed solution.

1. The Fragile Baseline: Standard ResNet-18 ([1.Baseline.ipynb](./1.Baseline.ipynb))
    - Description: A standard ResNet-18 model was trained on the CIFAR-100 dataset without any adversarial considerations. It serves as our baseline to measure vulnerability.

    - Metrics:
        - Clean Accuracy: 68.57%

2. Breaking the Baseline: PGD Attack Analysis ([2.PGD_Attack.ipynb](./2.PGD_Attack.ipynb))
    - Description: The baseline ResNet-18 was subjected to strong PGD attacks to quantify its vulnerability.

    - Results: The model proved to be extremely susceptible to adversarial perturbations. An attacker can successfully fool the baseline model more than 75% of the time, highlighting a significant security risk.

    - Metrics:
        - Targeted Attack Success Rate: 76.14%
        - Untargeted Attack Success Rate: 75.45%

3. Building the Defense: Adversarial Training with TRADES ([3.Adversarial_Training.py](./3.Adversarial_Training.py))
    - Description: A new ResNet-18 model was trained from scratch using the TRADES methodology. The model learns by simultaneously classifying clean images and defending against PGD attacks generated during the training loop. This is the core of our defense strategy.

4. The Robust ResNet-18: Performance Under Fire ([4.Adv_Model_Experiments.ipynb](./4.Adv_Model_Experiments.ipynb))
    - Description: The adversarially trained ResNet-18 was evaluated against the same PGD attacks.

    - Results: The robust model demonstrates a significant improvement in its ability to withstand adversarial attacks, showcasing the effectiveness of the TRADES framework. This comes at the cost of a slight reduction in clean accuracy, illustrating the classic trade-off between standard performance and security.

    - Metrics (Untargeted Attack):
        - Clean Accuracy: 45.36%
        - Attack Success Rate: 55.44%

    - Metrics (Targeted Attack):
        - Clean Accuracy: 45.36%
        - Attack Success Rate: 21.92%

Conclusion: The targeted attack success rate plummeted from 76.14% to a mere 21.92%, a 71% relative improvement in robustness, proving the success of our defense.

5. Interactive Demonstration ([5.Demo_App.py](./5.Demo_App.py))
    - Description: A user-friendly Streamlit application was developed to provide a hands-on experience with the trained models. It allows anyone to see the effects of adversarial attacks and the benefits of robust training.