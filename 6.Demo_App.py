import streamlit as st
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from adv_train import AdversarialTrainingModule

# Load all models
@st.cache_resource
def load_models():
    # Load your adversarially trained models
    robust_model = AdversarialTrainingModule.load_from_checkpoint("lightning_logs/version_2/checkpoints/best-robust.ckpt")
    clean_model = AdversarialTrainingModule.load_from_checkpoint("lightning_logs/version_2/checkpoints/best-clean.ckpt")
    
    # Load the original vulnerable model
    original_model = torchvision.models.resnet18(weights=None)
    original_model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    original_model.fc = torch.nn.Linear(original_model.fc.in_features, 100)  # 100 classes for CIFAR-100
    original_checkpoint = torch.load('best_resnet18_cifar100.pth')  # Your original checkpoint
    original_model.load_state_dict(original_checkpoint['state_dict'])
    
    # Set all models to evaluation mode
    robust_model.eval()
    clean_model.eval()
    original_model.eval()
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    robust_model = robust_model.to(device)
    clean_model = clean_model.to(device)
    original_model = original_model.to(device)
    
    return original_model, clean_model, robust_model

# Load CIFAR-100 test set with class names
@st.cache_resource
def load_cifar100():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])
    test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    class_names = test_dataset.classes
    return test_dataset, class_names

# PGD attack function
def pgd_attack(model, image, label, epsilon, alpha, iters, random_start=True):
    device = next(model.parameters()).device
    image = image.clone().detach().to(device)
    label = torch.tensor([label]).to(device)
    
    # Loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Initialize adversarial image
    adv_image = image.clone().detach()
    
    # Add random noise if requested
    if random_start:
        noise = torch.FloatTensor(image.shape).uniform_(-epsilon, epsilon).to(device)
        adv_image = adv_image + noise
        adv_image = torch.clamp(adv_image, 0, 1)
    
    # PGD attack
    for i in range(iters):
        adv_image.requires_grad = True
        output = model(adv_image)
        loss = -criterion(output, label)  # Maximize loss
        
        # Backward pass
        loss.backward()
        
        # Get gradient sign
        grad = adv_image.grad.sign()
        adv_image = adv_image.detach() - alpha * grad
        
        # Project back to epsilon ball and valid image range
        delta = torch.clamp(adv_image - image, min=-epsilon, max=epsilon)
        adv_image = torch.clamp(image + delta, min=0, max=1)
        
    return adv_image

# Function to evaluate models and get predictions
def get_predictions(models, image):
    results = []
    for model_name, model in models.items():
        with torch.no_grad():
            output = model(image)
            pred_class = output.argmax(1).item()
            confidence = torch.softmax(output, dim=1)[0][pred_class].item() * 100
            results.append({
                'model': model_name,
                'prediction': pred_class,
                'confidence': confidence
            })
    return results

# App UI
st.title("Adversarial Attack Demo: CIFAR-100")
st.write("Compare how different models respond to adversarial attacks")

# Load resources
original_model, clean_model, robust_model = load_models()
test_dataset, class_names = load_cifar100()

# Create model dictionary
models = {
    "Original (Vulnerable)": original_model,
    "Best Clean": clean_model,
    "Best Robust": robust_model
}

# Sidebar options
st.sidebar.header("Attack Configuration")
epsilon = st.sidebar.slider("Perturbation Size (ε)", 0.0, 32.0, 8.0) / 255.0
alpha = st.sidebar.slider("Step Size (α)", 0.0, 16.0, 2.0) / 255.0
iters = st.sidebar.slider("Attack Iterations", 1, 100, 20)
attack_target = st.sidebar.selectbox("Generate attack using which model?", list(models.keys()))

# Image selection
st.header("Select a Test Image")
cols = st.columns(2)
with cols[0]:
    # Option 1: Random sampling
    if st.button("Pick Random Image"):
        random_idx = np.random.randint(0, len(test_dataset))
        st.session_state.img_idx = random_idx
        
with cols[1]:
    # Option 2: User input index
    idx_input = st.number_input("Or enter image index:", 0, len(test_dataset)-1, 
                                value=st.session_state.get('img_idx', 0),
                                on_change=lambda: setattr(st.session_state, 'img_idx', int(idx_input)))

# Display selected image
img_idx = st.session_state.get('img_idx', 0)
image, true_label = test_dataset[img_idx]
image = image.unsqueeze(0)  # Add batch dimension
true_class_name = class_names[true_label]

# Process the image: original prediction for all models
device = next(original_model.parameters()).device
image = image.to(device)
orig_results = get_predictions(models, image)

# Show original predictions
st.header("Original Image")
st.image(np.transpose(image[0].cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5, clamp=True, width=300)
st.write(f"True class: {true_class_name}")

# Create a table of original predictions
orig_data = [
    [r['model'], class_names[r['prediction']], f"{r['confidence']:.1f}%", 
     "✓" if r['prediction'] == true_label else "✗"]
    for r in orig_results
]
st.table({
    "Model": [r[0] for r in orig_data],
    "Prediction": [r[1] for r in orig_data],
    "Confidence": [r[2] for r in orig_data],
    "Correct?": [r[3] for r in orig_data]
})

# Run attack if requested
if st.button("Generate Adversarial Example"):
    # Use the selected model to generate the attack
    attack_model = models[attack_target]
    adv_image = pgd_attack(attack_model, image, true_label, epsilon, alpha, iters)
    
    # Get predictions on adversarial image for all models
    adv_results = get_predictions(models, adv_image)
    
    # Calculate perturbation
    perturbation = adv_image - image
    
    # Display results
    st.header("Attack Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Adversarial Image")
        st.image(np.transpose(adv_image[0].cpu().numpy(), (1, 2, 0)) * 0.5 + 0.5, clamp=True, width=300)
        st.write(f"Attack created using: {attack_target} model")
    
    with col2:
        st.subheader("Perturbation (Magnified for Visibility)")
        # Scale perturbation for visibility
        pert_display = np.transpose(perturbation[0].cpu().numpy(), (1, 2, 0))
        # Normalize to [0,1] range for display
        pert_display = (pert_display - pert_display.min()) / (pert_display.max() - pert_display.min() + 1e-8)
        st.image(pert_display, clamp=True, width=300)
        st.write(f"Epsilon: {epsilon:.4f}, Attack iterations: {iters}")
    
    # Create a table comparing original vs adversarial predictions
    st.header("Model Performance Comparison")
    
    comparison_data = []
    for orig, adv in zip(orig_results, adv_results):
        model_name = orig['model']
        orig_pred = class_names[orig['prediction']]
        adv_pred = class_names[adv['prediction']]
        attack_success = "✓" if orig['prediction'] != adv['prediction'] else "✗"
        
        comparison_data.append([
            model_name, 
            orig_pred, 
            f"{orig['confidence']:.1f}%",
            adv_pred, 
            f"{adv['confidence']:.1f}%",
            attack_success
        ])
    
    st.table({
        "Model": [r[0] for r in comparison_data],
        "Original Prediction": [r[1] for r in comparison_data],
        "Original Confidence": [r[2] for r in comparison_data],
        "Adversarial Prediction": [r[3] for r in comparison_data],
        "Adversarial Confidence": [r[4] for r in comparison_data],
        "Attack Succeeded?": [r[5] for r in comparison_data]
    })
    
    # Display summary statistics
    st.subheader("Attack Success Summary")
    success_count = sum(1 for r in comparison_data if r[5] == "✓")
    
    st.write(f"Attack success rate: {success_count}/{len(models)} models fooled")
    
    # Highlight which models were robust
    robust_models = [r[0] for r in comparison_data if r[5] == "✗"]
    if robust_models:
        st.success(f"Models that resisted the attack: {', '.join(robust_models)}")
    else:
        st.error("All models were fooled by this attack!")
