import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.datasets import CIFAR100
import numpy as np
from tqdm import tqdm
import os
import sys
import logging
from datetime import datetime

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Generate timestamp for unique log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"logs/training_{timestamp}.log"

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)  # Explicitly use stdout
    ]
)

# Force logs to be written immediately
for handler in logging.root.handlers:
    if isinstance(handler, logging.FileHandler):
        handler.flush()

# Configure Lightning loggers
lightning_logger = logging.getLogger("lightning.pytorch")
lightning_logger.setLevel(logging.INFO)

# Add an explicit test log entry
logging.info(f"Logging initialized. Log file: {os.path.abspath(log_file)}")

ddp_strategy = DDPStrategy(process_group_backend="nccl", find_unused_parameters=False)

class CIFAR100DataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./data', batch_size=128, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Data transforms
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    
    def prepare_data(self):
        # Download data if needed
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)
    
    def setup(self, stage=None):
        # Load datasets
        cifar_full = CIFAR100(self.data_dir, train=True, transform=self.transform_train)
        
        # Split into train and validation
        train_size = int(0.9 * len(cifar_full))
        val_size = len(cifar_full) - train_size
        self.train_dataset, self.val_dataset = random_split(
            cifar_full, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create a separate validation dataset with test transforms
        cifar_val = CIFAR100(self.data_dir, train=True, transform=self.transform_test)
        _, val_indices = random_split(
            range(len(cifar_full)), [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        self.val_dataset = torch.utils.data.Subset(cifar_val, val_indices.indices)
        
        self.test_dataset = CIFAR100(self.data_dir, train=False, transform=self.transform_test)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True
        )

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.shortcut(x)
        return out

class WideResNet(nn.Module):
    def __init__(self, depth, widen_factor, dropout_rate, num_classes):
        super(WideResNet, self).__init__()
        self.in_planes = 16
        
        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor
        
        nStages = [16, 16*k, 32*k, 64*k]
        
        self.conv1 = nn.Conv2d(3, nStages[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._wide_layer(BasicBlock, nStages[1], n, stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._wide_layer(BasicBlock, nStages[2], n, stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._wide_layer(BasicBlock, nStages[3], n, stride=2, dropout_rate=dropout_rate)
        self.bn1 = nn.BatchNorm2d(nStages[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nStages[3], num_classes)
        
        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
    
    def _wide_layer(self, block, planes, num_blocks, stride, dropout_rate):
        strides = [stride] + [1]*(int(num_blocks)-1)
        layers = []
        
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_rate))
            self.in_planes = planes
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class AdversarialTrainingModule(pl.LightningModule):
    def __init__(self, 
                 pretrained_path=None,
                 depth=28,             # WideResNet depth parameter
                 widen_factor=10,      # WideResNet width parameter
                 dropout_rate=0.3,     # WideResNet dropout rate
                 epsilon=8/255, 
                 alpha=2/255, 
                 attack_iters=7, 
                 trades_beta=8.0,
                 learning_rate=0.05,
                 momentum=0.9,
                 accumulate_grad_batches=1,
                 weight_decay=5e-4,
                 lr_scheduler="cosine",
                 max_epochs=200):
        super().__init__()
        self.save_hyperparameters()
        
        # Create WideResNet model
        self.model = WideResNet(
            depth=depth, 
            widen_factor=widen_factor, 
            dropout_rate=dropout_rate, 
            num_classes=100
        )

        self.accumulate_grad_batches = accumulate_grad_batches
        self.automatic_optimization = False
        
        # Load pretrained model weights if provided
        if pretrained_path:
            checkpoint = torch.load(pretrained_path)
            # Handle potential key mismatches due to different architectures
            try:
                self.model.load_state_dict(checkpoint['state_dict'])
                print(f"Successfully loaded pretrained model from {pretrained_path}")
            except:
                print(f"Could not directly load weights from {pretrained_path} (different architecture)")
        
        # Save clean and robust accuracy as attributes
        self.best_clean_acc = 0.0
        self.best_robust_acc = 0.0
        
        # Class weights for balanced loss
        self.class_weights = None
        self.weight_update_counter = 0
    
    def forward(self, x):
        return self.model(x)
    
    def trades_loss(self, x_natural, y, perturb_steps=10):
        """TRADES loss for adversarial robustness"""
        # Loss functions
        criterion_kl = nn.KLDivLoss(reduction='batchmean')
        
        # Generate adversarial examples
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape, device=self.device)
        
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = criterion_kl(
                    F.log_softmax(self.model(x_adv), dim=1),
                    F.softmax(self.model(x_natural), dim=1)
                )
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + self.hparams.alpha * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - self.hparams.epsilon), 
                             x_natural + self.hparams.epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # Calculate the TRADES loss
        logits_natural = self.model(x_natural)
        logits_adv = self.model(x_adv)
        
        if self.class_weights is not None and self.class_weights.device != self.device:
            self.class_weights = self.class_weights.to(self.device)
        
        # Natural loss (possibly with class weights)
        if self.class_weights is not None:
            loss_natural = F.cross_entropy(logits_natural, y, weight=self.class_weights)
        else:
            loss_natural = F.cross_entropy(logits_natural, y)
        
        # KL divergence
        loss_robust = criterion_kl(
            F.log_softmax(logits_adv, dim=1),
            F.softmax(logits_natural, dim=1)
        )
        
        # Combined loss
        loss = loss_natural + self.hparams.trades_beta * loss_robust
        
        return loss, logits_natural, logits_adv, x_adv
    
    def compute_class_weights(self, dataloader):
        """Compute class weights based on model confusion"""
        confusion = torch.zeros(100, 100, device=self.device)
        self.eval()
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                _, preds = outputs.max(1)
                for t, p in zip(labels.view(-1), preds.view(-1)):
                    confusion[t.long(), p.long()] += 1
        
        # Normalize and compute weights
        confusion = confusion / (confusion.sum(dim=1, keepdim=True) + 1e-8)
        diag_indices = torch.arange(100, device=self.device)
        confusion[diag_indices, diag_indices] = 0
        class_weights = confusion.sum(dim=1)
        
        # Normalize weights
        class_weights = class_weights / class_weights.mean()
        class_weights = 0.5 + class_weights / 2
        
        return class_weights
    
    def training_step(self, batch, batch_idx):
        # Get optimizer
        opt = self.optimizers()
        
        # Get inputs and labels
        inputs, labels = batch
        
        # Use TRADES loss
        loss, logits_natural, logits_adv, adv_images = self.trades_loss(
            inputs, labels, perturb_steps=self.hparams.attack_iters
        )
        
        # Calculate metrics for logging
        _, nat_preds = logits_natural.max(1)
        nat_acc = nat_preds.eq(labels).float().mean()
        
        _, adv_preds = logits_adv.max(1)
        adv_acc = adv_preds.eq(labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_nat_acc', nat_acc * 100.0, prog_bar=True, sync_dist=True)
        self.log('train_adv_acc', adv_acc * 100.0, prog_bar=True, sync_dist=True)
        
        # Manual backward for gradient accumulation
        self.manual_backward(loss / self.accumulate_grad_batches)
        
        # Update weights after accumulate_grad_batches steps
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = F.cross_entropy(outputs, labels)
        
        _, preds = outputs.max(1)
        acc = preds.eq(labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        self.log('val_acc', acc * 100.0, prog_bar=True, sync_dist=True)
        
        # Add robustness evaluation to validation
        # This is a simplified version - you can make it more efficient
        if batch_idx == 0:  # Only test robustness on the first batch to save time
            clean_acc, robust_acc = self.evaluate_batch_robustness(inputs, labels)
            self.log('val_clean_acc', clean_acc * 100.0, sync_dist=True)
            self.log('val_robust_acc', robust_acc * 100.0, sync_dist=True)
        
        return {'val_loss': loss, 'val_acc': acc}
    
    def evaluate_batch_robustness(self, inputs, labels):
        """Evaluate robustness on a single batch"""
        # Clean accuracy
        with torch.no_grad():
            outputs = self.model(inputs)
            _, preds = outputs.max(1)
            clean_acc = preds.eq(labels).float().mean()
        
        # Create adversarial examples with PGD
        x_adv = inputs.clone() + 0.001 * torch.randn(inputs.shape, device=self.device)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        for _ in range(10):  # Fewer iterations during validation for speed
            x_adv.requires_grad_()
            with torch.enable_grad():
                outputs_adv = self.model(x_adv)
                loss = F.cross_entropy(outputs_adv, labels)
            
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + self.hparams.alpha * torch.sign(grad.detach())
            delta = torch.clamp(x_adv - inputs, -self.hparams.epsilon, self.hparams.epsilon)
            x_adv = torch.clamp(inputs + delta, 0.0, 1.0)
        
        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs = self.model(x_adv)
            _, preds = outputs.max(1)
            robust_acc = preds.eq(labels).float().mean()
        
        return clean_acc, robust_acc
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        
        # Clean accuracy
        with torch.no_grad():
            outputs = self.model(inputs)
            _, preds = outputs.max(1)
            clean_acc = preds.eq(labels).float().mean()
        
        # Generate adversarial examples with PGD
        x_adv = inputs.clone() + 0.001 * torch.randn(inputs.shape, device=self.device)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        for _ in range(20):  # More iterations for test-time evaluation
            x_adv.requires_grad_()
            with torch.enable_grad():
                outputs_adv = self.model(x_adv)
                loss = F.cross_entropy(outputs_adv, labels)
            
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + self.hparams.alpha * torch.sign(grad.detach())
            delta = torch.clamp(x_adv - inputs, -self.hparams.epsilon, self.hparams.epsilon)
            x_adv = torch.clamp(inputs + delta, 0.0, 1.0)
        
        # Evaluate on adversarial examples
        with torch.no_grad():
            outputs = self.model(x_adv)
            _, preds = outputs.max(1)
            robust_acc = preds.eq(labels).float().mean()
        
        # Log metrics
        self.log('test_clean_acc', clean_acc * 100.0, prog_bar=True, sync_dist=True)
        self.log('test_robust_acc', robust_acc * 100.0, prog_bar=True, sync_dist=True)
        
        return {'test_clean_acc': clean_acc, 'test_robust_acc': robust_acc}
    
    def configure_optimizers(self):
        # WideResNet typically benefits from a different LR schedule
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay,
            nesterov=True  # Add Nesterov momentum for better performance
        )
        
        # Configure learning rate scheduler
        if self.hparams.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.hparams.max_epochs
            )
        else:
            # This schedule is often used in WideResNet papers
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=[60, 120, 160],
                gamma=0.2
            )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

class ConsistentRobustnessEvaluator(pl.Callback):
    """Evaluates robustness on a fixed set of validation examples"""
    
    def __init__(self, num_samples=100, eval_every_n_epochs=3):
        super().__init__()
        self.num_samples = num_samples
        self.eval_every_n_epochs = eval_every_n_epochs
        self.fixed_samples = None
        self.fixed_labels = None
    
    def on_validation_epoch_start(self, trainer, pl_module):
        """Setup fixed evaluation set the first time and run evaluations periodically"""
        current_epoch = trainer.current_epoch
        
        # Skip during sanity checks
        if trainer.sanity_checking:
            return
            
        # Skip if not time to evaluate
        if current_epoch % self.eval_every_n_epochs != 0 and current_epoch != trainer.max_epochs - 1:
            return
        
        # Get validation dataloader
        val_dataloader = trainer.val_dataloaders
        if isinstance(val_dataloader, list):
            val_dataloader = val_dataloader[0]
        
        # Create fixed set of validation samples if not already done
        if self.fixed_samples is None:
            self._create_fixed_sample_set(val_dataloader, pl_module.device)
        
        # Evaluate robustness on fixed samples
        clean_acc, robust_acc = self._evaluate_robustness(pl_module)
        
        # Log metrics
        pl_module.log('val_clean_acc', clean_acc * 100.0, sync_dist=True)
        pl_module.log('val_robust_acc', robust_acc * 100.0, sync_dist=True)
        
        # Print for feedback
        print(f"\nEpoch {current_epoch}: Val Clean Acc: {clean_acc*100:.2f}%, Val Robust Acc: {robust_acc*100:.2f}%")

    
    def _create_fixed_sample_set(self, dataloader, device):
        """Create a fixed set of samples from validation set"""
        samples = []
        labels = []
        count = 0
        
        # Extract samples
        for inputs, targets in dataloader:
            batch_size = inputs.size(0)
            remaining = self.num_samples - count
            if remaining <= 0:
                break
                
            # Take only what we need
            take_size = min(batch_size, remaining)
            samples.append(inputs[:take_size].to(device))
            labels.append(targets[:take_size].to(device))
            
            count += take_size
            
        # Combine into tensors
        self.fixed_samples = torch.cat(samples, dim=0)
        self.fixed_labels = torch.cat(labels, dim=0)
        
        print(f"Created fixed evaluation set with {self.fixed_samples.size(0)} samples")
    
    def _evaluate_robustness(self, pl_module):
        """Evaluate clean and robust accuracy on fixed samples"""
        device = pl_module.device
        
        # Store original model state
        was_training = pl_module.training
        
        # Set to eval mode for clean accuracy
        pl_module.eval()
        
        # Parameters for PGD attack
        epsilon = pl_module.hparams.epsilon
        alpha = pl_module.hparams.alpha
        attack_iters = 20  # More iterations for better evaluation
        
        with torch.no_grad():
            # Clean accuracy
            outputs = pl_module(self.fixed_samples)
            _, preds = outputs.max(1)
            clean_acc = preds.eq(self.fixed_labels).float().mean().item()
        
        # Important: we need to create a fresh copy that requires gradients
        x_adv = self.fixed_samples.clone().detach()
        # Add small random noise for initial perturbation
        if pl_module.hparams.get('random_start', True):
            x_adv = x_adv + 0.001 * torch.randn_like(x_adv)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        # For robust accuracy, use a simplified PGD attack that doesn't rely on model gradients
        for _ in range(attack_iters):
            # Ensure we track gradients
            x_adv.requires_grad_(True)
            
            if x_adv.grad is not None:
                x_adv.grad.zero_()
                
            # We need to compute a new forward pass without no_grad context
            with torch.enable_grad():
                outputs = pl_module(x_adv)
                loss = F.cross_entropy(outputs, self.fixed_labels)
                
            # Compute gradient
            loss.backward()
            
            # Make sure we have a gradient
            if x_adv.grad is None:
                print("Warning: gradient is None, cannot create adversarial examples")
                robust_acc = float('nan')
                break
                
            # Update adversarial example
            grad = x_adv.grad.detach()
            x_adv = x_adv.detach() + alpha * grad.sign()
            delta = torch.clamp(x_adv - self.fixed_samples, min=-epsilon, max=epsilon)
            x_adv = torch.clamp(self.fixed_samples + delta, min=0.0, max=1.0)
        
        # Ensure model is back in eval mode for evaluation
        pl_module.eval()
        
        # Calculate accuracy on adversarial examples
        with torch.no_grad():
            outputs = pl_module(x_adv.detach())
            _, preds = outputs.max(1)
            robust_acc = preds.eq(self.fixed_labels).float().mean().item()
        
        # Restore original model state
        if was_training:
            pl_module.train()
            
        return clean_acc, robust_acc

class RobustnessEvaluationCallback(pl.Callback):
    def __init__(self, eval_every_n_epochs=3):
        super().__init__()
        self.eval_every_n_epochs = eval_every_n_epochs
        self.best_robust_acc = 0.0
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Evaluate robustness every N epochs
        current_epoch = trainer.current_epoch
        if current_epoch % self.eval_every_n_epochs == 0 or current_epoch == trainer.max_epochs - 1:
            # Get a batch from test set
            test_dataloader = trainer.datamodule.test_dataloader()
            batch = next(iter(test_dataloader))
            
            # Move batch to device
            inputs, labels = [x.to(pl_module.device) for x in batch]
            
            # Evaluate robustness
            clean_acc, robust_acc = pl_module.evaluate_batch_robustness(inputs, labels)
            
            # Log metrics
            pl_module.log('val_robust_acc', robust_acc * 100.0)
            pl_module.log('val_clean_acc', clean_acc * 100.0)
            
            # Track best and save model if improved
            if robust_acc > self.best_robust_acc:
                self.best_robust_acc = robust_acc
                # Log for user visibility  
                print(f"\nNew best robust accuracy: {robust_acc*100.0:.2f}%")

class ProgressiveAdversarialCallback(pl.Callback):
    def __init__(self, start_epoch=20, ramp_epochs=10, final_beta=6.0):
        super().__init__()
        self.start_epoch = start_epoch
        self.ramp_epochs = ramp_epochs
        self.final_beta = final_beta
        
    def on_train_epoch_start(self, trainer, pl_module):
        epoch = trainer.current_epoch
        
        # First start_epoch epochs: clean training (beta=0)
        if epoch < self.start_epoch:
            beta = 0.0
            logging.info(f"Epoch {epoch+1}: Clean training phase")
        
        # Ramp up beta gradually
        elif epoch < self.start_epoch + self.ramp_epochs:
            # Linear ramp from 0 to final_beta
            progress = (epoch - self.start_epoch) / self.ramp_epochs
            beta = progress * self.final_beta
            logging.info(f"Epoch {epoch+1}: Transition phase, beta={beta:.2f}")
        
        # Full adversarial training
        else:
            beta = self.final_beta
            logging.info(f"Epoch {epoch+1}: Full adversarial training, beta={beta:.2f}")
        
        # Update the beta parameter
        pl_module.hparams.trades_beta = beta

class BestAccuracyTracker(pl.Callback):
    """Callback to track best clean and robust accuracies with proper logging"""
    
    def __init__(self):
        super().__init__()
        self.best_clean_acc = 0.0
        self.best_robust_acc = 0.0
        self.logger = logging.getLogger(__name__)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Track best validation accuracy"""
        if 'val_acc' in trainer.callback_metrics:
            val_acc = trainer.callback_metrics['val_acc']
            if val_acc > self.best_clean_acc:
                self.best_clean_acc = val_acc.item()
                pl_module.best_clean_acc = self.best_clean_acc
                trainer.logger.log_metrics({'best_clean_acc': self.best_clean_acc}, step=trainer.global_step)
                self.logger.info(f"New best clean accuracy: {self.best_clean_acc:.2f}%")
        
        if 'val_robust_acc' in trainer.callback_metrics:
            val_robust_acc = trainer.callback_metrics['val_robust_acc']
            if val_robust_acc > self.best_robust_acc:
                self.best_robust_acc = val_robust_acc.item()
                pl_module.best_robust_acc = self.best_robust_acc
                trainer.logger.log_metrics({'best_robust_acc': self.best_robust_acc}, step=trainer.global_step)
                self.logger.info(f"New best robust accuracy: {self.best_robust_acc:.2f}%")
    
    def on_fit_end(self, trainer, pl_module):
        """Make sure final values are set in the module"""
        pl_module.best_clean_acc = self.best_clean_acc
        pl_module.best_robust_acc = self.best_robust_acc
        
        # Log the final values (will go to your training.log file)
        self.logger.info(f"Final best clean accuracy: {self.best_clean_acc:.2f}%")
        self.logger.info(f"Final best robust accuracy: {self.best_robust_acc:.2f}%")
        
        # Also print for immediate console feedback
        print(f"Final best clean accuracy: {self.best_clean_acc:.2f}%")
        print(f"Final best robust accuracy: {self.best_robust_acc:.2f}%")

def train_adversarial_model(
    pretrained_path=None,  # Set to None if starting from scratch
    batch_size=64,
    accumulate_grad_batches=2,
    max_epochs=200,
    depth=28,              # Common WideResNet depths: 16, 28, 40
    widen_factor=10,       # Common width factors: 1, 2, 4, 8, 10
    dropout_rate=0.3,      # Recommended for CIFAR-100
    learning_rate=0.1,     # WideResNet typically uses higher LR
    epsilon=8/255,
    alpha=2/255,
    attack_iters=7,
    trades_beta=8.0,
    num_workers=12
):
    # Set up data module
    data_module = CIFAR100DataModule(
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Initialize WideResNet model
    model = AdversarialTrainingModule(
        pretrained_path=pretrained_path,
        depth=depth,
        widen_factor=widen_factor,
        dropout_rate=dropout_rate,
        epsilon=epsilon,
        alpha=alpha,
        attack_iters=attack_iters,
        trades_beta=trades_beta,
        learning_rate=learning_rate,
        max_epochs=max_epochs,
        accumulate_grad_batches=accumulate_grad_batches
    )
    
    # Define callbacks - same as before
    checkpoint_callback_clean = ModelCheckpoint(
        monitor='val_acc',
        filename=f'wrn-{depth}-{widen_factor}-best-clean',
        save_top_k=1,
        mode='max',
        save_last=True
    )

    checkpoint_callback_robust = ModelCheckpoint(
        monitor='val_robust_acc',
        filename=f'wrn-{depth}-{widen_factor}-best-robust',
        save_top_k=1,
        mode='max'
    )

    accuracy_tracker = BestAccuracyTracker()
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    # logging_callback = LoggingCallback(log_every_n_batches=10)
    robustness_eval = RobustnessEvaluationCallback(eval_every_n_epochs=3)

    progressive_callback = ProgressiveAdversarialCallback(start_epoch=20, ramp_epochs=10, final_beta=8.0)

    consistent_evaluator = ConsistentRobustnessEvaluator(
        num_samples=100,  # Use 100 fixed samples
        eval_every_n_epochs=3  # Evaluate every 3 epochs
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,  # Adjust based on available GPUs
        strategy=ddp_strategy,
        max_epochs=max_epochs,
        callbacks=[
            checkpoint_callback_clean, 
            checkpoint_callback_robust,
            robustness_eval,
            lr_monitor,
            accuracy_tracker,
            progressive_callback,
            consistent_evaluator
        ],
        precision="16-mixed",
        log_every_n_steps=50,
        check_val_every_n_epoch=1
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Evaluate on test set
    trainer.test(model, data_module)
    
    return model, trainer, accuracy_tracker.best_clean_acc, accuracy_tracker.best_robust_acc

if __name__ == "__main__":
    pl.seed_everything(42)
    
    # Use the integrated pretraining approach
    model, trainer, best_clean, best_robust = train_adversarial_model(
        pretrained_path=None,  # Start from scratch with progressive training
        depth=28,
        widen_factor=10,
        dropout_rate=0.3,
        batch_size=128,
        max_epochs=200,
        learning_rate=0.1,
        epsilon=8/255,
        alpha=2/255,
        attack_iters=7,
        trades_beta=8.0  # This will be managed by the progressive callback
    )
    
    print("Training completed!")
    print(f"Best clean accuracy: {best_clean:.2f}%")
    print(f"Best robust accuracy: {best_robust:.2f}%")