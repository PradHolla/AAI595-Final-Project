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

# Important: Add this custom callback to your trainer
class LoggingCallback(pl.Callback):
    def __init__(self, log_every_n_batches=50):
        super().__init__()
        self.log_every_n_batches = log_every_n_batches
        
    def on_train_start(self, trainer, pl_module):
        logging.info("=== Training started ===")
        logging.info(f"Total epochs: {trainer.max_epochs}")
        logging.info(f"Number of GPUs: {trainer.num_devices}")
        logging.info(f"Batch size: {trainer.datamodule.batch_size}")
        
    def on_train_epoch_start(self, trainer, pl_module):
        logging.info(f"Starting epoch {trainer.current_epoch+1}/{trainer.max_epochs}")
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % self.log_every_n_batches == 0:
            # Force logs to disk periodically
            for handler in logging.root.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.flush()
                    
    def on_validation_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        logging.info(f"Validation results: {metrics}")
        
    def on_train_end(self, trainer, pl_module):
        logging.info("=== Training completed ===")
        logging.info(f"Best validation accuracy: {pl_module.best_clean_acc:.2f}%")

ddp_strategy = DDPStrategy(process_group_backend="gloo", find_unused_parameters=True)

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

class AdversarialTrainingModule(pl.LightningModule):
    def __init__(self, 
                 pretrained_path=None,
                 epsilon=8/255, 
                 alpha=2/255, 
                 attack_iters=7, 
                 trades_beta=8.0,
                 learning_rate=0.05,
                 momentum=0.9,
                 weight_decay=5e-4,
                 lr_scheduler="cosine",
                 max_epochs=200):
        super().__init__()
        self.save_hyperparameters()
        
        # Model definition
        self.model = models.resnet18(weights=None)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 100)
        
        # Load pretrained model if provided
        if pretrained_path:
            checkpoint = torch.load(pretrained_path)
            self.model.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded pretrained model from {pretrained_path}")
        
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
        inputs, labels = batch
        
        # Update class weights occasionally
        if self.trainer.current_epoch > 0 and self.trainer.current_epoch % 15 == 0 and self.weight_update_counter != self.trainer.current_epoch:
            self.weight_update_counter = self.trainer.current_epoch
            
            # Fix for DataLoader not being subscriptable
            val_dataloader = self.trainer.val_dataloaders
            if isinstance(val_dataloader, list):
                val_dataloader = val_dataloader[0]
            
            self.class_weights = self.compute_class_weights(val_dataloader)
            self.log('class_weights_min', self.class_weights.min())
            self.log('class_weights_max', self.class_weights.max())
        
        # Use TRADES loss
        loss, logits_natural, logits_adv, adv_images = self.trades_loss(
            inputs, labels, perturb_steps=self.hparams.attack_iters
        )
        
        # Calculate metrics
        _, nat_preds = logits_natural.max(1)
        nat_acc = nat_preds.eq(labels).float().mean()
        
        _, adv_preds = logits_adv.max(1)
        adv_acc = adv_preds.eq(labels).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        self.log('train_nat_acc', nat_acc * 100.0, prog_bar=True, sync_dist=True)
        self.log('train_adv_acc', adv_acc * 100.0, prog_bar=True, sync_dist=True)
        
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
    
    def on_test_epoch_end(self):
        # This is called automatically at the end of the test epoch
        pass
    
    def configure_optimizers(self):
        # Initialize optimizer with reduced learning rate for pretrained model
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        
        # Configure learning rate scheduler
        if self.hparams.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.hparams.max_epochs
            )
        else:
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


def train_adversarial_model(
    pretrained_path='best_resnet18_cifar100_untargeted_adv.pth',
    batch_size=128,
    max_epochs=200,
    learning_rate=0.05,
    epsilon=8/255,
    alpha=2/255,
    attack_iters=7,
    trades_beta=8.0,
    num_workers=4
):
    # Set up data module
    data_module = CIFAR100DataModule(
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    # Initialize model
    model = AdversarialTrainingModule(
        pretrained_path=pretrained_path,
        epsilon=epsilon,
        alpha=alpha,
        attack_iters=attack_iters,
        trades_beta=trades_beta,
        learning_rate=learning_rate,
        max_epochs=max_epochs
    )
    
    # Define callbacks
    checkpoint_callback_clean = ModelCheckpoint(
        monitor='val_acc',
        filename='best-clean',
        save_top_k=1,
        mode='max',
        save_last=True
    )

    checkpoint_callback_robust = ModelCheckpoint(
        monitor='val_robust_acc',
        filename='best-robust',
        save_top_k=1,
        mode='max'
    )

    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Set up logger
    logging_callback = LoggingCallback(log_every_n_batches=10)
    
    robustness_eval = RobustnessEvaluationCallback(eval_every_n_epochs=3)

    trainer = pl.Trainer(
        accelerator='gpu',
        devices=4,
        strategy=ddp_strategy,
        max_epochs=max_epochs,
        callbacks=[
            checkpoint_callback_clean, 
            checkpoint_callback_robust,
            robustness_eval,  # Add the custom callback
            lr_monitor,
            logging_callback
        ],
        precision=16,
        log_every_n_steps=50,
        check_val_every_n_epoch=1
    )
    
    # Train the model
    trainer.fit(model, data_module)
    
    # Evaluate on test set
    trainer.test(model, data_module)
    
    return model, trainer

if __name__ == "__main__":
    # Set random seeds for reproducibility
    pl.seed_everything(42)
    
    # # Make sure CUDA is available
    # print(f"CUDA available: {torch.cuda.is_available()}")
    # print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    # Start training
    model, trainer = train_adversarial_model(
        pretrained_path='best_resnet18_cifar100_untargeted_adv.pth',
        batch_size=128,
        max_epochs=200,
        learning_rate=0.05,  # Reduced from 0.1 since we're using a pretrained model
        epsilon=8/255,
        alpha=2/255,
        attack_iters=7,
        trades_beta=8.0
    )
    
    print("Training completed!")
    print(f"Best clean accuracy: {model.best_clean_acc:.2f}%")
    print(f"Best robust accuracy: {model.best_robust_acc:.2f}%")
