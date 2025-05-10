import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from read_data import ChestXrayDataSet
import torchvision
from transformers import get_linear_schedule_with_warmup


class ModifiedCheXNet(nn.Module):
    def __init__(self, num_classes=14, pretrained_path=None):
        super(ModifiedCheXNet, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)

        if pretrained_path and os.path.exists(pretrained_path):
            print("=> Loading pretrained CheXNet weights")
            checkpoint = torch.load(pretrained_path)
            state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]
                k = k.replace('conv.1', 'conv1').replace('conv.2', 'conv2') \
                    .replace('norm.1', 'norm1').replace('norm.2', 'norm2')
                new_state_dict[k] = v

            try:
                self.densenet121.load_state_dict(new_state_dict, strict=False)
                print("Successfully loaded pretrained weights")
            except RuntimeError as e:
                print(f"Error loading pretrained weights: {e}")

        # Freeze early layers
        frozen_layers = ['conv0', 'norm0', 'denseblock1', 'transition1']

        for name, param in self.densenet121.features.named_parameters():
            if any(layer in name for layer in frozen_layers):
                param.requires_grad = False

        # Modified classifier
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(512, num_classes)  # Remove sigmoid since using BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.densenet121(x)

    def print_trainable_parameters(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\nTrainable parameters: {trainable_params:,} ({trainable_params / total_params:.2%} of total)")

        print("\nTrainable layers:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(f"{name}: {param.numel():,} parameters")


def calculate_class_weights(dataset):
    all_labels = []
    for _, labels in dataset:
        all_labels.append(labels.numpy())

    labels_array = np.vstack(all_labels)
    pos_weights = []

    for i in range(labels_array.shape[1]):
        neg_count = len(labels_array) - np.sum(labels_array[:, i])
        pos_count = np.sum(labels_array[:, i])
        # Clamp weights to prevent extreme values
        weight = min(neg_count / pos_count if pos_count > 0 else 1.0, 10.0)
        pos_weights.append(weight)

    return torch.FloatTensor(pos_weights)


def get_weighted_sampler(dataset):
    all_labels = []
    for _, labels in dataset:
        all_labels.append(labels.numpy())

    labels_array = np.vstack(all_labels)
    class_weights = calculate_class_weights(dataset)

    sample_weights = np.zeros(len(dataset))
    for i, labels in enumerate(labels_array):
        positive_classes = np.where(labels == 1)[0]
        if len(positive_classes) > 0:
            sample_weights[i] = np.mean([class_weights[j] for j in positive_classes])
        else:
            sample_weights[i] = 1.0

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )


def train_model(model, train_loader, valid_loader, device, num_epochs=20):
    # Setup loss and optimizer
    pos_weights = calculate_class_weights(train_loader.dataset).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Count parameters for L1 normalization scaling
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Optimizer with different learning rates
    optimizer = optim.AdamW([
        {'params': model.densenet121.classifier.parameters(), 'lr': 1e-3},
        {'params': model.densenet121.features.parameters(), 'lr': 1e-4}
    ], weight_decay=0.001)

    # Learning rate scheduler with warmup
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_val_f1 = 0.0
    patience = 7
    patience_counter = 0
    accumulation_steps = 4  # Gradient accumulation steps

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_outputs = []

        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                    desc=f'Epoch {epoch + 1}/{num_epochs}')

        optimizer.zero_grad()  # Zero gradients at start of epoch

        for batch_idx, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add scaled L1 regularization
            l1_lambda = 1e-5
            l1_norm = sum(p.abs().sum() for p in model.parameters() if p.requires_grad)
            loss = loss + (l1_lambda * l1_norm / num_parameters)

            # Gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            running_loss += loss.item() * accumulation_steps
            all_train_labels.append(labels.cpu().numpy())
            all_train_outputs.append(torch.sigmoid(outputs).detach().cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

        # Calculate training metrics
        all_train_labels = np.vstack(all_train_labels)
        all_train_outputs = np.vstack(all_train_outputs)
        train_preds = (all_train_outputs > 0.5).astype(int)

        train_f1s = []
        train_aucs = []
        for i in range(all_train_outputs.shape[1]):
            if len(np.unique(all_train_labels[:, i])) > 1:
                train_f1s.append(f1_score(all_train_labels[:, i], train_preds[:, i]))
                train_aucs.append(roc_auc_score(all_train_labels[:, i], all_train_outputs[:, i]))

        train_f1 = np.mean(train_f1s)
        train_auc = np.mean(train_aucs)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_val_labels = []
        all_val_outputs = []

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                all_val_labels.append(labels.cpu().numpy())
                all_val_outputs.append(torch.sigmoid(outputs).cpu().numpy())

        all_val_labels = np.vstack(all_val_labels)
        all_val_outputs = np.vstack(all_val_outputs)
        val_preds = (all_val_outputs > 0.5).astype(int)

        val_f1s = []
        val_aucs = []
        for i in range(all_val_outputs.shape[1]):
            if len(np.unique(all_val_labels[:, i])) > 1:
                val_f1s.append(f1_score(all_val_labels[:, i], val_preds[:, i]))
                val_aucs.append(roc_auc_score(all_val_labels[:, i], all_val_outputs[:, i]))

        val_f1 = np.mean(val_f1s)
        val_auc = np.mean(val_aucs)

        # Print metrics
        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {running_loss / len(train_loader):.4f}, '
              f'Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss / len(valid_loader):.4f}, '
              f'Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}')

        # Early stopping and model saving
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_val_f1,
                'best_auc': val_auc,
            }, 'best_chexnet_finetuned_improved.pth')
            print(f"Saved new best model with F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data paths
    train_file = "split_data/train_16k.csv"
    valid_file = "split_data/valid_16k.csv"
    img_dir = "split_data/images"

    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomRotation(5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create datasets
    print("Loading datasets...")
    train_dataset = ChestXrayDataSet(img_dir, train_file, train_transform)
    valid_dataset = ChestXrayDataSet(img_dir, valid_file, val_transform)

    # Create data loaders
    print("Creating data loaders...")
    train_sampler = get_weighted_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize model
    print("Initializing model...")
    model = ModifiedCheXNet(
        num_classes=14,
        pretrained_path='chexnet/CheXNet/model.pth.tar'
    ).to(device)

    # Print model information
    model.print_trainable_parameters()

    # Train the model
    print("\nStarting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        num_epochs=20
    )


if __name__ == '__main__':
    main()