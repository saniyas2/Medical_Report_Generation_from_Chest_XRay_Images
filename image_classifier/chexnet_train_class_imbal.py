import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from read_data import ChestXrayDataSet
import torchvision
from sklearn.metrics import f1_score

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weights=None):
        super(WeightedBCELoss, self).__init__()
        self.pos_weights = pos_weights  # Shape should be [num_classes]

    def forward(self, outputs, targets):
        # Calculate weighted loss for each class
        loss = 0
        for i in range(outputs.size(1)):
            class_outputs = outputs[:, i]
            class_targets = targets[:, i]

            # Apply positive weights to positive samples
            if self.pos_weights is not None:
                weight = torch.ones_like(class_targets)
                weight[class_targets == 1] = self.pos_weights[i]
                class_loss = nn.BCELoss(weight=weight)(class_outputs, class_targets)
            else:
                class_loss = nn.BCELoss()(class_outputs, class_targets)

            loss += class_loss
        return loss / outputs.size(1)


class ModifiedCheXNet(nn.Module):
    def __init__(self, num_classes=14, pretrained_path=None):
        super(ModifiedCheXNet, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)

        if pretrained_path and os.path.exists(pretrained_path):
            print("=> Loading pretrained CheXNet weights")
            checkpoint = torch.load(pretrained_path)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # Remove 'module.' prefix and fix layer names
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    k = k[7:]  # Remove 'module.' prefix
                # Replace dots with underscores in layer names
                k = k.replace('conv.1', 'conv1')
                k = k.replace('conv.2', 'conv2')
                k = k.replace('norm.1', 'norm1')
                k = k.replace('norm.2', 'norm2')
                new_state_dict[k] = v

            # Load the processed state dict
            try:
                self.densenet121.load_state_dict(new_state_dict, strict=False)
                print("Successfully loaded pretrained weights")
            except RuntimeError as e:
                print(f"Error loading pretrained weights: {e}")

        # Freeze all layers except the last dense block and classifier
        frozen_layers = [
            'conv0', 'norm0', 'denseblock1', 'transition1'
        ]

        for name, param in self.densenet121.features.named_parameters():
            if any(layer in name for layer in frozen_layers):
                param.requires_grad = False

        # Modify the classifier
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.densenet121(x)

    def print_trainable_parameters(self):
        """Print which layers are trainable and which are frozen"""
        print("\nTrainable layers:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

        print("\nFrozen layers:")
        for name, param in self.named_parameters():
            if not param.requires_grad:
                print(name)


def calculate_class_weights(dataset):
    """Calculate weights for each class based on their frequency"""
    all_labels = []
    for _, labels in dataset:
        all_labels.append(labels.numpy())

    labels_array = np.vstack(all_labels)
    class_counts = np.sum(labels_array, axis=0)
    total_samples = len(dataset)

    # Calculate weights as inverse of frequency
    class_weights = total_samples / (len(class_counts) * class_counts)
    # Normalize weights
    class_weights = class_weights / np.sum(class_weights) * len(class_weights)

    return torch.FloatTensor(class_weights)


def get_weighted_sampler(dataset):
    """Create a weighted sampler to balance the dataset"""
    all_labels = []
    for _, labels in dataset:
        all_labels.append(labels.numpy())

    labels_array = np.vstack(all_labels)

    # Calculate sample weights based on class frequencies
    sample_weights = np.zeros(len(dataset))
    class_weights = calculate_class_weights(dataset)

    for i, labels in enumerate(labels_array):
        # For multi-label, take the average weight of all positive classes
        positive_classes = np.where(labels == 1)[0]
        if len(positive_classes) > 0:
            sample_weights[i] = np.mean([class_weights[j] for j in positive_classes])
        else:
            # Handle samples with no positive labels (No Finding)
            sample_weights[i] = 1.0

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True
    )


# def train_model(model, train_loader, valid_loader, device, num_epochs=10 ):
#     # Calculate positive weights for weighted BCE loss
#     pos_weights = calculate_class_weights(train_loader.dataset)
#     criterion = WeightedBCELoss(pos_weights=pos_weights.to(device))
#
#     # Different learning rates for different parts
#     classifier_params = list(model.densenet121.classifier.parameters())
#     feature_params = list(model.densenet121.features.parameters())
#
#     optimizer = optim.Adam([
#         {'params': classifier_params, 'lr': 1e-3},
#         {'params': feature_params, 'lr': 1e-5}
#     ])
#
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='max', patience=2, factor=0.1
#     )
#
#     best_val_auc = 0.0
#
#     for epoch in range(num_epochs):
#         # Training phase
#         model.train()
#         running_loss = 0.0
#         all_train_labels = []
#         all_train_outputs = []
#
#         pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
#         for inputs, labels in pbar:
#             inputs, labels = inputs.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#             all_train_labels.append(labels.cpu().numpy())
#             all_train_outputs.append(outputs.detach().cpu().numpy())
#
#             pbar.set_postfix({'loss': f'{loss.item():.4f}'})
#
#         # Calculate training metrics
#         all_train_labels = np.vstack(all_train_labels)
#         all_train_outputs = np.vstack(all_train_outputs)
#         train_aucs = []
#
#         for i in range(all_train_outputs.shape[1]):
#             if len(np.unique(all_train_labels[:, i])) > 1:
#                 auc = roc_auc_score(all_train_labels[:, i], all_train_outputs[:, i])
#                 train_aucs.append(auc)
#
#         train_auc = np.mean(train_aucs)
#
#         # Validation phase
#         model.eval()
#         val_loss = 0.0
#         all_val_labels = []
#         all_val_outputs = []
#
#         with torch.no_grad():
#             for inputs, labels in valid_loader:
#                 inputs, labels = inputs.to(device), labels.to(device)
#                 outputs = model(inputs)
#                 loss = criterion(outputs, labels)
#                 val_loss += loss.item()
#
#                 all_val_labels.append(labels.cpu().numpy())
#                 all_val_outputs.append(outputs.cpu().numpy())
#
#         all_val_labels = np.vstack(all_val_labels)
#         all_val_outputs = np.vstack(all_val_outputs)
#         val_aucs = []
#
#         # Calculate F1-score
#         # Convert predictions to binary (0 or 1)
#         val_predictions = (all_val_outputs > 0.5).astype(int)
#
#         # Calculate F1 score for each class
#         from sklearn.metrics import f1_score
#         f1_scores = []
#         for i in range(all_val_outputs.shape[1]):
#             if len(np.unique(all_val_labels[:, i])) > 1:
#                 f1 = f1_score(all_val_labels[:, i], val_predictions[:, i])
#                 f1_scores.append(f1)
#                 auc = roc_auc_score(all_val_labels[:, i], all_val_outputs[:, i])
#                 val_aucs.append(auc)
#
#         val_auc = np.mean(val_aucs)
#         macro_f1 = np.mean(f1_scores)
#
#         scheduler.step(val_auc)
#
#         print(f'\nEpoch {epoch + 1}/{num_epochs}:')
#         print(f'Train Loss: {running_loss / len(train_loader):.4f}, Train AUC: {train_auc:.4f}')
#         print(f'Val Loss: {val_loss / len(valid_loader):.4f}, Val AUC: {val_auc:.4f}, Val Macro F1: {macro_f1:.4f}')
#
#         # Save best model
#         if val_auc > best_val_auc:
#             best_val_auc = val_auc
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'best_auc': best_val_auc,
#                 'best_f1': macro_f1,
#             }, 'best_chexnet_finetuned.pth')

def train_model(model, train_loader, valid_loader, device, num_epochs=10):
    # Calculate class weights based on inverse frequency
    all_labels = []
    for _, labels in train_loader.dataset:
        all_labels.append(labels.numpy())
    labels_array = np.vstack(all_labels)

    # Calculate positive weights for weighted BCE loss
    pos_weights = []
    for i in range(labels_array.shape[1]):
        neg_count = len(labels_array) - np.sum(labels_array[:, i])
        pos_count = np.sum(labels_array[:, i])
        weight = neg_count / pos_count if pos_count > 0 else 1.0
        pos_weights.append(weight)

    pos_weights = torch.FloatTensor(pos_weights).to(device)

    # Use BCEWithLogitsLoss instead of BCELoss for better numerical stability
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    # Modify optimizer with weight decay for regularization
    optimizer = optim.AdamW([
        {'params': model.densenet121.classifier.parameters(), 'lr': 1e-4},
        {'params': model.densenet121.features.parameters(), 'lr': 1e-5}
    ], weight_decay=0.01)

    # Use cosine annealing scheduler for better convergence
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2, eta_min=1e-6
    )

    best_val_f1 = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        all_train_labels = []
        all_train_outputs = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Add L1 regularization
            l1_lambda = 0.01
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + l1_lambda * l1_norm

            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            # Apply sigmoid here since we're using BCEWithLogitsLoss
            all_train_labels.append(labels.cpu().numpy())
            all_train_outputs.append(torch.sigmoid(outputs).detach().cpu().numpy())

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Calculate metrics
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

        # Calculate validation metrics
        val_preds = (all_val_outputs > 0.5).astype(int)
        val_f1s = []
        val_aucs = []

        for i in range(all_val_outputs.shape[1]):
            if len(np.unique(all_val_labels[:, i])) > 1:
                val_f1s.append(f1_score(all_val_labels[:, i], val_preds[:, i]))
                val_aucs.append(roc_auc_score(all_val_labels[:, i], all_val_outputs[:, i]))

        val_f1 = np.mean(val_f1s)
        val_auc = np.mean(val_aucs)

        # Update scheduler
        scheduler.step()

        print(f'\nEpoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {running_loss / len(train_loader):.4f}, '
              f'Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f}')
        print(f'Val Loss: {val_loss / len(valid_loader):.4f}, '
              f'Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}')

        # Early stopping based on F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_val_f1,
                'best_auc': val_auc,
            }, 'best_chexnet_finetuned_imb_16k.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'\nEarly stopping triggered after {epoch + 1} epochs')
                break

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set paths - now only need CSV files
    train_file = "split_data/train_16k.csv"
    valid_file = "split_data/valid_16k.csv"
    img_dir = "split_data/images"
    # Data transforms remain the same
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # Remove horizontal flip
        transforms.RandomRotation(5),  # Reduced from 15
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

    # Create datasets - modified to only use CSV files
    train_dataset = ChestXrayDataSet( img_dir, train_file, train_transform)
    valid_dataset = ChestXrayDataSet( img_dir, valid_file, val_transform)

    # Create weighted sampler for training data
    train_sampler = get_weighted_sampler(train_dataset)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,  # Use weighted sampler instead of shuffle
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
    model = ModifiedCheXNet(
        num_classes=14,
        pretrained_path='chexnet/CheXNet/model.pth.tar'
    ).to(device)

    # Print trainable layers
    model.print_trainable_parameters()

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        num_epochs=20
    )


if __name__ == '__main__':
    main()