import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from read_data import ChestXrayDataSet
import torchvision


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


def train_model(model, train_loader, valid_loader, device, num_epochs=10):
    criterion = nn.BCELoss()

    # Different learning rates for different parts
    classifier_params = list(model.densenet121.classifier.parameters())
    feature_params = list(model.densenet121.features.parameters())

    optimizer = optim.Adam([
        {'params': classifier_params, 'lr': 1e-3},
        {'params': feature_params, 'lr': 1e-5}
    ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                     patience=2, factor=0.1)

    best_val_auc = 0.0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')

        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_loss = running_loss / len(train_loader)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_labels = []
        all_outputs = []

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                all_labels.append(labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        val_loss = val_loss / len(valid_loader)
        all_labels = np.concatenate(all_labels)
        all_outputs = np.concatenate(all_outputs)

        # Calculate AUC for each class
        aucs = []
        for i in range(all_outputs.shape[1]):
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
                aucs.append(auc)

        # Calculate F1 score
        from sklearn.metrics import f1_score
        # Convert outputs to binary predictions using 0.5 threshold
        predictions = (all_outputs > 0.5).astype(int)
        macro_f1 = f1_score(all_labels, predictions, average='macro')

        val_auc = np.mean(aucs)
        scheduler.step(val_auc)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {epoch_loss:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val Macro F1: {macro_f1:.4f}')

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_auc': best_val_auc,
                'best_f1': macro_f1,
            }, 'best_chexnet_finetuned.pth')


def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = "split_data/images"
    train_file = "split_data/train_16k.csv"
    valid_file = "split_data/valid_16k.csv"
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
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
    train_dataset = ChestXrayDataSet(
        data_dir,
       train_file,
        train_transform
    )

    valid_dataset = ChestXrayDataSet(
       data_dir,
        valid_file,
        val_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    # In your main function:
    model = ModifiedCheXNet(
        num_classes=14,
        pretrained_path='chexnet/CheXNet/model.pth.tar'
    ).to(device)

    # Check which layers are trainable
    model.print_trainable_parameters()

    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        num_epochs=100
    )


if __name__ == '__main__':
    main()