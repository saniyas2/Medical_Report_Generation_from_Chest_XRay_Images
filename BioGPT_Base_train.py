import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from pathlib import Path
import json
import nltk
from rouge_score import rouge_scorer

# Ensure NLTK packages are downloaded if needed:
# nltk.download('wordnet')

from BioGPT_Base_data_processing import get_dataloaders
from BioGPT_Base_alignment_model import ImageTextAlignmentModel
from BioGPT_Base_report_generator import MedicalReportGenerator
from biovil_t.pretrained import get_biovil_t_image_encoder


def save_checkpoint(epoch: int, alignment_model: nn.Module, report_generator: MedicalReportGenerator,
                    alignment_optimizer: torch.optim.Optimizer, generator_optimizer: torch.optim.Optimizer,
                    metrics: dict, save_path: Path) -> None:
    """Save intermediate model checkpoint for training resumption"""
    checkpoint = {
        'epoch': epoch,
        'alignment_model_state_dict': alignment_model.state_dict(),
        'report_generator_model': report_generator.model.state_dict(),
        'report_generator_projection': report_generator.input_projection.state_dict(),
        'alignment_optimizer_state_dict': alignment_optimizer.state_dict(),
        'generator_optimizer_state_dict': generator_optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)


def save_best_model(alignment_model: nn.Module, report_generator: MedicalReportGenerator, metrics: dict, save_dir: Path) -> None:
    """Save the best model with metrics using the proper PEFT saving methods."""
    # Save alignment model
    torch.save(alignment_model.state_dict(), save_dir / "best_alignment_model.pt")

    # Save report generator LoRA adapter weights and configuration
    report_generator.model.save_pretrained(save_dir / "best_report_generator")

    # Save the projection layer
    torch.save(report_generator.input_projection.state_dict(),
               save_dir / "best_report_generator_projection.pt")

    # Save metrics
    with open(save_dir / 'best_model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)


def compute_metrics(references, predictions):
    """
    Compute ROUGE-L scores.
    references: List of reference strings
    predictions: List of predicted strings
    """


    try:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        rouge_l_scores = []
        for r, p in zip(references, predictions):
            try:
                score = scorer.score(r, p)['rougeL'].fmeasure
                rouge_l_scores.append(score)
            except:
                rouge_l_scores.append(0.0)
        avg_rouge_l = (sum(rouge_l_scores) / len(rouge_l_scores)) * 100.0 if rouge_l_scores else 0.0
    except Exception:
        avg_rouge_l = 0.0

    return avg_rouge_l


def train_model(csv_path: str, save_dir: str, num_epochs: int = 30):
    """
    Train the medical report generation model

    Args:
        csv_path: Path to CSV file containing image paths and reports
        save_dir: Directory to save model checkpoints and final best model
        num_epochs: Number of training epochs
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models
    image_encoder = get_biovil_t_image_encoder()
    alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
    report_generator = MedicalReportGenerator()

    # Move models to device
    image_encoder = image_encoder.to(device)
    alignment_model = alignment_model.to(device)
    report_generator = report_generator.to(device)

    # Get dataloaders
    train_loader, val_loader = get_dataloaders(csv_path)

    # Optimizers
    alignment_optimizer = AdamW(alignment_model.parameters(), lr=2e-5)
    peft_params = [p for p in report_generator.model.parameters() if p.requires_grad]
    generator_optimizer = AdamW([
        {'params': peft_params, 'lr': 2e-5},
        {'params': report_generator.input_projection.parameters(), 'lr': 1e-4}
    ])

    # Loss function for alignment
    contrastive_loss = nn.CosineEmbeddingLoss()

    # Create save directories
    save_dir = Path(save_dir)
    checkpoints_dir = save_dir / "checkpoints"
    best_model_dir = save_dir / "best_model"

    for dir_path in [checkpoints_dir, best_model_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)

    # Track best validation metrics
    best_val_loss = float('inf')
    best_metrics = None

    # Load last checkpoint if exists
    last_checkpoint = max(checkpoints_dir.glob("checkpoint_*.pt"), default=None,
                          key=lambda x: int(x.stem.split('_')[1]))
    start_epoch = 0

    if last_checkpoint:
        print(f"Loading checkpoint: {last_checkpoint}")
        checkpoint = torch.load(last_checkpoint, map_location=device)
        alignment_model.load_state_dict(checkpoint['alignment_model_state_dict'])
        report_generator.model.load_state_dict(checkpoint['report_generator_model'])
        report_generator.input_projection.load_state_dict(checkpoint['report_generator_projection'])
        alignment_optimizer.load_state_dict(checkpoint['alignment_optimizer_state_dict'])
        generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training Phase
        image_encoder.eval()  # Keep image encoder in eval mode
        alignment_model.train()
        report_generator.train()

        train_align_losses = []
        train_gen_losses = []

        progress_bar = tqdm(train_loader, desc='Training')

        for batch_idx, (images, impressions) in enumerate(progress_bar):
            images = images.to(device)

            # Get image embeddings
            with torch.no_grad():
                image_embeddings = image_encoder(images).img_embedding

            # Alignment phase
            alignment_optimizer.zero_grad()
            projected_image, projected_text = alignment_model(image_embeddings, impressions)
            batch_size = images.size(0)
            labels = torch.ones(batch_size).to(device)
            align_loss = contrastive_loss(projected_image, projected_text, labels)
            align_loss.backward()
            alignment_optimizer.step()

            # Generation phase
            generator_optimizer.zero_grad()
            target_encoding = report_generator.tokenizer(
                impressions,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=150
            ).to(device)
            target_ids = target_encoding['input_ids']
            gen_loss, logits = report_generator(projected_image.detach(), target_ids)
            gen_loss.backward()
            generator_optimizer.step()

            train_align_losses.append(align_loss.item())
            train_gen_losses.append(gen_loss.item())

            progress_bar.set_postfix({
                'Align Loss': f'{align_loss.item():.4f}',
                'Gen Loss': f'{gen_loss.item():.4f}'
            })

            # Print sample outputs every 50 batches
            if batch_idx % 50 == 0:
                with torch.no_grad():
                    sample_report = report_generator.generate_report(projected_image[0:1].detach())[0]
                    print("\nSample Generation:")
                    print(f"Generated: {sample_report}")
                    print(f"Target: {impressions[0]}\n")

        # Calculate average training losses
        avg_train_align_loss = sum(train_align_losses) / len(train_align_losses)
        avg_train_gen_loss = sum(train_gen_losses) / len(train_gen_losses)

        # Validation Phase
        alignment_model.eval()
        report_generator.eval()

        val_align_losses = []
        val_gen_losses = []
        val_references = []
        val_predictions = []

        print("\nRunning validation...")
        with torch.no_grad():
            for val_images, val_impressions in val_loader:
                val_images = val_images.to(device)
                val_image_embeddings = image_encoder(val_images).img_embedding
                val_projected_image, val_projected_text = alignment_model(val_image_embeddings, val_impressions)

                # Alignment loss
                val_labels = torch.ones(val_images.size(0)).to(device)
                val_align_loss = contrastive_loss(val_projected_image, val_projected_text, val_labels)
                val_align_losses.append(val_align_loss.item())

                # Generation loss
                val_target_encoding = report_generator.tokenizer(
                    val_impressions,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=150
                ).to(device)
                val_target_ids = val_target_encoding['input_ids']

                val_gen_loss, _ = report_generator(val_projected_image, val_target_ids)
                val_gen_losses.append(val_gen_loss.item())

                # Generate predictions for metrics
                batch_predictions = report_generator.generate_report(val_projected_image)
                val_predictions.extend(batch_predictions)
                val_references.extend(val_impressions)

        # Calculate average validation losses
        avg_val_align_loss = sum(val_align_losses) / len(val_align_losses)
        avg_val_gen_loss = sum(val_gen_losses) / len(val_gen_losses)

        # Compute metrics
        rouge_l = compute_metrics(val_references, val_predictions)

        # Current metrics
        current_metrics = {
            'epoch': epoch + 1,
            'train_align_loss': avg_train_align_loss,
            'train_gen_loss': avg_train_gen_loss,
            'val_align_loss': avg_val_align_loss,
            'val_gen_loss': avg_val_gen_loss,
            'rouge_l': rouge_l
        }

        print(f"\nEpoch {epoch + 1} Summary:")
        print(json.dumps(current_metrics, indent=4))

        # Save checkpoint for each epoch
        checkpoint_path = checkpoints_dir / f"checkpoint_{epoch}.pt"
        save_checkpoint(
            epoch=epoch,
            alignment_model=alignment_model,
            report_generator=report_generator,
            alignment_optimizer=alignment_optimizer,
            generator_optimizer=generator_optimizer,
            metrics=current_metrics,
            save_path=checkpoint_path
        )
        print(f"\nSaved checkpoint to {checkpoint_path}")

        # Save best model if validation loss improved
        val_loss = avg_val_align_loss + avg_val_gen_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = current_metrics
            print("\nSaving best model...")
            save_best_model(alignment_model, report_generator, current_metrics, best_model_dir)

            # Save pointer to best checkpoint
            with open(save_dir / "best_checkpoint.txt", "w") as f:
                f.write(f"checkpoint_{epoch}.pt")

    print("\nTraining completed!")
    if best_metrics:
        print("\nBest model metrics:")
        print(json.dumps(best_metrics, indent=4))

    return alignment_model, report_generator


if __name__ == "__main__":
    # Set paths
    csv_path = "Data/final.csv"  
    save_dir = "checkpoints"      

    # Start training
    print("Starting training...")
    alignment_model, report_generator = train_model(csv_path=csv_path,
                                                    save_dir=save_dir,
                                                    num_epochs=30)
    print("Training completed!")