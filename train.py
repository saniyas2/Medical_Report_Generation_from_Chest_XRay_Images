import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import logging
import wandb
from pathlib import Path
from typing import Dict, Any
from torch.cuda.amp import autocast, GradScaler
from datetime import datetime

from data2 import data_processing
from alignment_model import ImageTextAlignmentModel
from report_generator import MedicalReportGenerator
from biovil_t.pretrained import get_biovil_t_image_encoder  # Ensure this import path is correct
from rouge_score import rouge_scorer

def train_epoch(image_encoder, alignment_model, report_generator, train_loader,
                contrastive_loss, alignment_optimizer, generator_optimizer,
                alignment_scheduler, generator_scheduler, scaler, device,
                gradient_accumulation_steps, max_grad_norm, epoch):
    alignment_model.train()
    report_generator.train()
    image_encoder.eval()

    # Metrics tracking
    total_train_loss = 0.0
    total_align_loss = 0.0
    total_gen_loss = 0.0
    total_samples = 0

    progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch}')

    for batch_idx, (images, findings_texts, findings_lists) in enumerate(progress_bar):
        images = images.to(device)
        batch_size = images.size(0)
        total_samples += batch_size

        # Get image embeddings
        with torch.no_grad():
            image_embeddings = image_encoder(images).img_embedding

        # Create prompts using findings_lists (for generation)
        batch_prompts = [
            f"Findings: {', '.join(findings) if findings else 'No Findings'}."
            for findings in findings_lists
        ]

        # Use findings_texts (actual findings) for alignment
        actual_findings = findings_texts

        # Mixed precision training
        with autocast():
            # Alignment phase
            projected_image, projected_text = alignment_model(image_embeddings, actual_findings)

            # Contrastive loss
            labels = torch.ones(batch_size).to(device)
            align_loss = contrastive_loss(projected_image, projected_text, labels)
            align_loss = align_loss / gradient_accumulation_steps

        # Scale and accumulate alignment gradients
        scaler.scale(align_loss).backward()

        # Generation phase

        # Tokenize the prompts
        prompt_encoding = report_generator.tokenizer(
            batch_prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(device)

        # Tokenize target texts (actual findings)
        target_encoding = report_generator.tokenizer(
            actual_findings,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        ).to(device)

        with autocast():
            gen_loss, _ = report_generator(
                image_embeddings=image_embeddings.detach(),
                prompt_input_ids=prompt_encoding['input_ids'],
                target_ids=target_encoding['input_ids']
            )
            gen_loss = gen_loss / gradient_accumulation_steps

        # Scale and accumulate generator gradients
        scaler.scale(gen_loss).backward()

        # Update metrics
        total_align_loss += align_loss.item() * gradient_accumulation_steps * batch_size
        total_gen_loss += gen_loss.item() * gradient_accumulation_steps * batch_size
        total_train_loss += (align_loss.item() + gen_loss.item()) * gradient_accumulation_steps * batch_size

        # Step optimizers and schedulers
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            # Unscale gradients
            scaler.unscale_(alignment_optimizer)
            scaler.unscale_(generator_optimizer)

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                alignment_model.parameters(), max_grad_norm
            )
            torch.nn.utils.clip_grad_norm_(
                report_generator.parameters(), max_grad_norm
            )

            # Step optimizers
            scaler.step(alignment_optimizer)
            scaler.step(generator_optimizer)
            scaler.update()

            # Zero gradients
            alignment_optimizer.zero_grad()
            generator_optimizer.zero_grad()

            # Step schedulers
            alignment_scheduler.step()
            generator_scheduler.step()

        # Update progress bar
        progress_bar.set_postfix({
            'align_loss': f"{align_loss.item():.4f}",
            'gen_loss': f"{gen_loss.item():.4f}"
        })

    epoch_align_loss = total_align_loss / total_samples
    epoch_gen_loss = total_gen_loss / total_samples
    epoch_train_loss = total_train_loss / total_samples

    return {
        'train_loss': epoch_train_loss,
        'train_align_loss': epoch_align_loss,
        'train_gen_loss': epoch_gen_loss,
    }

def validate_epoch(image_encoder, alignment_model, report_generator, val_loader,
                   contrastive_loss, device, epoch):
    alignment_model.eval()
    report_generator.eval()
    image_encoder.eval()

    # Metrics storage
    total_val_loss = 0.0
    total_align_loss = 0.0
    total_gen_loss = 0.0
    total_samples = 0
    all_generated = []
    all_references = []

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f'Validation Epoch {epoch}')

        for batch_idx, (images, findings_texts, findings_lists) in enumerate(progress_bar):
            images = images.to(device)
            batch_size = images.size(0)
            total_samples += batch_size

            # Get image embeddings
            image_embeddings = image_encoder(images).img_embedding

            # Create prompts using findings_lists
            batch_prompts = [
                f"Findings: {', '.join(findings) if findings else 'No Findings'}."
                for findings in findings_lists
            ]

            # Actual findings for alignment and reference
            actual_findings = findings_texts

            # Alignment phase
            projected_image, projected_text = alignment_model(image_embeddings, actual_findings)
            labels = torch.ones(batch_size).to(device)
            align_loss = contrastive_loss(projected_image, projected_text, labels)

            # Generation phase
            prompt_encoding = report_generator.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(device)

            target_encoding = report_generator.tokenizer(
                actual_findings,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(device)

            # Compute generation loss
            gen_loss, _ = report_generator(
                image_embeddings=image_embeddings,
                prompt_input_ids=prompt_encoding['input_ids'],
                target_ids=target_encoding['input_ids']
            )

            # Generate text for evaluation
            generated_texts = report_generator(
                image_embeddings=image_embeddings,
                prompt_input_ids=prompt_encoding['input_ids'],
                target_ids=None
            )

            # Store the generated and reference texts for ROUGE calculation
            all_generated.extend(generated_texts)
            all_references.extend(actual_findings)

            # Update totals
            total_align_loss += align_loss.item() * batch_size
            total_gen_loss += gen_loss.item() * batch_size
            total_val_loss += (align_loss.item() + gen_loss.item()) * batch_size

            # Print sample generation
            if batch_idx % 10 == 0:
                print(f"\nSample Generation (Batch {batch_idx}):")
                print(f"Generated: {generated_texts[0]}")
                print(f"Reference: {actual_findings[0]}")
                # Also display the pathologies findings from findings_lists
                print(f"Pathologies/Findings List: {findings_lists[0]}\n")

        # Calculate overall metrics
        epoch_align_loss = total_align_loss / total_samples
        epoch_gen_loss = total_gen_loss / total_samples
        epoch_val_loss = total_val_loss / total_samples

    # Compute ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l_scores = []
    for ref, gen in zip(all_references, all_generated):
        score = scorer.score(ref, gen)['rougeL'].fmeasure
        rouge_l_scores.append(score)
    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

    # Display validation losses and ROUGE-L
    print(f"\nEpoch {epoch} Validation Metrics:")
    print(f"Validation Loss: {epoch_val_loss:.4f}")
    print(f"Alignment Loss: {epoch_align_loss:.4f}")
    print(f"Generation Loss: {epoch_gen_loss:.4f}")
    print(f"ROUGE-L: {avg_rouge_l:.4f}")

    return {
        'val_loss': epoch_val_loss,
        'val_align_loss': epoch_align_loss,
        'val_gen_loss': epoch_gen_loss,
        'val_rouge_l': avg_rouge_l
    }


def train_model(
        csv_with_image_paths: str,
        csv_with_labels: str,
        num_epochs: int = 30,
        batch_size: int = 8,
        train_split: float = 0.85,
        num_workers: int = 4,
        learning_rate: float = 2e-4,
        warmup_steps: int = 1000,
        gradient_accumulation_steps: int = 4,
        max_grad_norm: float = 1.0,
        use_wandb: bool = True,
        checkpoint_dir: str = "checkpoints",
        seed: int = 42
):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize models
    image_encoder = get_biovil_t_image_encoder()
    alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
    report_generator = MedicalReportGenerator(image_embedding_dim=512)

    # Move models to device
    image_encoder = image_encoder.to(device)
    alignment_model = alignment_model.to(device)
    report_generator = report_generator.to(device)

    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="medical-report-generation",
            config={
                "learning_rate": learning_rate,
                "epochs": num_epochs,
                "batch_size": batch_size,
                "warmup_steps": warmup_steps,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            }
        )
        wandb.watch(models=[alignment_model, report_generator], log="all")

    # Get dataloaders
    train_loader, val_loader = data_processing.get_dataloaders(
        csv_with_image_paths=csv_with_image_paths,
        csv_with_labels=csv_with_labels,
        batch_size=batch_size,
        train_split=train_split,
        num_workers=num_workers,
        seed=seed,
    )

    # Initialize optimizers
    alignment_optimizer = AdamW(
        alignment_model.parameters(),
        lr=learning_rate,
        weight_decay=0.01
    )
    generator_optimizer = AdamW([
        {'params': report_generator.model.parameters(), 'lr': learning_rate},
        {'params': report_generator.image_projection.parameters(), 'lr': learning_rate * 10}
    ])

    # Initialize schedulers
    num_training_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
    alignment_scheduler = get_linear_schedule_with_warmup(
        alignment_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    generator_scheduler = get_linear_schedule_with_warmup(
        generator_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    # Initialize loss function and scaler
    contrastive_loss = nn.CosineEmbeddingLoss()
    scaler = GradScaler()

    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Training phase
        train_metrics = train_epoch(
            image_encoder=image_encoder,
            alignment_model=alignment_model,
            report_generator=report_generator,
            train_loader=train_loader,
            contrastive_loss=contrastive_loss,
            alignment_optimizer=alignment_optimizer,
            generator_optimizer=generator_optimizer,
            alignment_scheduler=alignment_scheduler,
            generator_scheduler=generator_scheduler,
            scaler=scaler,
            device=device,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            epoch=epoch + 1
        )

        # Validation phase
        val_metrics = validate_epoch(
            image_encoder=image_encoder,
            alignment_model=alignment_model,
            report_generator=report_generator,
            val_loader=val_loader,
            contrastive_loss=contrastive_loss,
            device=device,
            epoch=epoch + 1
        )

        # Display training and validation losses
        print(f"\nEpoch {epoch + 1} Training Loss: {train_metrics['train_loss']:.4f}")
        print(f"Epoch {epoch + 1} Validation Loss: {val_metrics['val_loss']:.4f}")
        print(f"Alignment Loss - Train: {train_metrics['train_align_loss']:.4f}, Val: {val_metrics['val_align_loss']:.4f}")
        print(f"Generation Loss - Train: {train_metrics['train_gen_loss']:.4f}, Val: {val_metrics['val_gen_loss']:.4f}")
        print(f"ROUGE-L (Val): {val_metrics['val_rouge_l']:.4f}")

        # Log metrics to wandb
        if use_wandb:
            wandb.log({**train_metrics, **val_metrics})

        # Save model checkpoint after each epoch
        checkpoint_save_path = checkpoint_dir / f"model_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'image_encoder_state_dict': image_encoder.state_dict(),
            'alignment_model_state_dict': alignment_model.state_dict(),
            'report_generator_state_dict': report_generator.state_dict(),
            'alignment_optimizer_state_dict': alignment_optimizer.state_dict(),
            'generator_optimizer_state_dict': generator_optimizer.state_dict(),
            'alignment_scheduler_state_dict': alignment_scheduler.state_dict(),
            'generator_scheduler_state_dict': generator_scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'config': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'max_grad_norm': max_grad_norm,
            }
        }, checkpoint_save_path)
        logging.info(f"Saved checkpoint: {checkpoint_save_path}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Path to your CSV files
    csv_with_image_paths = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/final.csv"
    csv_with_labels = "/home/ubuntu/NLP/NLP_Project/Temp_3_NLP/Data/labeled_reports_with_images.csv"

    # Training configuration
    config = {
        'num_epochs': 30,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'warmup_steps': 1000,
        'gradient_accumulation_steps': 4,
        'use_wandb': True,
        'checkpoint_dir': 'checkpoints',
        'seed': 42
    }

    # Start training
    train_model(
        csv_with_image_paths=csv_with_image_paths,
        csv_with_labels=csv_with_labels,
        **config
    )
