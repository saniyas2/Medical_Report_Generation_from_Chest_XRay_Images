# app.py

import streamlit as st
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from pathlib import Path
import logging
import sys
import os
import numpy as np
import torchvision.transforms as transforms
from typing import Union, List, Dict

# Import the MedicalReportGenerator from the appropriate modules with aliases
from report_generator_bioclip import MedicalReportGenerator as BioClipMedicalReportGenerator
from report_generator_concat import MedicalReportGenerator as BioViltMedicalReportGenerator

# Import the ModifiedCheXNet model class
from chexnet_train import ModifiedCheXNet

# Import BioVilt specific modules
from alignment_concat import ImageTextAlignmentModel
from biovil_t.pretrained import get_biovil_t_image_encoder  # Ensure this import path is correct

# Additional imports for BioVilt pipeline
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import re

# Suppress excessive warnings from transformers and torchvision
import warnings
warnings.filterwarnings("ignore")

# To disable torchvision beta transforms warnings
import torchvision
if hasattr(torchvision, 'disable_beta_transforms_warning'):
    torchvision.disable_beta_transforms_warning()

# Import torchxrayvision
import torchxrayvision as xrv

# ---------------------- Grayscale Classification ---------------------- #

def is_grayscale(image: Image.Image, threshold: float = 90.0) -> bool:
    """
    Determine if the image is predominantly grayscale.
    Removed multiple checks and kept only one check 
    """
    try:
        # Ensure image is in RGB
        image = image.convert("RGB")
        w, h = image.size
        pixels = image.getdata()
        grayscale_pixels = sum(1 for pixel in pixels if pixel[0] == pixel[1] == pixel[2])
        total_pixels = w * h
        grayscale_percentage = (grayscale_pixels / total_pixels) * 100
        return grayscale_percentage > threshold
    except Exception as e:
        logging.error(f"Error in is_grayscale: {e}")
        return False

# ---------------------- Inference Pipelines ---------------------- #

class ChestXrayFullInference:
    def __init__(
        self,
        chexnet_model_path: str,
        blip2_model_name: str = "Salesforce/blip2-opt-2.7b",
        blip2_device_map: str = 'auto',
        chexnet_num_classes: int = 14,
        report_generator_checkpoint: str = None,
        device: str = None
    ):
        """
        Initialize the full inference pipeline with CheXNet, BLIP-2, and BioClip MedicalReportGenerator.
        
        Args:
            chexnet_model_path (str): Path to the trained CheXNet model checkpoint.
            blip2_model_name (str): Hugging Face model name for BLIP-2.
            blip2_device_map (str): Device mapping for BLIP-2 ('auto' by default).
            chexnet_num_classes (int): Number of classes for CheXNet.
            report_generator_checkpoint (str): Path to the BioClip MedicalReportGenerator checkpoint.
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.logger = self._setup_logger()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Initialize CheXNet Predictor
        self.chexnet_predictor = self._initialize_chexnet(
            chexnet_model_path, chexnet_num_classes
        )

        # Initialize BLIP-2 Processor and Model
        self.processor, self.blip_model = self._initialize_blip2(
            blip2_model_name, blip2_device_map
        )

        # Initialize BioClip MedicalReportGenerator
        self.report_generator = self._initialize_report_generator(
            report_generator_checkpoint
        )

        # Define label columns
        self.label_columns = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('ChestXrayFullInference')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)

        return logger

    def _initialize_chexnet(self, model_path: str, num_classes: int) -> ModifiedCheXNet:
        """Initialize the CheXNet model."""
        try:
            self.logger.info("Initializing CheXNet model...")
            chexnet = ModifiedCheXNet(num_classes=num_classes).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                chexnet.load_state_dict(checkpoint['model_state_dict'])
            else:
                chexnet.load_state_dict(checkpoint)

            chexnet.eval()
            self.logger.info("CheXNet model loaded successfully.")
            return chexnet

        except Exception as e:
            self.logger.error(f"Error initializing CheXNet model: {str(e)}")
            raise

    def _initialize_blip2(
        self, model_name: str, device_map: str
    ) -> (Blip2Processor, Blip2ForConditionalGeneration):
        """Initialize the BLIP-2 processor and model."""
        try:
            self.logger.info("Initializing BLIP-2 model and processor...")
            processor = Blip2Processor.from_pretrained(model_name, force_download=True)
            blip_model = Blip2ForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map=device_map
            )
            blip_model.eval()
            self.logger.info("BLIP-2 model and processor loaded successfully.")
            return processor, blip_model

        except Exception as e:
            self.logger.error(f"Error initializing BLIP-2 model: {str(e)}")
            raise

    def _initialize_report_generator(self, checkpoint_path: str) -> BioClipMedicalReportGenerator:
        """Initialize the BioClip MedicalReportGenerator."""
        try:
            self.logger.info("Initializing BioClip MedicalReportGenerator...")
            vision_hidden_size = self.blip_model.vision_model.config.hidden_size
            report_gen = BioClipMedicalReportGenerator(input_embedding_dim=vision_hidden_size)

            # Load trained weights
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            report_gen.load_state_dict(checkpoint['model_state_dict'])
            report_gen.to(self.device)
            report_gen.eval()
            self.logger.info("BioClip MedicalReportGenerator loaded successfully.")
            return report_gen

        except Exception as e:
            self.logger.error(f"Error initializing BioClip MedicalReportGenerator: {str(e)}")
            raise

    def _get_transform(self) -> transforms.Compose:
        """Get the transformation pipeline for CheXNet."""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])

    def _convert_labels_to_findings(self, binary_labels: List[int]) -> str:
        """Convert binary labels to a comma-separated string of findings."""
        findings = [label for label, val in zip(self.label_columns, binary_labels) if val == 1]
        return ", ".join(findings) if findings else "No Findings"

    def predict_labels(self, image: Image.Image, threshold: float = 0.5) -> List[int]:
        """
        Predict binary labels for the given image using CheXNet.
        
        Args:
            image (PIL.Image.Image): Input image.
            threshold (float): Probability threshold for positive prediction.
        
        Returns:
            List[int]: Binary labels (0 or 1) for each condition.
        """
        try:
            self.logger.info("Predicting labels using CheXNet...")
            transform = self._get_transform()
            image_tensor = transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.chexnet_predictor(image_tensor)
                probabilities = torch.sigmoid(output).cpu().numpy()[0]

            binary_labels = [1 if prob >= threshold else 0 for prob in probabilities]
            self.logger.info(f"Predicted binary labels: {binary_labels}")
            return binary_labels

        except Exception as e:
            self.logger.error(f"Error predicting labels: {str(e)}")
            raise

    def extract_image_features(self, image: Image.Image) -> torch.Tensor:
        """
        Extract image features using BLIP-2.
        
        Args:
            image (PIL.Image.Image): Input image.
        
        Returns:
            torch.Tensor: Image features tensor.
        """
        try:
            self.logger.info("Extracting image features using BLIP-2...")
            processed = self.processor(images=image, return_tensors="pt")
            pixel_values = processed.pixel_values.to(self.device)

            with torch.no_grad():
                vision_outputs = self.blip_model.vision_model(pixel_values)
                image_features = vision_outputs.pooler_output

            self.logger.info(f"Extracted image features with shape: {image_features.shape}")
            return image_features

        except Exception as e:
            self.logger.error(f"Error extracting image features: {str(e)}")
            raise

    def generate_report(self, image: Union[str, Path, Image.Image], threshold: float = 0.5) -> Dict:
        """
        Generate a medical report for the given chest X-ray image.
        
        Args:
            image (str, Path, or PIL.Image.Image): Input image or path to the image.
            threshold (float): Probability threshold for positive prediction.
        
        Returns:
            Dict: Contains the generated report and binary labels.
        """
        try:
            if isinstance(image, (str, Path)):
                self.logger.info(f"Generating report for image path: {image}")
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file {image_path} does not exist.")
                # Load image
                image = Image.open(image_path).convert('RGB')
            elif isinstance(image, Image.Image):
                self.logger.info("Generating report for uploaded image.")
            else:
                raise TypeError("Image must be a string path or a PIL.Image.Image object.")

            # Predict labels
            binary_labels = self.predict_labels(image, threshold=threshold)

            # Convert binary labels to findings string
            findings = self._convert_labels_to_findings(binary_labels)
            prompt = f"Findings: {findings}."

            # Tokenize prompt
            self.logger.info("Tokenizing prompt...")
            prompt_encoding = self.report_generator.tokenizer(
                [prompt],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)

            # Extract image features
            image_features = self.extract_image_features(image)

            # Start report generation
            self.logger.info("Starting report generation...")
            # Corrected: Do not pass 'prompt' argument
            generated_report = self.report_generator.generate_report(
                input_embeddings=image_features,
                labels=torch.tensor(binary_labels, dtype=torch.float32).unsqueeze(0).to(self.device)
            )
            self.logger.info("Report generation completed.")

            # Check if generated_report is a list or similar iterable
            if isinstance(generated_report, (list, tuple)):
                if len(generated_report) == 0:
                    raise ValueError("MedicalReportGenerator returned an empty report list.")
                generated_report_text = generated_report[0]
            elif isinstance(generated_report, str):
                generated_report_text = generated_report
            else:
                raise TypeError("MedicalReportGenerator.generate_report returned an unsupported type.")

            # Create labels dictionary
            labels_dict = {
                label: int(val) for label, val in zip(self.label_columns, binary_labels)
            }

            self.logger.info("Report generation successful.")
            return {
                'report': generated_report_text,
                'labels': labels_dict
            }

        except Exception as e:
            self.logger.error(f"Error generating report: {str(e)}")
            raise


class ChestXrayBioViltInference:
    def __init__(
        self,
        chexnet_model_path: str,
        biovilt_checkpoint_path: str,
        device: str = None
    ):
        """
        Initialize the inference pipeline with CheXNet and BioVilt + BioGPT.
        
        Args:
            chexnet_model_path (str): Path to the trained CheXNet model checkpoint.
            biovilt_checkpoint_path (str): Path to the BioVilt + BioGPT model checkpoint.
            device (str): Device to use ('cuda' or 'cpu').
        """
        self.logger = self._setup_logger()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")

        # Initialize CheXNet Predictor
        self.chexnet_predictor = self._initialize_chexnet(
            chexnet_model_path, num_classes=14  # Corrected parameter name
        )

        # Initialize BioVilt components
        self.image_encoder, self.alignment_model, self.report_generator = self._initialize_biovilt(
            biovilt_checkpoint_path
        )

        # Define label columns
        self.label_columns = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('ChestXrayBioViltInference')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            logger.addHandler(handler)

        return logger

    def _initialize_chexnet(self, model_path: str, num_classes: int) -> ModifiedCheXNet:
        """Initialize the CheXNet model."""
        try:
            self.logger.info("Initializing CheXNet model for BioVilt pipeline...")
            chexnet = ModifiedCheXNet(num_classes=num_classes).to(self.device)
            checkpoint = torch.load(model_path, map_location=self.device)

            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                chexnet.load_state_dict(checkpoint['model_state_dict'])
            else:
                chexnet.load_state_dict(checkpoint)

            chexnet.eval()
            self.logger.info("CheXNet model loaded successfully for BioVilt pipeline.")
            return chexnet

        except Exception as e:
            self.logger.error(f"Error initializing CheXNet model for BioVilt pipeline: {str(e)}")
            raise

    def _initialize_biovilt(self, checkpoint_path: str):
        """Initialize BioVilt Image Encoder, Alignment Model, and Report Generator."""
        try:
            self.logger.info("Initializing BioVilt Image Encoder, Alignment Model, and Report Generator...")
            image_encoder, alignment_model, report_generator = load_biovilt_checkpoint(
                checkpoint_path, self.device
            )
            self.logger.info("BioVilt components loaded successfully.")
            return image_encoder, alignment_model, report_generator

        except Exception as e:
            self.logger.error(f"Error initializing BioVilt components: {str(e)}")
            raise

    def _get_transform(self) -> A.Compose:
        """Get the transformation pipeline for CheXNet."""
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])

    def _convert_labels_to_findings(self, binary_labels: List[int]) -> str:
        """Convert binary labels to a comma-separated string of findings."""
        findings = [label for label, val in zip(self.label_columns, binary_labels) if val == 1]
        return ", ".join(findings) if findings else "No Findings"

    def predict_labels(self, image: Image.Image, threshold: float = 0.5) -> List[int]:
        """
        Predict binary labels for the given image using CheXNet.
        
        Args:
            image (PIL.Image.Image): Input image.
            threshold (float): Probability threshold for positive prediction.
        
        Returns:
            List[int]: Binary labels (0 or 1) for each condition.
        """
        try:
            self.logger.info("Predicting labels using CheXNet for BioVilt pipeline...")
            transform = self._get_transform()
            image_np = np.array(image)
            transformed = transform(image=image_np)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.chexnet_predictor(image_tensor)
                probabilities = torch.sigmoid(output).cpu().numpy()[0]

            binary_labels = [1 if prob >= threshold else 0 for prob in probabilities]
            self.logger.info(f"Predicted binary labels for BioVilt pipeline: {binary_labels}")
            return binary_labels

        except Exception as e:
            self.logger.error(f"Error predicting labels for BioVilt pipeline: {str(e)}")
            raise

    def generate_report(self, image: Union[str, Path, Image.Image], threshold: float = 0.5) -> Dict:
        """
        Generate a medical report for the given chest X-ray image using BioVilt + BioGPT.
        
        Args:
            image (str, Path, or PIL.Image.Image): Input image or path to the image.
            threshold (float): Probability threshold for positive prediction.
        
        Returns:
            Dict: Contains the generated report and binary labels.
        """
        try:
            if isinstance(image, (str, Path)):
                self.logger.info(f"Generating BioVilt report for image path: {image}")
                image_path = Path(image)
                if not image_path.exists():
                    raise FileNotFoundError(f"Image file {image_path} does not exist.")
                # Load image
                image = Image.open(image_path).convert('RGB')
            elif isinstance(image, Image.Image):
                self.logger.info("Generating BioVilt report for uploaded image.")
            else:
                raise TypeError("Image must be a string path or a PIL.Image.Image object.")

            # Predict labels
            binary_labels = self.predict_labels(image, threshold=threshold)

            # Convert binary labels to findings string
            findings = self._convert_labels_to_findings(binary_labels)
            prompt = f"Findings: {findings}."

            # Tokenize prompt
            self.logger.info("Tokenizing prompt...")
            prompt_encoding = self.report_generator.tokenizer(
                [prompt],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(self.device)

            # Extract image embeddings using BioVilt Image Encoder
            self.logger.info("Extracting image embeddings using BioVilt Image Encoder...")
            image_np = np.array(image)
            transform = A.Compose([
                A.Resize(224, 224),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
            transformed = transform(image=image_np)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_encoder_output = self.image_encoder(image_tensor)
                # Extract the tensor from ImageModelOutput
                if hasattr(image_encoder_output, 'img_embedding'):
                    image_embeddings = image_encoder_output.img_embedding
                else:
                    raise AttributeError("Image encoder output does not have 'img_embedding' attribute.")

            # Generate medical report
            self.logger.info("Generating medical report using BioVilt + BioGPT...")
            generated_report = self.report_generator(
                image_embeddings=image_embeddings,
                prompt_input_ids=prompt_encoding['input_ids'],
                target_ids=None  # Not needed during inference
            )
            self.logger.info("Report generation completed using BioVilt + BioGPT.")

            # Check if generated_report is a list or similar iterable
            if isinstance(generated_report, (list, tuple)):
                if len(generated_report) == 0:
                    raise ValueError("MedicalReportGenerator returned an empty report list.")
                generated_report_text = generated_report[0]
            elif isinstance(generated_report, str):
                generated_report_text = generated_report
            else:
                raise TypeError("MedicalReportGenerator.generate_report returned an unsupported type.")

            # Clean the generated report
            cleaned_report = self.clean_report(generated_report_text)

            # Create labels dictionary
            labels_dict = {
                label: int(val) for label, val in zip(self.label_columns, binary_labels)
            }

            self.logger.info("BioVilt report generation successful.")
            return {
                'report': cleaned_report,
                'labels': labels_dict
            }

        except Exception as e:
            self.logger.error(f"Error generating BioVilt report: {str(e)}")
            raise

    def clean_report(self, text: str) -> str:
        """
        Remove non-English characters, any occurrence of 'madeupword' followed by digits,
        and discard any text after the last period.
        
        Args:
            text (str): The generated medical report text.
        
        Returns:
            str: The cleaned medical report.
        """
        try:
            self.logger.info("Cleaning the generated BioVilt report...")

            # Remove 'madeupword' followed by any number of digits
            text = re.sub(r'madeupword\d+', '', text, flags=re.IGNORECASE)

            # Remove any non-ASCII characters
            text = text.encode('ascii', 'ignore').decode('ascii')

            # Remove extra spaces created by removals
            text = ' '.join(text.split())

            # Truncate the text after the last period
            last_period_index = text.rfind('.')
            if last_period_index != -1:
                text = text[:last_period_index + 1]
            else:
                # If no period is found, return the text as is
                self.logger.warning("No period found in the text. Returning the original text.")

            self.logger.info("BioVilt report cleaned successfully.")
            return text

        except Exception as e:
            self.logger.error(f"Error cleaning BioVilt report: {str(e)}")
            raise

def load_biovilt_checkpoint(checkpoint_path: str, device: torch.device):
    """
    Load the BioVilt checkpoint and initialize the models.
    
    Args:
        checkpoint_path (str): Path to the BioVilt checkpoint.
        device (torch.device): Device to load the models onto.
    
    Returns:
        Tuple containing image_encoder, alignment_model, report_generator
    """
    logging.info(f"Loading BioVilt checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize models
    image_encoder = get_biovil_t_image_encoder()
    alignment_model = ImageTextAlignmentModel(image_embedding_dim=512)
    report_generator = BioViltMedicalReportGenerator(image_embedding_dim=512)

    # Load state dicts
    image_encoder.load_state_dict(checkpoint['image_encoder_state_dict'])
    alignment_model.load_state_dict(checkpoint['alignment_model_state_dict'])
    report_generator.load_state_dict(checkpoint['report_generator_state_dict'])

    # Move to device
    image_encoder = image_encoder.to(device)
    alignment_model = alignment_model.to(device)
    report_generator = report_generator.to(device)

    # Set to eval mode
    image_encoder.eval()
    alignment_model.eval()
    report_generator.eval()

    logging.info("BioVilt models loaded successfully.")
    return image_encoder, alignment_model, report_generator

def load_bioclip_checkpoint(checkpoint_path: str, device: torch.device) -> BioClipMedicalReportGenerator:
    """
    Load the BioClip MedicalReportGenerator checkpoint.
    
    Args:
        checkpoint_path (str): Path to the BioClip MedicalReportGenerator checkpoint.
        device (torch.device): Device to load the model onto.
    
    Returns:
        BioClipMedicalReportGenerator: The loaded MedicalReportGenerator model.
    """
    logging.info(f"Loading BioClip MedicalReportGenerator checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Initialize BioClip MedicalReportGenerator
    vision_hidden_size = 768  # Update this based on your model's hidden size
    report_generator = BioClipMedicalReportGenerator(input_embedding_dim=vision_hidden_size)

    # Load state dict
    report_generator.load_state_dict(checkpoint['model_state_dict'])

    # Move to device and set to eval mode
    report_generator.to(device)
    report_generator.eval()

    logging.info("BioClip MedicalReportGenerator loaded successfully.")
    return report_generator

# ---------------------- Streamlit Application ---------------------- #

def main():
    st.set_page_config(page_title="Chest X-ray Medical Report Generator", layout="centered")
    st.title("Chest X-ray Medical Report Generator")

    st.markdown("""
    Upload a chest X-ray image, and click the **Generate Report** button to receive a detailed medical report along with predicted conditions.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload a chest X-ray image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Chest X-ray Image', use_container_width=True)

        # Perform Grayscale Classification
        with st.spinner("Verifying if the uploaded image is a chest X-ray..."):
            is_cxr = is_grayscale(image, threshold=90.0)  # Adjust threshold as needed

        if not is_cxr:
            st.error("This image is not a chest X-ray image, please upload a chest X-ray image.")
            st.stop()  # Stop further execution
        else:
            st.success("Image verified as a chest X-ray. Proceeding with report generation.")

        # Initialize the inference pipelines
        @st.cache_resource
        def load_inference_pipelines():
            # Paths for BLIP2 + BioGPT
            blip2_model_name = "Salesforce/blip2-opt-2.7b"
            blip2_device_map = 'auto'
            blip2_checkpoint = r"C:\Users\anand\Downloads\checkpoint_epoch_20.pt"  # Update path as needed

            blip2_pipeline = ChestXrayFullInference(
                chexnet_model_path=r"C:\Users\anand\Downloads\best_chexnet_finetuned_16_f1.pth",  # Update path as needed
                blip2_model_name=blip2_model_name,
                blip2_device_map=blip2_device_map,
                chexnet_num_classes=14,
                report_generator_checkpoint=blip2_checkpoint
            )

            # Paths for BioVilt + BioGPT
            biovilt_checkpoint_path = r"C:\Users\anand\Downloads\model_epoch_7.pt"  # Update path as needed

            biovilt_pipeline = ChestXrayBioViltInference(
                chexnet_model_path=r"C:\Users\anand\Downloads\best_chexnet_finetuned_16_f1.pth",  # Update path as needed
                biovilt_checkpoint_path=biovilt_checkpoint_path
            )

            return blip2_pipeline, biovilt_pipeline

        try:
            blip2_pipeline, biovilt_pipeline = load_inference_pipelines()
        except Exception as e:
            st.error(f"Failed to load inference pipelines: {e}")
            st.stop()

        # Define buttons for model selection
        col1, col2 = st.columns(2)

        with col1:
            blip2_button = st.button("Generate Report with BLIP2 + BioGPT")

        with col2:
            biovilt_button = st.button("Generate Report with BioVilt + BioGPT")

        # Handle BLIP2 + BioGPT report generation
        if blip2_button:
            with st.spinner("Generating report with BLIP2 + BioGPT..."):
                try:
                    result = blip2_pipeline.generate_report(image, threshold=0.65)
                    
                    # Display the report
                    st.subheader("Generated Medical Report (BLIP2 + BioGPT)")
                    st.write(result['report'])

                except Exception as e:
                    st.error(f"Failed to generate BLIP2 + BioGPT report: {e}")

        # Handle BioVilt + BioGPT report generation
        if biovilt_button:
            with st.spinner("Generating report with BioVilt + BioGPT..."):
                try:
                    result = biovilt_pipeline.generate_report(image, threshold=0.65)
                    
                    # Display the report
                    st.subheader("Generated Medical Report (BioVilt + BioGPT)")
                    st.write(result['report'])

                except Exception as e:
                    st.error(f"Failed to generate BioVilt + BioGPT report: {e}")

if __name__ == "__main__":
    import pandas as pd  
    main()
