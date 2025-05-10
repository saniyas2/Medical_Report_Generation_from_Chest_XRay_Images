# Automated Radiology Report Generation from Chest X-Rays

## Table of Contents
- [Introduction](#introduction)
- [Dataset Description](#dataset-description)
- [Methodology](#methodology)
  - [1. Data Collection and Preprocessing](#1-data-collection-and-preprocessing)
    - [a. Data Extraction](#a-data-extraction)
    - [b. Data Pre-processing](#b-data-pre-processing)
    - [c. Dataset Split](#c-dataset-split)
  - [2. Extracting Labels Using CheXbert](#2-extracting-labels-using-chexbert)
  - [3. ChexNet for Structural Findings Extraction](#3-chexnet-for-structural-findings-extraction)
  - [4. Model Architectures](#4-model-architectures)
    - [Model 1: BioVilt + Alignment + BioGPT](#model-1-biovilt--alignment--biogpt)
    - [Model 2: BioVilt + ChexNet + Alignment + BioGPT](#model-2-biovilt--chexnet--alignment--biogpt)
- [Results](#results)
- [Challenges Faced](#challenges-faced)
- [Deployment](#deployment)
- [References](#references)

---

## Introduction

In modern healthcare, radiology plays an essential role in diagnosing and managing numerous medical conditions. **Chest X-rays** are among the most widely used diagnostic tools to detect abnormalities such as **Pneumonia, Hernia, and Cardiomegaly**.

**Project Motivation:**  
This project aims to **automate the generation of preliminary radiology reports** from chest X-ray images by leveraging advanced computer vision techniques and large language models. This system serves as an aid for radiologists by:
- Enhancing productivity
- Reducing delays
- Minimizing errors due to workload fatigue

The report covers:
- An overview of the dataset structure and features
- Detailed methodology and preprocessing steps
- Model design, training, evaluation, and optimization techniques
- Performance metrics and analysis
- Potential further improvements

---

## Dataset Description

The project uses the **MIMIC-CXR** dataset, which includes:
- **15,000 chest X-ray images** (originally in DICOM format, converted to PNG)
- Associated radiology reports in XML format

**Key Dataset Features:**
- **Image File Path:** Location/link of the corresponding chest X-ray image.
- **Findings:** Textual descriptions of abnormalities or observations.
- **Impression:** A concise summary of the primary conclusions.

**Pathology Labels (14 Total):**
- Atelectasis
- Cardiomegaly
- Consolidation
- Edema
- Enlarged Cardiomediastinum
- Fracture
- Lung Lesion
- Lung Opacity
- Pleural Effusion
- Pleural Other
- Pneumonia
- Pneumothorax
- Support Devices
- No Finding

---

## Methodology

The project is structured into several key stages:

### 1. Data Collection and Preprocessing

#### a. Data Extraction
- **DICOM to PNG Conversion:**  
  A custom script converts the original DICOM images to PNG format, reducing file size while preserving image quality for efficient loading and processing.

- **CSV Creation:**  
  A dedicated script extracts the following fields:
  - `image_ID`: Unique identifier for each image.
  - `image_path`: Consolidated file paths to each PNG image.
  - `findings` and `impressions`: Parsed from XML reports.

#### b. Data Pre-processing
- **Text Cleaning:**  
  - Expanding abbreviations (e.g., "lat" → "lateral")
  - Removing special characters
  - Fixing spacing around punctuation

- **Filtering and Label Mapping:**  
  Invalid or missing entries are removed, and findings are mapped to a list of specific disease labels.

- **Image Augmentation:**  
  Applied techniques include:
  - Resizing to (224, 224)
  - Random rotations and flips
  - Noise addition
  - Normalization

#### c. Dataset Split
A custom function `get_dataloaders` creates PyTorch DataLoader objects for training and validation with parameters:
- **Batch Size:** Default is 8.
- **Train Split:** 85% training, 15% validation.
- **Num Workers:** Default is 4 for faster loading.
- **Collate Function:** Custom function to merge samples, particularly for variable-length inputs like text.

---

### 2. Extracting Labels Using CheXbert

**CheXbert** is a transformer-based model fine-tuned for medical text classification using the BERT architecture. It extracts multi-label classifications from chest X-ray radiology reports.

#### Process:
1. **Text Processing:**  
   - Extract "Findings" and "Impressions" from reports.
   - Tokenize and format text for CheXbert.
   - Generate high-dimensional contextual embeddings.

2. **Label Extraction:**  
   - A classification layer predicts probabilities for each clinical condition.
   - Probabilities are thresholded at **0.5** to produce binary labels.

3. **Dataset Preparation:**  
   The binary labels are integrated into a CSV file to enrich the dataset for multi-label classification.

![CheXbert Workflow](https://github.com/user-attachments/assets/29b4921c-d5e8-431d-ba86-8b73ca16b8b6)

---

### 3. ChexNet for Structural Findings Extraction

**ChexNet** (based on DenseNet-121) is fine-tuned for multi-label classification of chest X-rays, focusing on structural abnormalities.

#### Key Points:
- **Base Model:** DenseNet-121 with pre-trained ImageNet weights.
- **Layer Freezing:**  
  Initial layers are frozen; only the last two dense blocks and the classifier head are fine-tuned.
- **Custom Classifier:**  
  - **Input:** 1024 features from DenseNet-121.
  - **Hidden Layer:** 512 units with ReLU activation.
  - **Dropout:** 0.3 for regularization.
  - **Output:** 14 sigmoid-activated nodes for multi-label classification.
- **Training Procedure:**  
  - **Loss Function:** Custom Weighted Binary Cross-Entropy Loss (WeightedBCELoss)
  - **Optimizer:** Adam with differential learning rates.
  - **Scheduler:** ReduceLROnPlateau.
  - **Metric:** Achieved an F1-micro score of **0.70**.

![ChexNet Workflow](https://github.com/user-attachments/assets/eaf445e8-5696-43d0-998b-4905b36507e6)

---

### 4. Model Architectures

Two distinct model architectures were experimented with to generate medical reports:

#### Model 1: BioVilt + Alignment + BioGPT

1. **Components:**
   - **BioVilt:**  
     - Uses a ResNet backbone (ResNet-50/ResNet-18) for feature extraction.
     - Produces a 512-dimensional global embedding.
   - **Alignment Module:**  
     - Bridges image embeddings with textual representations.
   - **BioGPT:**  
     - A powerful GPT-2 based language model pre-trained on biomedical literature (approx. 347M parameters).

2. **Configuration:**
   - **BioVilt:**  
     - Backbone: ResNet-50  
     - Output: 512-dimensional embedding.
   - **Alignment Module:**  
     - Text encoder: Microsoft BioGPT.
     - Projection layers map image embeddings to BioGPT’s 768-dimensional space.
     - **Loss Function:** Contrastive Loss.
   - **BioGPT (PEFT via LoRA):**  
     - **Rank:** 16  
     - **Alpha:** 32  
     - **Dropout:** 0.1  
   - **Generation Parameters:**  
     - `max_length`: 150 tokens  
     - `temperature`: 0.8  
     - `top_k`: 50  
     - `top_p`: 0.85  

3. **Integration and Flow:**
   - **Image Preprocessing:** Resize and augment PNG images.
   - **Image Encoding:** BioVilt extracts image features.
   - **Alignment:** Projects image embeddings to align with BioGPT's text embeddings.
   - **Report Generation:** The aligned embeddings are fed into BioGPT to generate the final report.

---

#### Model 2: BioVilt + ChexNet + Alignment + BioGPT

![Model 2 Overview](https://github.com/user-attachments/assets/f7da053d-97ec-43d3-b66c-c976ecd269ed)

1. **Components:**
   - **BioVilt:**  
     - ResNet-50 based image encoder.
   - **ChexNet:**  
     - Multi-label classifier (DenseNet-121) for structural findings.
   - **Alignment Module:**  
     - Integrates image and label embeddings with text embeddings.
   - **BioGPT:**  
     - Fine-tuned for biomedical report generation.

2. **Configuration:**
   - **BioVilt:**  
     - Backbone: ResNet-50  
     - Output: 512-dimensional embedding.
   - **ChexNet:**  
     - Backbone: DenseNet-121  
     - Output: Multi-label predictions for 14 clinical findings.
   - **Alignment Module:**  
     - Text encoder: Microsoft BioGPT.
     - Projection layers map image embeddings to 768 dimensions and separately project text from the ground truth reports.
     - **Loss Function:** Contrastive Loss.
   - **BioGPT (PEFT via LoRA):**  
     - **Rank:** 16  
     - **Alpha:** 32  
     - **Dropout:** 0.1  
   - **Generation Parameters:**  
     - `max_length`: 150 tokens  
     - `temperature`: 0.8  
     - `top_k`: 50  
     - `top_p`: 0.85  

3. **Integration and Flow:**
   - **Image Preprocessing:** Resize and augment PNG images.
   - **Image Encoding:** BioVilt extracts image features.
   - **ChexNet Classification:** Identifies structural findings and generates binary labels.
   - **Alignment:** Combines image embeddings with label information and projects them to align with BioGPT’s text embeddings.
   - **Concatenation:** The image embeddings and prompt text embeddings (with a `<SEP>` token separator) are concatenated.
   - **Report Generation:** The concatenated embeddings are fed into BioGPT to generate the final report.

---

## Results

In this analysis, a comprehensive comparison is conducted between the two distinct models. The **ROUGE** metric (Recall Oriented Understudy for Gisting Evaluation) is used as the primary evaluation metric, measuring the overlap between generated and reference text across several dimensions such as recall, precision, and F1-score.

**ROUGE-L (Longest Common Subsequence):**  
This metric evaluates the longest common subsequence between the generated and reference texts, giving credit for correctly ordered content even if the content is spread out.

Graph snippets for **(BioGPT + Image Encoder)** and **(BioGPT + Image Encoder + ChexNet Labels)** are provided below:

![image](https://github.com/user-attachments/assets/7db55f12-ca80-4f3d-8d9c-7ac39579754e)

- **Model 1: BioVilt + Alignment + BioGPT**

  ![Model 1 Results](https://github.com/user-attachments/assets/a45cb640-50bc-4556-89f6-06e068e8a24a)

- **Model 2: BioVilt + ChexNet + Alignment + BioGPT**

  ![Model 2 Results](https://github.com/user-attachments/assets/598b4263-2dc2-4620-9587-648e3701a79b)

---

## Challenges Faced

- **Limited Computation Power:**  
  Resource constraints affected training and model size selection.

- **Model Complexity:**  
  Smaller models failed to capture detailed findings, while larger models were required for improved accuracy.

- **Error Propagation:**  
  The clinical findings extraction model introduces some errors that can impact the final report quality.

---

## Deployment

The model is deployed using **Streamlit** on an **AWS EC2** instance for real-time inference.

---

## References

- **CheXbert:** [CheXbert GitHub Repository](https://github.com/stanfordmlgroup/CheXbert)
- **ChexNet:** [ChexNet: Radiologist-Level Pneumonia Detection on Chest X-Rays (arXiv)](https://arxiv.org/abs/1711.05225)
- **BioVilt:** [BioViLT: A Vision-Language Transformer for Medical Image Report Generation (arXiv)](https://arxiv.org/abs/2206.09993)
- **BioGPT:** [BioGPT BioGPT: Generative Pre-trained Transformer for Biomedical Text Generation and Mining](https://arxiv.org/abs/2210.10341)
- **PEFT Techniques (LoRA):** [LoRA: Low-Rank Adaptation for Fast Training of Neural Networks (arXiv)](https://arxiv.org/abs/2106.09685)

---

*This project demonstrates a synergistic approach combining computer vision and natural language processing to assist radiologists by generating detailed preliminary reports from chest X-ray images.*

Feel free to explore the repository for code, experiments, and further documentation.
