import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from chexnet_train import ModifiedCheXNet  # Import your model class
import argparse


class ChestXrayPredictor:
    def __init__(self, model_path, device=None):
        """
        Initialize the predictor with a trained model
        Args:
            model_path: Path to the trained model checkpoint
            device: torch.device to use for inference
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        self.labels = [
            'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity',
            'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
            'Atelectasis', 'Pneumothorax', 'Pleural Effusion',
            'Pleural Other', 'Fracture', 'Support Devices', 'No Finding'
        ]

    def _load_model(self, model_path):
        """Load the trained model"""
        model = ModifiedCheXNet(num_classes=14).to(self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

        model.eval()
        return model

    def _get_transforms(self):
        """Get the transformation pipeline"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path, threshold=0.5):
        """
        Predict probabilities for a single image
        Args:
            image_path: Path to the input image
            threshold: Probability threshold for positive prediction
        Returns:
            Dictionary of predictions with probabilities and binary predictions
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Get predictions
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = output.cpu().numpy()[0]

        # Create prediction dictionary
        predictions = {
            'probabilities': {},
            'binary_predictions': {},
            'findings': []
        }

        # Process predictions
        for label, prob in zip(self.labels, probabilities):
            predictions['probabilities'][label] = float(prob)
            predictions['binary_predictions'][label] = bool(prob >= threshold)
            if prob >= threshold:
                predictions['findings'].append(label)

        return predictions

    def batch_predict(self, image_paths, threshold=0.5):
        """
        Predict probabilities for multiple images
        Args:
            image_paths: List of paths to input images
            threshold: Probability threshold for positive prediction
        Returns:
            List of prediction dictionaries
        """
        predictions = []
        for image_path in image_paths:
            pred = self.predict(image_path, threshold)
            predictions.append({
                'image_path': image_path,
                'predictions': pred
            })
        return predictions


def main():
    parser = argparse.ArgumentParser(description='ChexNet Inference')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to the input image or directory of images')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Probability threshold for positive prediction')
    parser.add_argument('--batch', action='store_true',
                        help='Process multiple images from a directory')
    args = parser.parse_args()

    # Initialize predictor
    predictor = ChestXrayPredictor(args.model_path)

    # Process single image or batch
    if args.batch:
        import os
        image_paths = [os.path.join(args.image_path, f) for f in os.listdir(args.image_path)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        predictions = predictor.batch_predict(image_paths, args.threshold)

        # Print batch results
        for pred in predictions:
            print(f"\nResults for {pred['image_path']}:")
            findings = pred['predictions']['findings']
            if findings:
                print("Detected conditions:")
                for finding in findings:
                    prob = pred['predictions']['probabilities'][finding]
                    print(f"- {finding}: {prob:.3f}")
            else:
                print("No significant findings detected")
    else:
        # Process single image
        predictions = predictor.predict(args.image_path, args.threshold)

        print("\nPrediction Results:")
        if predictions['findings']:
            print("Detected conditions:")
            for finding in predictions['findings']:
                prob = predictions['probabilities'][finding]
                print(f"- {finding}: {prob:.3f}")
        else:
            print("No significant findings detected")


if __name__ == '__main__':
    main()