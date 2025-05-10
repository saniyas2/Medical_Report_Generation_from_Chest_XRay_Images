import os
import sys
import argparse
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

# Local imports
import utils
from models.bert_labeler import bert_labeler
from bert_tokenizer import tokenize
from transformers import BertTokenizer
from datasets.unlabeled_dataset import UnlabeledDataset
from constants import *


def collate_fn_no_labels(sample_list):
    tensor_list = [s['imp'] for s in sample_list]
    batched_imp = torch.nn.utils.rnn.pad_sequence(
        tensor_list, batch_first=True, padding_value=PAD_IDX
    )
    len_list = [s['len'] for s in sample_list]
    batch = {'imp': batched_imp, 'len': len_list}
    return batch


def load_unlabeled_data(csv_path, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False):
    collate_fn = collate_fn_no_labels
    dset = UnlabeledDataset(csv_path)
    loader = torch.utils.data.DataLoader(
        dset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_fn
    )
    return loader


def label(checkpoint_path, csv_path):
    ld = load_unlabeled_data(csv_path)
    
    model = bert_labeler()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 0: 
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model) 
        model = model.to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            name = k[7:] 
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        
    was_training = model.training
    model.eval()
    y_pred = [[] for _ in range(len(CONDITIONS))]

    print("\nBegin report impression labeling. The progress bar counts the # of batches completed:")
    print("The batch size is %d" % BATCH_SIZE)
    with torch.no_grad():
        for i, data in enumerate(tqdm(ld)):
            batch = data['imp']
            batch = batch.to(device)
            src_len = data['len']
            batch_size = batch.shape[0]
            attn_mask = utils.generate_attention_masks(batch, src_len, device)

            out = model(batch, attn_mask)

            for j in range(len(out)):
                curr_y_pred = out[j].argmax(dim=1)
                y_pred[j].append(curr_y_pred)

        for j in range(len(y_pred)):
            y_pred[j] = torch.cat(y_pred[j], dim=0)
             
    if was_training:
        model.train()

    y_pred = [t.tolist() for t in y_pred]
    return y_pred


def save_preds(y_pred, csv_path, out_path):
    y_pred = np.array(y_pred)
    y_pred = y_pred.T

    # Read original dataset to include image_id
    original_dataset = pd.read_csv(csv_path)

    # Ensure the 'image_id' column exists
    if 'image_id' not in original_dataset.columns:
        raise ValueError("The input CSV must have an 'image_id' column.")

    # Extract image IDs and report impressions
    image_ids = original_dataset['image_id']
    reports = original_dataset['Report Impression']

    # Create DataFrame with predictions
    df = pd.DataFrame(y_pred, columns=CONDITIONS)
    df['image_id'] = image_ids
    df['Report Impression'] = reports

    # Reorder columns
    new_cols = ['image_id', 'Report Impression'] + CONDITIONS
    df = df[new_cols]

    # Replace classes with their appropriate values
    df.replace(0, np.nan, inplace=True)  # Blank class is NaN
    df.replace(3, -1, inplace=True)      # Uncertain class is -1
    df.replace(2, 0, inplace=True)       # Negative class is 0 

    # Save output CSV
    output_file = os.path.join(out_path, 'labeled_reports_with_images.csv')
    df.to_csv(output_file, index=False)
    print(f"Labeled reports saved with image file names: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label a csv file containing radiology reports')
    parser.add_argument('-d', '--data', type=str, nargs='?', required=True,
                        help='path to csv containing reports. The reports should be \
                              under the \"Report Impression\" column')
    parser.add_argument('-o', '--output_dir', type=str, nargs='?', required=True,
                        help='path to intended output folder')
    parser.add_argument('-c', '--checkpoint', type=str, nargs='?', required=True,
                        help='path to the pytorch checkpoint')
    args = parser.parse_args()
    csv_path = args.data
    out_path = args.output_dir
    checkpoint_path = args.checkpoint

    y_pred = label(checkpoint_path, csv_path)
    save_preds(y_pred, csv_path, out_path)


# Sample Usage : # You will need to download chexbert checkpoint : "https://stanfordmedicine.app.box.com/s/c3stck6w6dol3h36grdc97xoydzxd7w9" (Dowload Chexbert.pth)

"""
python label.py -d="C:\Anand\Projects_GWU\NLP_Project\FinalProject-Group6\Data\final_cleaned.csv" -o="C:\Anand\Projects_GWU\NLP_Project\FinalProject-Group6\Results" -c="C:\Users\anand\Downloads\chexbert.pth"
"""