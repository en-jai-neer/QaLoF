import os
import torch
from torch.nn.functional import softmax
from tqdm import tqdm

def compute_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    labels = torch.tensor(labels)
    correct = (preds == labels).sum().item()
    return correct / len(labels)

def evaluate_from_pt_files(folder_path):
    all_clean_logits = []
    all_adv_logits = []
    all_labels = []

    # Iterate over .pt files in the folder
    for file_name in tqdm(os.listdir(folder_path)):
        if file_name.endswith('.pth'):
            data = torch.load(os.path.join(folder_path, file_name), map_location='cpu')
            print(data.keys())
            all_clean_logits.append(data['clean_logits'])
            all_adv_logits.append(data['adv_logits'])
            all_labels.extend(data['labels'])

    # Concatenate all logits
    clean_logits = torch.cat(all_clean_logits, dim=0)
    adv_logits = torch.cat(all_adv_logits, dim=0)

    # Compute accuracies
    clean_acc = compute_accuracy(clean_logits, all_labels)
    adv_acc = compute_accuracy(adv_logits, all_labels)

    print(f"Clean Accuracy: {clean_acc * 100:.2f}%")
    print(f"Adversarial Accuracy: {adv_acc * 100:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--adv_samples_folder', type=str, required=True,
                        help='Folder containing .pt files with adv and clean logits')
    args = parser.parse_args()

    evaluate_from_pt_files(args.adv_samples_folder)
