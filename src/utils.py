import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
import numpy as np

def evaluate_model(model, dataloader, device, context="Evaluation"):
    """
    Evaluates the model's performance on a given dataset, returning loss, accuracy, and F1-score.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_targets = []
    running_loss = 0.0
    
    if not dataloader or len(dataloader.dataset) == 0:
        # print(f"[{context}] DataLoader is empty, skipping evaluation.")
        return {"loss": 0.0, "accuracy": 0.0, "f1_score": 0.0}

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    total_samples = len(all_targets)
    if total_samples == 0:
        # print(f"[{context}] No samples in validation set, cannot evaluate.")
        return {"loss": 0.0, "accuracy": 0.0, "f1_score": 0.0}

    avg_loss = running_loss / total_samples
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_targets)) / total_samples
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return {"loss": avg_loss, "accuracy": accuracy, "f1_score": f1}


def plot_server_fl_history(history, output_dir):
    """
    Plots and saves the federated learning history curves for the server model.
    """
    if not history['round']:
        print("No server aggregation history to plot.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # --- English Labels ---
    ax1.plot(history['round'], history['accuracy'], marker='o', label='Accuracy')
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Server Global Model Performance on Unified Validation Set")
    ax1.grid(True)
    
    ax2.plot(history['round'], history['f1_score'], marker='o', color='g', label='F1-Score (Macro)')
    ax2.set_ylabel("F1 Score (Macro)")
    ax2.grid(True)

    ax3.plot(history['round'], history['loss'], marker='o', color='r', label='Loss')
    ax3.set_ylabel("Loss")
    ax3.set_xlabel("Server Communication Round")
    ax3.grid(True)
    
    fig.tight_layout()
    plot_save_path = os.path.join(output_dir, 'server_federated_learning_plot.png')
    plt.savefig(plot_save_path)
    print(f"Server FL training plot saved to: {plot_save_path}")
    plt.close()