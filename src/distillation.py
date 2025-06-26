# src/distillation.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .models import build_student_model
from .utils import evaluate_model
import copy
import os

def distill_model(
    teacher_model: nn.Module,
    farm_id: str,
    num_farm_classes: int,
    distill_train_loader: torch.utils.data.DataLoader,
    distill_val_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    temperature: float,
    alpha: float,
    device: torch.device,
    output_dir: str
):
    """
    Trains a student model using knowledge distillation.
    """
    print(f"  [Distillation Farm {farm_id}] Starting process...")

    # 1. Build the student model
    student_model = build_student_model(num_classes=num_farm_classes).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    teacher_model.eval()
    student_model.train()

    best_student_accuracy = 0.0
    best_student_model_state = None

    for epoch in range(epochs):
        running_loss_kd = 0.0
        running_loss_ce = 0.0
        running_loss_total = 0.0
        processed_samples = 0

        for inputs, hard_labels in distill_train_loader:
            inputs, hard_labels = inputs.to(device), hard_labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            student_outputs = student_model(inputs)

            loss_kd = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs / temperature, dim=1)
            ) * (temperature * temperature)

            loss_ce = nn.CrossEntropyLoss()(student_outputs, hard_labels)
            total_loss = alpha * loss_kd + (1 - alpha) * loss_ce

            total_loss.backward()
            optimizer.step()

            running_loss_kd += loss_kd.item() * inputs.size(0)
            running_loss_ce += loss_ce.item() * inputs.size(0)
            running_loss_total += total_loss.item() * inputs.size(0)
            processed_samples += inputs.size(0)
        
        if processed_samples == 0: continue

        avg_loss_total = running_loss_total / processed_samples
        print(f"    Epoch {epoch+1}/{epochs} | Avg Distill Loss: {avg_loss_total:.4f}")

        # --- vvvvvvvvvvvv MODIFICATION START vvvvvvvvvvvv ---
        if distill_val_loader and len(distill_val_loader.dataset) > 0:
            # Assign the returned dictionary to a single variable 'metrics'
            metrics = evaluate_model(student_model, distill_val_loader, device, context=f"Distill Eval Farm {farm_id}")
            
            # Access values by key
            student_val_loss = metrics['loss']
            student_val_accuracy = metrics['accuracy']
            
            print(f"      Student Model Val: Loss = {student_val_loss:.4f}, Accuracy = {student_val_accuracy:.2f}%")
            
            if student_val_accuracy > best_student_accuracy:
                best_student_accuracy = student_val_accuracy
                best_student_model_state = copy.deepcopy(student_model.state_dict())
                print(f"      New best student model for Farm {farm_id}! Accuracy: {best_student_accuracy:.2f}%")
        else:
             best_student_model_state = copy.deepcopy(student_model.state_dict())
        # --- ^^^^^^^^^^^^^^ MODIFICATION END ^^^^^^^^^^^^^^ ---


    if best_student_model_state:
        student_model.load_state_dict(best_student_model_state)
        print(f"  [Distillation Farm {farm_id}] Final student model loaded with best accuracy: {best_student_accuracy:.2f}%")

        student_model_save_path = os.path.join(output_dir, f'student_model_farm_{farm_id}.pth')
        torch.save(student_model.state_dict(), student_model_save_path)
        print(f"  Student model for Farm {farm_id} saved to: {student_model_save_path}")
    else:
        print(f"  Warning: Could not produce a valid student model for Farm {farm_id}.")

    return student_model