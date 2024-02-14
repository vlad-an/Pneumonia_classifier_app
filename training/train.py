import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import mlflow
from utils.data_loader import load_data
import sys
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau


from models.lenet import LeNet
from models.alexnet import AlexNet
from models.skipconnect import SkipConnectionCNN
from models.compactnet import CompactCNN

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on Chest X-Ray Images")
    parser.add_argument('--model', type=str, required=True,
                        choices=['LeNet', 'AlexNet', 'SkipConnectionCNN', 'CompactCNN'],
                        help='Model to train')
    args = parser.parse_args()
    return args


def train_model(model_name, model, train_loader, valid_loader, criterion, optimizer, learning_rate, num_epochs=10, device="cpu"):
    # Start MLFlow run
    with mlflow.start_run():
        mlflow.set_tag("model", model_name)
        # Set a custom run name
        mlflow.set_tag("mlflow.runName", f"{model_name}_{learning_rate}")
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("optimizer", type(optimizer).__name__)
        mlflow.log_param("loss_function", type(criterion).__name__)
        mlflow.log_param("model", model_name)
        mlflow.log_param("learning_rate", learning_rate)

    
        model.train()
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        best_val_loss = float('inf')
        best_epoch = -1
        epochs_no_improve = 0

        # Object to store best model state
        best_model_state = None

        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_loss = running_loss / len(train_loader)
            train_accuracy = 100 * correct / total

            # Validation phase
            model.eval()
            val_loss, val_accuracy = validate(model, valid_loader, criterion, device)

            # Early stopping and Learning rate scheduling
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                # Save the best model state
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == 4:
                    print("Early stopping triggered")
                    break  # Stop training

            scheduler.step(val_loss)
            lr_after = optimizer.param_groups[0]['lr']
            lr_before = scheduler.get_last_lr()[0]

            if lr_after < lr_before:
                print(f"Learning rate adjusted from {lr_before} to {lr_after}")


            # Log metrics for each epoch to MLFlow
            mlflow.log_metrics({"train_loss": train_loss, "train_accuracy": train_accuracy,
                                "val_loss": val_loss, "val_accuracy": val_accuracy}, step=epoch)

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        print(f"Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
        
        # After training, save the best model
        model_save_path = f'models/checkpoints/{model_name}_best_model.pth'
        torch.save(best_model_state, model_save_path)
        print(f"Best model saved to {model_save_path}")

        # Log the best model with MLflow
        mlflow.pytorch.log_model(model, f"models/{model_name}_best", registered_model_name=f"{model_name}_best")

        # Log best metrics
        mlflow.log_metrics({"best_val_loss": best_val_loss, "best_epoch": best_epoch + 1})

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def main():
    args = parse_args()  # Parse command line arguments

    # Initialize MLFlow
    mlflow.set_experiment("ChestXRayPneumoniaClassification")
    
    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    data_dir = '/Users/vladandreichuk/Desktop/ChestXRayPneumoniaClassification/training/data'  
    train_loader, valid_loader = load_data(data_dir)

    # Model selection based on command line argument
    if args.model == 'LeNet':
        model = LeNet().to(device)
    elif args.model == 'AlexNet':
        model = AlexNet().to(device)
    elif args.model == 'SkipConnectionCNN':
        model = SkipConnectionCNN().to(device)
    elif args.model == 'CompactCNN':
        model = CompactCNN().to(device)
  
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(args.model ,model, train_loader, valid_loader, criterion, optimizer, learning_rate,num_epochs=10,device=device)

if __name__ == "__main__":
    main()
