import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import mlflow
from utils.data_loader import load_data
import sys
import argparse
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from models.lenet import LeNet
from models.alexnet import AlexNet
from models.skipconnect import SkipConnectionCNN
from models.compactnet import CompactCNN


def parse_args():
    """
    Parses command line arguments for training a model on Chest X-Ray Images.

    This function defines and parses command-line arguments necessary for specifying the model architecture
    to be trained on chest X-Ray images. It supports a selection of predefined models, ensuring that the user
    selects a valid model architecture for training. The function is built using the argparse library.

    Args:
        None. Arguments are provided via the command line. The following argument is expected:
        --model (str): Mandatory. Specifies the model to train. The available options are 'LeNet', 'AlexNet',
                       'SkipConnectionCNN', and 'CompactCNN'. The choice determines the architecture of the neural
                       network to be used for training on the dataset.

    Returns:
        argparse.Namespace: An object containing the parsed command line arguments. Specifically, it includes
                            a 'model' attribute, which holds the name of the selected model architecture as a string.
    """
    parser = argparse.ArgumentParser(description="Train a model on Chest X-Ray Images")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["LeNet", "AlexNet", "SkipConnectionCNN", "CompactCNN"],
        help="Model to train",
    )
    args = parser.parse_args()
    return args


def train_model(
    model_name,
    model,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    learning_rate,
    num_epochs=10,
    device="cpu",
):
    """
    Trains a specified model on a dataset using provided training and validation data loaders, and logs the training
    process with MLFlow.

    This function handles the training process of a machine learning model, including logging parameters, metrics,
    and the best model state using MLFlow. It supports early stopping and learning rate adjustment based on validation
    loss performance.

    Args:
        model_name (str): The name of the model being trained.
        model (torch.nn.Module): The model to train.
        train_loader (DataLoader): DataLoader for the training dataset.
        valid_loader (DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): Loss function to use for training.
        optimizer (torch.optim.Optimizer): Optimizer to use for training.
        learning_rate (float): Initial learning rate for the optimizer.
        num_epochs (int, optional): Number of epochs to train for. Defaults to 10.
        device (str, optional): Device to train on ('cpu' or 'cuda'). Defaults to "cpu".

    Returns:
        None. The function logs training and validation metrics to MLFlow, saves the best model state, and optionally
        logs the model with MLFlow. It does not return any values.
    """
    # Start MLFlow run
    with mlflow.start_run():
        # Log tags
        mlflow.set_tag("model", model_name)
        mlflow.set_tag("mlflow.runName", f"{model_name}_{learning_rate}")

        # Log parameters
        mlflow.log_param("epochs", num_epochs)
        mlflow.log_param("optimizer", type(optimizer).__name__)
        mlflow.log_param("loss_function", type(criterion).__name__)
        mlflow.log_param("model", model_name)
        mlflow.log_param("learning_rate", learning_rate)

        # Training mode
        model.train()
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3, verbose=True
        )

        best_val_loss = float("inf")
        best_val_accuracy = 0  # Initialize best validation accuracy
        best_epoch_loss = -1
        best_epoch_accuracy = -1  # Initialize epoch for best validation accuracy
        best_val_f1 = 0  # Initialize best validation F1-Score
        best_val_roc_auc = 0  # Initialize best validation ROC-AUC
        best_epoch_f1 = -1  # Initialize epoch for best validation F1-Score
        best_epoch_roc_auc = -1  # Initialize epoch for best validation ROC-AUC
        epochs_no_improve = 0

        # Object to store best model state
        best_model_state = None

        # Training phase
        for epoch in range(num_epochs):
            running_loss = 0.0
            correct = 0
            total = 0
            train_preds, train_targets = [], []


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
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.view(-1).cpu().numpy())
            train_targets.extend(labels.view(-1).cpu().numpy())

            # Calculate training metrics
            train_f1 = f1_score(train_targets, train_preds)
            train_roc_auc = roc_auc_score(train_targets, train_preds)

            # Validation phase with modifications to collect predictions and true labels
            model.eval()
            val_loss, val_accuracy, val_preds, val_targets = validate(model, valid_loader, criterion, device)
            val_f1 = f1_score(val_targets, val_preds)
            val_roc_auc = roc_auc_score(val_targets, val_preds)

            # Early stopping and Learning rate scheduling
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                # Save the best model state
                best_model_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                

            # Check if current validation accuracy is the best
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_epoch_accuracy = epoch  # Update epoch for best accuracy

            # Update best validation F1-Score and ROC-AUC score if current values are better
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch_f1 = epoch

            if val_roc_auc > best_val_roc_auc:
                best_val_roc_auc = val_roc_auc
                best_epoch_roc_auc = epoch

            
            if epochs_no_improve == 4:
                    print("Early stopping triggered")
                    break  # Stop training


            scheduler.step(val_loss)
            lr_after = optimizer.param_groups[0]["lr"]
            lr_before = scheduler.get_last_lr()[0]

            if lr_after < lr_before:
                print(f"Learning rate adjusted from {lr_before} to {lr_after}")

            # Log metrics for each epoch to MLFlow, including F1-Score and ROC-AUC
            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "train_f1": train_f1,
                    "train_roc_auc": train_roc_auc,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                    "val_f1": val_f1,
                    "val_roc_auc": val_roc_auc,
                },
                step=epoch,
            )


            print(
                f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.2f}%, "
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%"
            )

        print(f"Best Validation Loss: {best_val_loss:.4f} at epoch {best_epoch+1}")
        print(f"Best Validation Accuracy: {best_val_accuracy:.2f}% at epoch {best_epoch_accuracy+1}")

        # After training, save the best model
        model_save_path = f"models/checkpoints/{model_name}_best_model.pth"
        torch.save(best_model_state, model_save_path)
        print(f"Best model saved to {model_save_path}")

        # Log the best model with MLflow
        mlflow.pytorch.log_model(
            model,
            f"models/{model_name}_best",
            registered_model_name=f"{model_name}_best",
        )

        # Log the best metrics including best validation accuracy and its epoch
        mlflow.log_metrics(
            {
                "best_val_loss": best_val_loss,
                "best_epoch_loss": best_epoch_loss + 1, # +1 to convert from 0-based to 1-based indexing
                "best_val_accuracy": best_val_accuracy,
                "best_epoch_accuracy": best_epoch_accuracy + 1,
                "best_val_f1": best_val_f1,
                "best_epoch_f1": best_epoch_f1 + 1,  
                "best_val_roc_auc": best_val_roc_auc,
                "best_epoch_roc_auc": best_epoch_roc_auc + 1,
            }
        )


def validate(model, loader, criterion, device):
    """
    Validates a trained model on a given dataset loader and computes the loss and accuracy.

    This function evaluates the model in inference mode (model.eval()) on the provided dataset using the specified
    device. It calculates the average loss and accuracy over the dataset. The function does not modify the model
    parameters.

    Args:
        model (torch.nn.Module): The trained model to validate.
        loader (torch.utils.data.DataLoader): DataLoader for the dataset to validate against.
        criterion (torch.nn.Module): The loss function used for validation.
        device (str): The device ('cpu' or 'cuda') to perform validation computations on.

    Returns:
        tuple: A tuple containing two elements:
            - avg_loss (float): The average loss of the model on the validation dataset.
            - accuracy (float): The accuracy of the model on the validation dataset, as a percentage.

    The function iterates through the dataset provided by the loader, computes the loss for each batch, and tracks
    the number of correctly predicted samples to calculate the overall accuracy. It uses `torch.no_grad()` to disable
    gradient computation, reducing memory consumption and speeding up the process.
    """
    # Set model to evaluation mode
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_targets = [], []

    # Disable gradient computation for efficiency
    with torch.no_grad():
        for images, labels in loader:
            # Move data to the specified device
            images, labels = images.to(device), labels.to(device)
            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # Calculate the batch loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            # Convert output probabilities to predicted class
            _, predicted = torch.max(outputs.data, 1)
            # Count total and correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_targets.extend(labels.view(-1).cpu().numpy())


    # Calculate average loss and accuracy
    avg_loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, all_preds, all_targets


def main():
    """
    Main function to train a model on Chest X-Ray images for pneumonia classification.

    This function orchestrates the process of model training for pneumonia detection using Chest X-Ray images. It
    starts by parsing command line arguments for model selection, initializes MLFlow for experiment tracking, prepares
    the training and validation datasets, selects the model architecture based on user input, and finally trains the
    model. The process includes setting up the loss function, optimizer, and executing the training loop with the
    specified number of epochs and learning rate.

    Args:
        None. The function retrieves its arguments from the command line input specifying the model to train.

    Returns:
        None. The function does not return a value but instead trains the model, logs the training process with MLFlow,
        and saves the trained model.
    """
    args = parse_args()  # Parse command line arguments

    # Initialize MLFlow
    mlflow.set_experiment("ChestXRayPneumoniaClassification")

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load data
    data_dir = (
        "/Users/vladandreichuk/Desktop/git_reps/ChestXRayPneumoniaClassification/training/data"
    )
    train_loader, valid_loader = load_data(data_dir)

    # Model selection based on command line argument
    if args.model == "LeNet":
        model = LeNet().to(device)
    elif args.model == "AlexNet":
        model = AlexNet().to(device)
    elif args.model == "SkipConnectionCNN":
        model = SkipConnectionCNN().to(device)
    elif args.model == "CompactCNN":
        model = CompactCNN().to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    train_model(
        args.model,
        model,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        learning_rate,
        num_epochs=10,
        device=device,
    )


if __name__ == "__main__":
    main()
