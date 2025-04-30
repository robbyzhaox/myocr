import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from myocr.utils import setup_plots, update_plots


class Evaluator:
    def __init__(self):
        pass


class Trainer:
    def __init__(self, model, evaluators, loss_fn, optimizer, num_epochs, batch_size):
        self.model = model
        self.device = model.device

        self.evaluators = evaluators

        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def dataloader(self, dataset) -> DataLoader:
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def fit(self, trainingDataset, validateDataset):
        device = self.device
        # train
        train_loader = self.dataloader(trainingDataset)
        val_loader = self.dataloader(validateDataset)
        self.model.train()
        train_losses = []

        val_losses = []
        val_accuracies = []
        fig, ax1, ax2, train_line, val_line = setup_plots()

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss = 0.0
            for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.num_epochs}"):
                data, labels = data.to(device), labels.to(device)

                # forward
                outputs = self.model(data)
                loss = self.loss_fn(outputs, labels)

                # backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running_loss += loss.item()

            train_loss = running_loss / len(train_loader)
            train_losses.append(train_loss)

            # validation
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                correct = 0
                total = 0
                for data, labels in val_loader:
                    data, labels = data.to(device), labels.to(device)
                    outputs = self.model(data)
                    loss = self.loss_fn(outputs, labels)
                    val_loss += loss.item()

                    predicted = torch.argmax(outputs.data, dim=1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                val_loss = val_loss / len(val_loader)
                val_losses.append(val_loss)

                val_accuracy = 100 * correct / total
                val_accuracies.append(val_accuracy)
                update_plots(fig, ax1, ax2, train_line, val_line, train_losses, val_losses, epoch)

                # print(
                #     f"""Epoch {epoch+1}/{trainer.num_epochs}:
                #             Train Loss: {train_loss:.4f},
                #             Val Loss: {val_loss:.4f},
                #             Val Acc: {val_accuracy:.2f}%"""
                # )
