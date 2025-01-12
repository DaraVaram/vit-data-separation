import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from typing import Type, List

from analyze import FuzzinessAnalyzer
from logger import ExperimentLogger
import numpy as np

class ModelHandler:
    def __init__(self, model_class: Type[torch.nn.Module], model_params: dict, device: torch.device,
                 criterion_class: Type[torch.nn.Module], optimizer_class: Type[Optimizer], optimizer_params: dict,
                 learning_rate: float = 1e-3, num_epochs: int = 10, log_dir: str = 'logs', num_epochs_per_analysis: int = 10):
        self.model = model_class(**model_params).to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.criterion = criterion_class()
        self.optimizer = optimizer_class(self.model.parameters(), lr=learning_rate, **optimizer_params)
        self.fuzziness_analyzer = FuzzinessAnalyzer(self.device)
        self.logger = ExperimentLogger(log_dir)
        self.num_epochs_per_analysis = num_epochs_per_analysis

    def train(self, train_loader: DataLoader, test_loader: DataLoader):
        self.model.train()
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct, total = 0, 0
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
            
            train_accuracy = 100 * correct / total

            test_correct, test_total = 0, 0
            with torch.no_grad():
                for test_inputs, test_targets in test_loader:
                    test_inputs, test_targets = test_inputs.to(self.device), test_targets.to(self.device)
                    test_outputs, _ = self.model(test_inputs)
                    _, test_predicted = torch.max(test_outputs.data, 1)
                    test_total += test_targets.size(0)
                    test_correct += (test_predicted == test_targets).sum().item()
            test_accuracy = 100 * test_correct / test_total

            train_loss = running_loss / len(train_loader)
            
            print(f'Epoch [{epoch + 1}/{self.num_epochs}], Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
            if epoch % self.num_epochs_per_analysis == 0 or epoch == self.num_epochs - 1:
                self.logger.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'fuzziness': self.analyze_fuzziness(train_loader)
                })

        print('Finished Training')

    def evaluate(self, test_loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, _ = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        accuracy = 100 * correct / total
        print(f'Accuracy on the test data: {accuracy}%')
        return accuracy

    def analyze_fuzziness(self, train_loader: DataLoader) -> List[float]:
        raw_fuzziness, vit_fuzziness_list, final_fuzziness = self.fuzziness_analyzer.analyze_fuzziness(self.model, train_loader)
        fuzziness_array = np.concatenate(([raw_fuzziness], vit_fuzziness_list, [final_fuzziness]))
        return fuzziness_array