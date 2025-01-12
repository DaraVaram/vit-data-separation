import csv
import os
from typing import Dict, Any

class ExperimentLogger:
    def __init__(self, log_dir: str, filename: str = 'experiment_log.csv'):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.filepath = os.path.join(log_dir, filename)
        self.fieldnames = ['epoch', 'train_loss', 'train_accuracy', 'test_accuracy', 'fuzziness']

        with open(self.filepath, mode='w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, log_data: Dict[str, Any]):       
        with open(self.filepath, mode='a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(log_data)