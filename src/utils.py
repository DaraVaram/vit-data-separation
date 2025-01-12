import os
import json
import torch
import re
from datetime import datetime
    
from visualization import Plotter
from datasets import DatasetLoader
from modelHandler import ModelHandler
from models.vision_transformer import VisionTransformerWithIntermediateOutputs

def plot_avg_runs(data_paths):
    match = re.search(r'experiment_([\w-]+)_([\w\d.-]+)_\d+_\d+', data_paths[0])

    if not match:
        print(f"Error: Invalid file name format in the first data path: {data_paths[0]}")
        return

    NAME = match.group(1)
    VALUE = match.group(2)

    for path in data_paths:
        match = re.search(r'experiment_([\w-]+)_([\w\d.-]+)_\d+_\d+', path)
        if not match:
            print(f"Error: Invalid file name format in data path: {path}")
            return
        if match.group(1) != NAME or match.group(2) != VALUE:
            print(f"Error: Inconsistent experiment in data path: {path}")
            return

    plotter = Plotter(data_paths, output_dir=f'experiments/{NAME}/experiment_{NAME}_{VALUE}_averages')
    plotter.plot_all()

def setup_experiment(
    dataset_name: str,
    variable_name: str,
    variable_value: str,
    batch_size: int,
    patch_size: int,
    in_channels: int,
    embed_dim: int,
    num_heads: int,
    mlp_dim: int,
    dropout: float,
    num_layers: int,
    learning_rate: float,
    num_epochs: int,
    num_epochs_per_analysis: int = 10,
    model_class: torch.nn.Module = VisionTransformerWithIntermediateOutputs,
    criterion_class: torch.nn.Module = torch.nn.CrossEntropyLoss,
    optimizer_class: torch.optim.Optimizer = torch.optim.Adam,
    optimizer_params: dict = None,
):
    if optimizer_params is None:
        optimizer_params = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the dataset
    dataset_loader = DatasetLoader(dataset_name, batch_size=batch_size)
    train_loader = dataset_loader.get_train_loader()
    test_loader = dataset_loader.get_test_loader()
    info = dataset_loader.get_dataset_info()

    # Hyperparameters and configuration
    model_params = {
        'img_size': info['image_size'],
        'patch_size': patch_size,
        'in_channels': in_channels,
        'embed_dim': embed_dim,
        'num_heads': num_heads,
        'mlp_dim': mlp_dim,
        'dropout': dropout,
        'num_layers': num_layers,
        'num_classes': info['num_classes']
    }

    # Create an experiment directory and save hyperparameters
    experiment_dir = ExperimentManager.create_experiment_directory(variable_name, variable_value)
    ExperimentManager.save_experiment_params(experiment_dir, {
        'dataset_name': dataset_name,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'model_params': model_params
    })

    if variable_name == 'optimizers':
        valid_optimizers = {
            'AdamW': torch.optim.AdamW,
            'SGD': torch.optim.SGD,
            'ASGD': torch.optim.ASGD
        }
        optimizer_class = valid_optimizers.get(variable_value, torch.optim.Adam)

    # Create model handler
    model_handler = ModelHandler(
        model_class,
        model_params,
        device,
        criterion_class,
        optimizer_class,
        optimizer_params,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        log_dir=experiment_dir,
        num_epochs_per_analysis=num_epochs_per_analysis
    )

    # Train and evaluate model
    model_handler.train(train_loader, test_loader)
    # accuracy = model_handler.evaluate(test_loader)
    # fuzziness_array = model_handler.analyze_fuzziness(train_loader)
    
    # Plot the fuzziness
    data_path = os.path.join(experiment_dir, "experiment_log.csv")
    plotter = Plotter(data_path, output_dir=experiment_dir + "/plots")
    plotter.plot_all()

    # Save the model
    model_path = os.path.join(experiment_dir, "model_vit.pt")
    torch.save(model_handler.model.state_dict(), model_path)
    print(f"Model saved at {model_path}")

    return data_path

class ExperimentManager:
    base_dir = "experiments"

    @staticmethod
    def create_experiment_directory(variable_name: str, variable_value: str) -> str:
        """Create a new directory for the experiment, return the directory path."""
        if not os.path.exists(ExperimentManager.base_dir):
            os.makedirs(ExperimentManager.base_dir)
        
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(ExperimentManager.base_dir, variable_name, f"experiment_{variable_name}_{variable_value}_{experiment_id}")
        os.makedirs(experiment_dir)
        return experiment_dir

    @staticmethod
    def save_experiment_params(experiment_dir: str, params: dict):
        """Save parameters and config of the experiment to a JSON file."""
        params_filepath = os.path.join(experiment_dir, "parameters.json")
        with open(params_filepath, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"Experiment parameters saved at {params_filepath}")