# Characterizing Spike-Decay in ViTs: How Do Vision Transformers Separate Data?

This repository contains code and resources for the research paper titled "Characterizing Spike-Decay in ViTs: How Do Vision Transformers Separate Data?" The project investigates Vision Transformers (ViTs) through data separation characteristics, focusing on a "spike-decay" pattern. Various experiments were conducted to analyze how parameter adjustments impact inter- and intra-class separability across ViT layers.

# System Overview

![System Overview](/imgs/System%20Overview.png)


## Table of Contents

- [Installation](#installation)
  - [Clone the Repository](#clone-the-repository)
  - [Set Up a Virtual Environment](#set-up-a-virtual-environment)
  - [Install Dependencies](#install-dependencies)
- [Usage](#usage)
  - [Running Experiments with Default Parameters](#running-experiments-with-default-parameters)
  - [Customizing Parameters](#customizing-parameters)
  - [Parameters Explained](#parameters-explained)
  - [How `main.py` Utilizes `setup_experiment`](#how-mainpy-utilizes-setupexperiment)
- [Adding Your Own Dataset](#adding-your-own-dataset)
- [Experiment Outputs](#experiment-outputs)
  - [CSV Files](#csv-files)
  - [Epoch Fuzziness Plots](#epoch-fuzziness-plots)
  - [Training Fuzziness GIF](#training-fuzziness-gif)
  - [Fuzziness vs. Epochs Plot](#fuzziness-vs-epochs-plot)
  - [Per-Layer Fuzziness Evolution Plots](#per-layer-fuzziness-evolution-plots)
  - [Accuracy vs. Epochs Plot](#accuracy-vs-epochs-plot)
- [Citation and Reaching Out](#citation-and-reaching-out)
  - [BibTeX](#bibtex)

## Installation

First, clone this repository and install the required dependencies:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DaraVaram/Transformers-Data-Separation.git
   cd Transformers-Data-Separation
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows, use `env\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run experiments with default parameters, use:

```bash
python main.py
```

### Customizing Parameters

To customize experiment parameters, modify the `setup_experiment` function call in `main.py`:

```python
setup_experiment(
    dataset_name='FashionMNIST',
    variable_name=variable_name,
    variable_value=f'{variable_value}',
    batch_size=64,
    patch_size=7,
    in_channels=1,
    embed_dim=64,
    num_heads=8,
    mlp_dim=128,
    dropout=0.2,
    num_layers=7,
    learning_rate=0.001,
    num_epochs=150,
    num_epochs_per_analysis=10,
    criterion_class=torch.nn.CrossEntropyLoss
)
```

### Parameters Explained

- **`dataset_name`**: Name of the dataset (e.g., 'FashionMNIST').
- **`variable_name` & `variable_value`**: Specify the variable being tested.
- **`batch_size`**: Number of samples per batch.
- **`patch_size`**: Size of image patches.
- **`in_channels`**: Number of input channels (e.g., 1 for grayscale).
- **`embed_dim`**: Dimension of embedding space.
- **`num_heads`**: Number of attention heads.
- **`mlp_dim`**: Dimension of the MLP layer.
- **`dropout`**: Dropout rate.
- **`num_layers`**: Number of transformer layers.
- **`learning_rate`**: Learning rate for optimization.
- **`num_epochs`**: Total epochs for training.
- **`num_epochs_per_analysis`**: Frequency of analysis.
- **`criterion_class`**: Loss function, e.g., `torch.nn.CrossEntropyLoss`.

## Adding Your Own Dataset

If you would like to use a dataset that is not currently supported by this project, you can follow these steps to integrate it. This involves modifying the `DatasetLoader` class in the `dataset_loader.py` file (or similar file where the class is implemented). Here's a step-by-step guide:

1. **Define Transformations:**
   - First, you need to specify the image transformations suitable for your dataset. You can add a new condition in the `set_transformations` method to define how your images should be pre-processed.
   - Common transformations include resizing, normalization, and converting images to tensor format.

   ```python
   if 'YourDatasetName' in self.dataset_name:
       self.transform = transforms.Compose([
           transforms.Resize((height, width)),  # if necessary
           transforms.ToTensor(),
           transforms.Normalize((mean,), (std,))
       ])
       self.img_size = (height, width)
       self.num_classes = number_of_classes
   ```

2. **Load the Dataset:**
   - Modify the `load_data` method to include your dataset.
   - Use the appropriate PyTorch dataset class (or create one if necessary). Common datasets might require using a class from `torchvision.datasets` or creating a custom dataset class.

   ```python
   if self.dataset_name == 'YourDatasetName':
       dataset = CustomDataset(root='path_to_data', train=True, transform=self.transform, download=True)
       test_dataset = CustomDataset(root='path_to_data', train=False, transform=self.transform, download=True)
   ```

3. **Split and Load Data:**
   - Ensure your dataset is split into training and testing subsets.
   - You can adapt the `random_split` function for your dataset if needed.

   ```python
   total_size = int(len(dataset) * self.use_percentage)
   train_size = int(total_size * self.train_percentage)
   test_size = total_size - train_size

   # Adjust if you need a specific way to split your dataset
   subset_dataset, _ = random_split(dataset, [total_size, len(dataset) - total_size])
   train_dataset, test_dataset = random_split(subset_dataset, [train_size, test_size])
   ```

4. **DataLoader Integration:**
   - Utilize PyTorch's `DataLoader` to allow easy batch processing during training.
   - Ensure the `get_train_loader` and `get_test_loader` methods return DataLoader instances for your specific dataset configuration.

   ```python
   self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
   self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
   ```

5. **Update Dataset Information Method:**
   - Modify `get_dataset_info` to properly reflect the specifics about your dataset when it is integrated for use within this framework.

   ```python
   def get_dataset_info(self):
       return {
           "dataset_name": self.dataset_name,
           "image_size": self.img_size,
           "num_classes": self.num_classes
       }
   ```

By following these steps, your custom dataset will be available for use in the experiments, and you can configure it through the `setup_experiment` function in `main.py` as outlined earlier in the guide.

## Experiment Outputs

Running an experiment in this project generates several output files and visualizations, which are saved in the specified `output_dir` (default is "plots"). These outputs aid in analyzing the model's performance and understanding the "spike-decay" characteristics in Vision Transformers. Here’s a list of the outputs produced after running an experiment:

1. **CSV Files:**
   - Contains raw data after each experiment, including metrics like fuzziness, train loss, train accuracy, and test accuracy per epoch.

2. **Epoch Fuzziness Plots:**
   - Located in `plots/epoch_sub_plots/` directory, these plots visualize fuzziness for each layer at every epoch.
   - A **highlighted plot** indicates the first and last layer in each epoch, with exponential decay curves fitted to intermediate layers to represent data separability changes over epochs.

3. **Training Fuzziness GIF:**
   - Located in the `plots/` directory, this GIF (`training_fuzziness.gif`) animates the fuzziness plots over all epochs, providing a dynamic view of how model separation characteristics evolve during training.

4. **Fuzziness vs. Epochs Plot:**
   - Located as `fuzziness_vs_epochs.png` in the `plots/` directory, this plot shows how fuzziness for each layer changes across epochs, highlighting trends or shifts in data separability over time.

5. **Per-Layer Fuzziness Evolution Plots:**
   - Saved in `plots/layer_sub_plot/` directory, these plots display fuzziness vs. epochs for each individual layer. This allows for a focused analysis of how specific layers contribute to the overall model behavior across training iterations.

6. **Accuracy vs. Epochs Plot:**
   - Saved as `accuracy_vs_epochs.png` within the `plots/` directory, this plot shows both train and test accuracies over epochs. It provides insights into model performance and potential overfitting or underfitting conditions.

### How `main.py` Utilizes `setup_experiment`

The `main.py` script calls `setup_experiment` from the `utils` module to configure datasets, models, and training processes. Results and logs are generated for further analysis.

## Citation and reaching out
If you found our work useful or helpful for your own research, please consider citing us using the below: 
- ### BibTeX:


```

```

If you have any questions, please feel free to reach out to me through email (b00081313@alumni.aus.edu) or by connecting with me on [LinkedIn](www.linkedin.com/in/dara-varam). 