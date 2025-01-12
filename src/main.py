import torch
from utils import setup_experiment, plot_avg_runs
import multiprocessing
import time

def add_data_path(variable_name, variable_value, queue, run_index):
    time.sleep(run_index * 2)
    result = setup_experiment(
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
    queue.put(result)

def process_exp(variable_name, variable_value, num_runs):
    data_paths = []
    queue = multiprocessing.Queue()

    processes = []
    for run_index in range(num_runs):
        process = multiprocessing.Process(target=add_data_path, args=(variable_name, variable_value, queue, run_index))
        process.start()
        processes.append(process)

    for process in processes:
        process.join()
    
    while not queue.empty():
        data_paths.append(queue.get())

    if len(data_paths) > 1:
        plot_avg_runs(data_paths)

if __name__ == "__main__":
    num_runs = 10

    # variable_name = 'dropout'
    # variable_list = [0.1, 0.3, 0.4, 0.5, 0.6, 0.7]

    # variable_name = 'embed-dim'
    # variable_list = [32, 128, 256, 512]

    # variable_name = 'mlp-dim'
    # variable_list = [32, 64, 256, 512]

    # variable_name = 'num-heads'
    # variable_list = [2, 4, 16, 32]

    variable_name = 'optimizers'
    variable_list = ['AdamW', 'SGD', 'ASGD']

    top_level_processes = []
    for variable_value in variable_list:
        if len(top_level_processes) >= 1:
            top_level_processes[0].join()
            top_level_processes.pop(0)
            
        top_level_process = multiprocessing.Process(target=process_exp, args=(variable_name, variable_value, num_runs))
        top_level_process.start()
        top_level_processes.append(top_level_process)
        time.sleep(2 * num_runs + 1)

    for process in top_level_processes:
        process.join()