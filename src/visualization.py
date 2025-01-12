import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import imageio
import os

sns.set_palette('deep')

class Plotter:
    def __init__(self, csv_files, output_dir: str = "plots"):
        if isinstance(csv_files, str):
            csv_files = [csv_files]

        parsed_fuzziness_data = []

        for file in csv_files:
            df = pd.read_csv(file)
            fuzziness_values = df['fuzziness'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
            parsed_fuzziness_data.append(fuzziness_values.tolist())

        avg_fuzziness = []

        for epoch_fuzziness in zip(*parsed_fuzziness_data):
            stack = np.vstack(epoch_fuzziness)
            mean_values = stack.mean(axis=0)
            avg_fuzziness.append(mean_values)

        base_data = pd.read_csv(csv_files[0])
        averaged_data = base_data.drop(columns=['fuzziness']).copy()

        for other_file in csv_files[1:]:
            temp_data = pd.read_csv(other_file).drop(columns=['fuzziness'])
            averaged_data = averaged_data.add(temp_data)

        averaged_data /= len(csv_files)
        averaged_data['fuzziness'] = avg_fuzziness

        self.data = averaged_data
        self.all_fuzziness_values = self.data['fuzziness']
        
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def plot_epoch_fuzziness(self):
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
        plots_filenames = []
        global_min = min(np.min(fuzziness) for fuzziness in self.all_fuzziness_values)
        global_max = max(np.max(fuzziness) for fuzziness in self.all_fuzziness_values)

        for index, row in self.data.iterrows():
            epoch = row['epoch']
            train_loss = row['train_loss']
            train_accuracy = row['train_accuracy']
            test_accuracy = row['test_accuracy']
            fuzziness = self.all_fuzziness_values[index]

            first_value = fuzziness[0]
            main_values = fuzziness[1:-1]
            last_value = fuzziness[-1]

            plt.figure(figsize=(11, 6))
            sns.scatterplot(x=[0], y=[first_value], color='red')
            sns.scatterplot(x=range(1, len(fuzziness) - 1), y=main_values, color='blue')
            sns.scatterplot(x=[len(fuzziness) - 1], y=[last_value], color='green')
            # plt.yscale('log')
            plt.title(f'Epoch {epoch}')
            plt.xlabel('Layer Index')
            plt.ylabel('Fuzziness')
            plt.xlim(-0.5, len(fuzziness) - 0.5)
            plt.ylim(global_min, global_max * 1.05)

            try:
                x_data_fit = np.arange(1, len(main_values) + 1)
                popt, _ = curve_fit(exp_decay, x_data_fit, main_values, p0=(100, 0.1, 0), maxfev=2000)
                x_fit = np.linspace(1, len(main_values), 100)
                plt.plot(x_fit, exp_decay(x_fit, *popt), label="Exponential Decay Fit", color="red", linestyle='--')

                y_pred = exp_decay(x_data_fit, *popt)
                r_squared = r2_score(main_values, y_pred)
            except RuntimeError as e:
                print(f"Error in curve fitting for epoch {epoch}: {e}")
                r_squared = float('nan')

            plt.text(
                len(fuzziness) * 0.75, global_max * 0.9,
                (f'Train Loss: {train_loss:.2f}\n'
                 f'Train Acc: {train_accuracy:.2f}%\n'
                 f'Test Acc: {test_accuracy:.2f}%\n'
                 f'R² (Decay Fit): {r_squared:.2f}'),
                fontsize=10, va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.5)
            )

            if not os.path.exists(self.output_dir + '/epoch_sub_plots'):
                os.makedirs(self.output_dir + '/epoch_sub_plots')
            filename = os.path.join(self.output_dir + "/epoch_sub_plots/", f'epoch_{epoch}.png')
            plt.savefig(filename)
            plots_filenames.append(filename)
            plt.close()

        self.create_gif(plots_filenames)
    
    def create_gif(self, plot_filenames):
        gif_filename = os.path.join(self.output_dir, 'training_fuzziness.gif')
        frames = [imageio.imread(filename) for filename in plot_filenames]
        imageio.mimsave(gif_filename, frames, format='GIF', fps=1)
        print(f'GIF saved as {gif_filename}')

    def plot_fuzziness_vs_epochs(self):
        epochs = self.data['epoch']
        plt.figure(figsize=(11, 6))
        palette = sns.color_palette('deep', n_colors=len(self.all_fuzziness_values[0]) - 2)

        layer_count = len(self.all_fuzziness_values[0]) - 2
        for layer_idx in range(layer_count):
            fuzziness_over_epochs = [fuzziness[layer_idx + 1] for fuzziness in self.all_fuzziness_values]
            plt.plot(epochs, fuzziness_over_epochs, marker='o', label=f'Layer {layer_idx + 1}', color=palette[layer_idx])

        # plt.plot(epochs, self.data['test_accuracy'], marker='o', label='Test Accuracy', linestyle='--', color='black')
        plt.xlabel('Epochs')
        plt.ylabel('Fuzziness / Accuracy')
        plt.title('Fuzziness vs Epochs for Each Layer')
        plt.legend()
        plt.ylim(0, None)
        plt.xticks(epochs)
        plt.savefig(os.path.join(self.output_dir, 'fuzziness_vs_epochs.png'))
        plt.close()
        
    def plot_fuzziness_vs_epochs_per_layer(self):
        epochs = self.data['epoch']
        layer_count = len(self.all_fuzziness_values[0]) - 2
        palette = sns.color_palette('deep', n_colors=layer_count)

        global_min = min(np.min(fuzziness[1:-1]) for fuzziness in self.all_fuzziness_values)
        global_max = max(np.max(fuzziness[1:-1]) for fuzziness in self.all_fuzziness_values)

        for layer_idx in range(layer_count):
            plt.figure(figsize=(11, 6))
            fuzziness_over_epochs = [fuzziness[layer_idx + 1] for fuzziness in self.all_fuzziness_values]
            plt.plot(epochs, fuzziness_over_epochs, marker='o', label=f'Layer {layer_idx + 1}', color=palette[layer_idx])
            plt.xlabel('Epochs')
            plt.ylabel('Fuzziness')
            plt.title(f'Fuzziness vs Epochs for Layer {layer_idx + 1}')
            # plt.ylim(0, global_max * 1.05)
            plt.xticks(epochs)
            plt.legend()
            if not os.path.exists(self.output_dir + '/layer_sub_plot'):
                os.makedirs(self.output_dir + '/layer_sub_plot')
            plt.savefig(os.path.join(self.output_dir + "/layer_sub_plot/", f'fuzziness_vs_epochs_layer_{layer_idx + 1}.png'))
            plt.close()

        plt.figure(figsize=(11, 6))
        plt.plot(epochs, self.data['test_accuracy'], marker='o', label='Test Accuracy', linestyle='--', color='black')
        plt.plot(epochs, self.data['train_accuracy'], marker='o', label='Train Accuracy', linestyle='--', color='red')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epochs')
        plt.ylim(0, 100)
        plt.xticks(epochs)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, 'accuracy_vs_epochs.png'))
        plt.close()
        
    def plot_all(self):
        self.plot_epoch_fuzziness()
        self.plot_fuzziness_vs_epochs()
        self.plot_fuzziness_vs_epochs_per_layer()
