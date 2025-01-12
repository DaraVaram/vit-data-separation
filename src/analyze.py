import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List

class FuzzinessAnalyzer:
    def __init__(self, device: torch.device):
        self.device = device

    def analyze_fuzziness(self, model: torch.nn.Module, train_loader: DataLoader) -> Tuple[float, List[float], float]:
        self.model = model
        raw_fuzziness, vit_fuzziness_list, final_fuzziness = self.analyze_representations_full(train_loader)
        return raw_fuzziness, vit_fuzziness_list, final_fuzziness

    def analyze_representations_full(self, train_loader: DataLoader) -> Tuple[float, List[float], float]:
        print("Calculating Separation Fuzziness Before ViT...")
        raw_features, raw_labels = self.get_raw_image_features_labels(train_loader)
        raw_fuzziness = self.get_variation(raw_features, raw_labels)
        print(f"Separation Fuzziness (Before ViT): {raw_fuzziness}")

        print("Calculating Separation Fuzziness During ViT...")
        vit_fuzziness_list = self.analyze_representations_vit(train_loader)

        print("Calculating Separation Fuzziness After ViT...")
        final_features, final_labels = self.get_final_output_features_labels(train_loader)
        final_fuzziness = self.get_variation(final_features, final_labels)
        print(f"Separation Fuzziness (After ViT): {final_fuzziness}")

        return raw_fuzziness, vit_fuzziness_list, final_fuzziness

    @staticmethod
    def get_variation(features: np.ndarray, labels: np.ndarray) -> float:
        features = features.astype(np.float64).reshape(len(features), -1)
        labels = copy.deepcopy(labels)
        num_classes = len(np.unique(labels))
        avg_feature = np.mean(features, axis=0)
        features -= avg_feature
        class_features = FuzzinessAnalyzer.get_class_features(features, labels)
        feature_dim = features.shape[1]

        between_class_covariance = np.zeros((feature_dim, feature_dim))
        within_class_covariance = np.zeros((feature_dim, feature_dim))

        for cur_class_features in class_features:
            cur_class_avg_feature = np.mean(cur_class_features, axis=0)

            # Between-class covariance
            between_class_covariance += np.outer(cur_class_avg_feature, cur_class_avg_feature)

            # Within-class covariance
            cur_class_centralized_features = cur_class_features - cur_class_avg_feature
            cur_class_covariance = cur_class_centralized_features.T @ cur_class_centralized_features / len(cur_class_features)

            within_class_covariance += cur_class_covariance

        between_class_covariance /= num_classes
        within_class_covariance /= num_classes

        # Pseudo-inverse of the between-class covariance matrix
        between_class_inverse_covariance = np.linalg.pinv(between_class_covariance, rcond=1e-10)

        # Calculate D = trace(SSW * SSB_inv)
        D = np.trace(within_class_covariance @ between_class_inverse_covariance)
        return D

    @staticmethod
    def get_class_features(features: np.ndarray, labels: np.ndarray) -> List[np.ndarray]:
        num_classes = np.unique(labels).size
        class_features = [[] for _ in range(num_classes)]
        for i in range(labels.size):
            class_features[labels[i]].append(features[i])
        return [np.array(c) for c in class_features]

    def get_features_labels_vit(self, train_loader: DataLoader, option: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        total_features, total_labels = [], []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            _, intermediate_outputs = self.model(inputs)
            features = intermediate_outputs[option][:, 0, :]
            total_features.extend(features.cpu().data.numpy())
            total_labels.extend(targets.cpu().data.numpy())

        return np.array(total_features), np.array(total_labels, dtype=int)

    def get_raw_image_features_labels(self, train_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        total_features, total_labels = [], []
        for inputs, targets in train_loader:
            inputs = inputs.view(inputs.size(0), -1)
            total_features.extend(inputs.cpu().data.numpy())
            total_labels.extend(targets.cpu().data.numpy())
        return np.array(total_features), np.array(total_labels, dtype=int)

    def get_final_output_features_labels(self, train_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        total_features, total_labels = [], []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            final_output, _ = self.model(inputs)
            total_features.extend(final_output.cpu().data.numpy())
            total_labels.extend(targets.cpu().data.numpy())
        return np.array(total_features), np.array(total_labels, dtype=int)

    def analyze_representations_vit(self, train_loader: DataLoader) -> List[float]:
        rate_reduction_list = []
        total_layers = len(list(self.model.encoder_layers))
        for feature_option in range(total_layers):
            features, labels = self.get_features_labels_vit(train_loader, feature_option)
            rate_reduction = self.get_variation(features, labels)
            print(f'Layer-{feature_option+1} Separation Fuzziness: {rate_reduction}')
            rate_reduction_list.append(rate_reduction)
        return rate_reduction_list