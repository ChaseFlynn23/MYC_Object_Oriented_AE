import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import mode
import os
import json
import pandas as pd
from tqdm.notebook import tqdm
from IPython.display import Image, display

class XGBoostTrainer:
    def __init__(self, wt_dict, D132H_dict, window_sizes, default_hyperparameters, eta_values, max_depth_values, subsample_values):
        self.wt_dict = wt_dict
        self.D132H_dict = D132H_dict
        self.window_sizes = window_sizes
        self.default_hyperparameters = default_hyperparameters
        self.eta_values = eta_values
        self.max_depth_values = max_depth_values
        self.subsample_values = subsample_values
        self.default_accuracy_values = {}
        self.best_accuracy_values = {}
        self.best_eta_values = {}
        self.best_max_depth_values = {}
        self.best_subsample_values = {}
        self.graph_save_path = 'XGB_Tuning_Graphs'
        os.makedirs(self.graph_save_path, exist_ok=True)

    @staticmethod
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    def prepare_data(self, window_size):
        wildtype_data = self.wt_dict[window_size]
        wildtype_label = np.zeros(len(wildtype_data))
        mutant_data = self.D132H_dict[window_size]
        mutant_label = np.ones(len(mutant_data))

        lcc_data = np.vstack((wildtype_data, mutant_data))
        label_data = np.hstack((wildtype_label, mutant_label))
        lcc_data, label_data = self.unison_shuffled_copies(lcc_data, label_data)
        lcc_data /= 100
        upper_training_limit = int(len(lcc_data) * 0.8)

        return lcc_data[:upper_training_limit], label_data[:upper_training_limit], lcc_data[upper_training_limit:], label_data[upper_training_limit:]

    def train_and_evaluate(self, train_data, train_label, test_data, test_label, **hyperparameters):
        model = XGBClassifier(**hyperparameters)
        model.fit(train_data, train_label)
        predictions = model.predict(test_data)
        return accuracy_score(test_label, predictions)

    def find_best_hyperparameter(self, train_data, train_label, test_data, test_label, hyperparameter_name, values):
        best_score = 0
        best_value = None
        for value in values:
            self.default_hyperparameters[hyperparameter_name] = value
            score = self.train_and_evaluate(train_data, train_label, test_data, test_label, **self.default_hyperparameters)
            if score > best_score:
                best_score = score
                best_value = value
        return best_value, best_score

    def evaluate_default_hyperparameters(self):
        for window_size in self.window_sizes:
            train_data, train_label, test_data, test_label = self.prepare_data(window_size)
            accuracy = self.train_and_evaluate(train_data, train_label, test_data, test_label, **self.default_hyperparameters)
            self.default_accuracy_values[window_size] = accuracy

    def tune_hyperparameters(self):
        for window_size in self.window_sizes:
            train_data, train_label, test_data, test_label = self.prepare_data(window_size)
            hyperparameters = self.default_hyperparameters.copy()
            best_eta, _ = self.find_best_hyperparameter(train_data, train_label, test_data, test_label, 'eta', self.eta_values)
            self.best_eta_values[window_size] = best_eta
            hyperparameters['eta'] = best_eta

            best_max_depth, _ = self.find_best_hyperparameter(train_data, train_label, test_data, test_label, 'max_depth', self.max_depth_values)
            self.best_max_depth_values[window_size] = best_max_depth
            hyperparameters['max_depth'] = best_max_depth

            best_subsample, best_accuracy = self.find_best_hyperparameter(train_data, train_label, test_data, test_label, 'subsample', self.subsample_values)
            self.best_subsample_values[window_size] = best_subsample
            self.best_accuracy_values[window_size] = best_accuracy

    def tune_hyperparameters_and_save(self):
        self.tune_hyperparameters()
        self.save_tuning_results()

    def save_tuning_results(self, trial_number=None):
        if trial_number is None:
            trial_number = self.get_next_trial_number()

        save_path = f'XGB_Tuning/XGB_Tuning_Trial_{trial_number}'
        os.makedirs(save_path, exist_ok=True)

        results = {
            'best_eta_values': self.best_eta_values,
            'best_max_depth_values': self.best_max_depth_values,
            'best_subsample_values': self.best_subsample_values,
            'best_accuracy_values': self.best_accuracy_values
        }

        with open(f'{save_path}/tuning_results.json', 'w') as f:
            json.dump(results, f, indent=4)

        print(f"Tuning results saved in {save_path}")

    def load_tuning_results(self, trial_number):
        path = f'XGB_Tuning/XGB_Tuning_Trial_{trial_number}/tuning_results.json'
        if os.path.exists(path):
            with open(path, 'r') as f:
                results = json.load(f)
            self.best_eta_values = results['best_eta_values']
            self.best_max_depth_values = results['best_max_depth_values']
            self.best_subsample_values = results['best_subsample_values']
            self.best_accuracy_values = results['best_accuracy_values']
            print(f"Tuning results loaded from {path}")
        else:
            raise FileNotFoundError(f"No tuning results found for trial number {trial_number}")

    @staticmethod
    def get_next_trial_number():
        base_path = 'XGB_Tuning'
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            return 1
        else:
            existing_trials = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
            trial_numbers = [int(trial.split('_')[-1]) for trial in existing_trials if trial.startswith('XGB_Tuning_Trial_')]
            if trial_numbers:
                return max(trial_numbers) + 1
            else:
                return 1
        
    def calculate_mode_hyperparameter_accuracies(self):
        def custom_mode(values):
            if not values:
                return None
            value_counts = {}
            for v in values:
                if v in value_counts:
                    value_counts[v] += 1
                else:
                    value_counts[v] = 1
            max_count = max(value_counts.values())
            most_common = [k for k, v in value_counts.items() if v == max_count]
            return most_common[0] if most_common else None

        # Calculate custom mode for each hyperparameter
        common_hyperparameters = {
            'eta': custom_mode(list(self.best_eta_values.values())),
            'max_depth': custom_mode(list(self.best_max_depth_values.values())),
            'subsample': custom_mode(list(self.best_subsample_values.values())),
            **self.default_hyperparameters
        }

        mode_accuracies = {}
        for window_size in self.window_sizes:
            train_data, train_label, test_data, test_label = self.prepare_data(window_size)
            accuracy = self.train_and_evaluate(train_data, train_label, test_data, test_label, **common_hyperparameters)
            mode_accuracies[window_size] = accuracy

        return mode_accuracies
        
    def save_feature_importance_and_plot(self):
        feature_importances_folder = 'XGB_Position_Importance_Values'
        feature_importance_plot_folder = 'XGB_Pos_Imp_Figs'

        # Check if folders exist and contain files
        if os.path.exists(feature_importances_folder) and os.listdir(feature_importances_folder) and os.path.exists(feature_importance_plot_folder) and os.listdir(feature_importance_plot_folder):
            print("Position importance already determined")
            return

        # If folders are empty or do not exist, proceed to create them
        os.makedirs(feature_importances_folder, exist_ok=True)
        os.makedirs(feature_importance_plot_folder, exist_ok=True)

        # Custom mode calculation function
        def custom_mode(values):
            if not values:
                return None
            value_counts = {}
            for v in values:
                if v in value_counts:
                    value_counts[v] += 1
                else:
                    value_counts[v] = 1
            max_count = max(value_counts.values())
            most_common = [k for k, v in value_counts.items() if v == max_count]
            return most_common[0] if most_common else None

        # Calculate custom mode for each hyperparameter
        common_hyperparameters = {
            'eta': custom_mode(list(self.best_eta_values.values())),
            'max_depth': custom_mode(list(self.best_max_depth_values.values())),
            'subsample': custom_mode(list(self.best_subsample_values.values())),
            **self.default_hyperparameters
        }

        total_positions_saved = 0

        for window_size in self.window_sizes:
            train_data, train_label, _, _ = self.prepare_data(window_size)
            model = XGBClassifier(**common_hyperparameters)
            model.fit(train_data, train_label)

            importances = model.feature_importances_

            # Save feature importances for each window size
            positions = np.arange(1, len(importances) + 1)
            df_importances = pd.DataFrame({'Position': positions, 'Importance': importances})
            importance_file_path = os.path.join(feature_importances_folder, f'feature_importances_window_{window_size}.csv')
            df_importances.to_csv(importance_file_path, index=False)

            # Generate and save feature importance plots
            fig, ax = plt.subplots(figsize=(9, 6))
            indices = np.arange(len(importances))
            ax.bar(indices, importances, color='#0504aa')
            ax.set_title(f'Feature Importance Values for Window Size {window_size}')
            ax.grid(True)
            ax.set_xlabel('Feature')
            ax.set_ylabel('Feature Importance')
            ax.set_ylim(0, np.max(importances) * 1.1)
            plot_path = os.path.join(feature_importance_plot_folder, f'feature_importance_{window_size}.png')
            fig.savefig(plot_path)
            plt.close(fig)

    def plot_hyperparameter_values(self, window_sizes, default_accuracies, tuned_accuracies, mode_accuracies, title):
        plt.figure(figsize=(10, 6))
        window_sizes_list = [int(ws) for ws in window_sizes]
        default_acc_values = [default_accuracies.get(ws, None) for ws in window_sizes_list]
        tuned_acc_values = [tuned_accuracies.get(str(ws), None) for ws in window_sizes_list]
        mode_acc_values = [mode_accuracies.get(ws, None) for ws in window_sizes_list]

        valid_window_sizes = [ws for ws in window_sizes_list if default_accuracies.get(ws, None) is not None or tuned_accuracies.get(str(ws), None) is not None or mode_accuracies.get(ws, None) is not None]

        plt.plot(valid_window_sizes, default_acc_values, label='Default', marker='o')
        plt.plot(valid_window_sizes, tuned_acc_values, label='Tuned', marker='x')
        plt.plot(valid_window_sizes, mode_acc_values, label='Mode Hyperparameters', marker='^')  

        plt.xticks(valid_window_sizes)
        plt.xlabel('Window Size')
        plt.ylabel('Accuracy')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()


    def plot_specific_hyperparameter_values(self, hyperparameter_values_dict, title, ylabel, possible_values=None):
        window_sizes = list(hyperparameter_values_dict.keys())
        hyperparameter_values = list(hyperparameter_values_dict.values())

        plt.figure(figsize=(10, 6))
        plt.plot(window_sizes, hyperparameter_values, marker='o')
        plt.xticks(window_sizes)
        plt.xlabel('Window Size')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True)
        plt.savefig(os.path.join(self.graph_save_path, f"{ylabel.replace(' ', '_')}.png"))
        plt.close()

    def display_feature_importance_plots(self, output_folder):
        for window_size in self.window_sizes:
            image_path = f"{output_folder}/feature_importance_{window_size}.png"
            display(Image(filename=image_path))
            
    def run_analysis(self, importance_thresholds):
        elbow_threshold = self.calculate_and_plot_positions(importance_thresholds)

        self.save_positions_above_threshold(elbow_threshold)
        
        print(f"Elbow Threshold: {elbow_threshold:.3f}")
            
    def calculate_adjusted_threshold(self, importance_threshold, window_size):
        return importance_threshold * (68 / (70 - window_size))
    
    def read_feature_importances(self, window_size):
        """
        Reads the feature importance values from the CSV files generated by save_important_features_and_plot.
        """
        feature_importances_folder = 'XGB_Position_Importance_Values'
        importance_file_path = os.path.join(feature_importances_folder, f'feature_importances_window_{window_size}.csv')
        if os.path.exists(importance_file_path):
            return pd.read_csv(importance_file_path)
        else:
            print(f"No feature importances file found for window size {window_size}.")
            return pd.DataFrame({'Position': [], 'Importance': []})

    def calculate_and_plot_positions(self, importance_thresholds):
        total_positions_by_threshold = {}
        previous_covered_positions = set()

        for threshold in importance_thresholds:
            total_positions = 0
            plt.figure(figsize=(10, 6))
            current_covered_positions = set()

            for window_size in self.window_sizes:
                adjusted_threshold = self.calculate_adjusted_threshold(threshold, window_size)
                df_importances = self.read_feature_importances(window_size)

                if not df_importances.empty:
                    important_features = df_importances[df_importances['Importance'] >= adjusted_threshold]['Position']
                    total_positions += len(important_features)

                    for feature in important_features:
                        start_residue = feature + 90
                        end_residue = start_residue + window_size
                        current_position_range = (start_residue, end_residue)
                        current_covered_positions.add(current_position_range)

                        color = 'blue' if current_position_range in previous_covered_positions else 'red'
                        plt.plot([start_residue, end_residue], [window_size, window_size], color=color, marker='o', markersize=5, linewidth=2)

            new_positions = current_covered_positions - previous_covered_positions

            previous_covered_positions.update(current_covered_positions)

            total_positions_by_threshold[threshold] = total_positions

            plt.text(0.95, 0.95, f"New Positions: {len(new_positions)}\nTotal Positions: {total_positions}", verticalalignment='top', horizontalalignment='right', transform=plt.gca().transAxes)

            from matplotlib.lines import Line2D
            custom_lines = [Line2D([0], [0], color='red', lw=4),
                            Line2D([0], [0], color='blue', lw=4)]
            plt.legend(custom_lines, ['New Positions', 'Existing Positions'], loc='upper left')

            plt.title(f'Residues Covered for Threshold {threshold:.3f}')
            plt.xlabel('Residue Position')
            plt.ylabel('Window Size')
            plt.xlim(91, 160)
            plt.ylim(0, 51)
            plt.xticks(range(91, 161, 5))
            plt.yticks(range(0, 52, 2))
            plt.grid(True)
            plt.show()
       
        # Convert the total_positions_by_threshold dictionary to a pandas DataFrame
        df_total_positions = pd.DataFrame(list(total_positions_by_threshold.items()), columns=['Threshold', 'TotalPositions'])

        # Save the DataFrame to a CSV file
        df_total_positions.to_csv('total_positions_by_threshold.csv', index=False)
        
        # After processing all thresholds, plot the total number of positions above each threshold
        thresholds = sorted(total_positions_by_threshold.keys())
        total_positions = [total_positions_by_threshold[threshold] for threshold in thresholds]

        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, total_positions, '-o', color='blue')

        # Identify the elbow point
        kneedle = KneeLocator(thresholds, total_positions, curve='convex', direction='decreasing', interp_method='polynomial')
        elbow_threshold = kneedle.elbow
        elbow_total_positions = kneedle.elbow_y

        # Plot the elbow point
        plt.scatter(elbow_threshold, elbow_total_positions, color='red', s=100, label=f'Elbow at {elbow_threshold:.3f}', zorder=5)
        
        # Plot formatting
        plt.title('Total Positions Above Threshold by Threshold')
        plt.xlabel('Importance Threshold')
        plt.ylabel('Total Number of Positions Above Threshold')
        plt.grid(True)
        plt.gca().invert_xaxis()
        plt.show()
        
        return elbow_threshold
                
    def save_positions_above_threshold(self, elbow_threshold):
        """
        Filters and saves positions above the adjusted threshold for each window size.
        If 'XGB_filtered_data' folder exists with any number of files, it prints 
        "Filtered data already generated" and does not overwrite existing data.
        """
        output_folder = 'XGB_filtered_data'

        # Check if folder exists and contains any files
        if os.path.exists(output_folder) and os.listdir(output_folder):
            print("Filtered data already generated")
            return  # Exit the function early

        os.makedirs(output_folder, exist_ok=True)

        total_positions_saved = 0
        for window_size in self.window_sizes:
            adjusted_threshold = self.calculate_adjusted_threshold(elbow_threshold, window_size)
            df_importances = self.read_feature_importances(window_size)

            if not df_importances.empty:
                important_features = df_importances[df_importances['Importance'] >= adjusted_threshold]['Position']
                total_positions_saved += len(important_features)

                if len(important_features) > 0:
                    print(f"Window size {window_size}: {len(important_features)} positions saved.")

                    # Filter WT and D132H data based on important features
                    wt_filtered_data = self.wt_dict[window_size].filter(items=important_features.tolist(), axis='columns')
                    D132H_filtered_data = self.D132H_dict[window_size].filter(items=important_features.tolist(), axis='columns')

                    # Save the filtered data to CSV
                    wt_filtered_data.to_csv(os.path.join(output_folder, f'wt_{window_size}f.csv'), index_label='Index')
                    D132H_filtered_data.to_csv(os.path.join(output_folder, f'D132H_{window_size}f.csv'), index_label='Index')

        print(f"Total positions saved: {total_positions_saved}")
