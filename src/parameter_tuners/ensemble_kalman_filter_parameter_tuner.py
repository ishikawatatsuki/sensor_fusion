import sys
if __name__ == "__main__":
    sys.path.append('../../src')

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from data_loader import DataLoader
from configs import SetupEnum, MeasurementDataEnum, Configs, ErrorEnum
from kalman_filters.ensemble_kalman_filter import EnsembleKalmanFilter
import matplotlib.pyplot as plt
import seaborn as sns

import warnings


class EnsembleKalmanFilterParameterTuner:

    time_df = None
    mae_error_df = None
    rmse_error_df = None
    max_error_df = None
    
    
    def __init__(
        self, 
        n_samples, 
        kitti_dataset,
        file_export_path,
        kitti_root_dir,
        vo_root_dir,
        setup=SetupEnum.SETUP_1, 
        measurement_type=MeasurementDataEnum.ALL_DATA):

        self.setup = setup
        self.measurement_type = measurement_type
        self.n_samples = n_samples
        self.dropout_ratios = [0.0, 0.1, 0.2, 0.3, 0.4]
        
        self.kitti_drive = kitti_dataset
        self.file_export_path = file_export_path
        self.kitti_root_dir = kitti_root_dir
        self.vo_root_dir = vo_root_dir
        

    def run_dummy_filter(self, base_time):
        return np.random.rand(), base_time + np.random.randint(0, 30)
    
    def run_filter_setup1(self, N, data):
        x, P, H = data.get_initial_data(setup=self.setup)

        enkf = EnsembleKalmanFilter(
            N=N, 
            x=x.copy(), 
            P=P.copy(), 
            H=H.copy())
        
        start = datetime.now()
        error = enkf.run(
            data=data, 
            setup=self.setup,
            measurement_type=self.measurement_type)
        
        end = datetime.now()
        processing_time = (end - start).total_seconds()
        
        return error, np.round(processing_time / data.N, Configs.processing_time_decimal_place)
    
    def run_filter_setup2(self, N, data):
        x, P, H = data.get_initial_data(setup=self.setup)

        enkf = EnsembleKalmanFilter(
            N=N, 
            x=x.copy(), 
            P=P.copy(), 
            H=H.copy())
        
        start = datetime.now()
        error = enkf.run(
            data=data, 
            setup=self.setup,
            measurement_type=self.measurement_type)
        
        end = datetime.now()
        processing_time = (end - start).total_seconds()

        return error, np.round(processing_time / data.N, Configs.processing_time_decimal_place)
    
    def run_filter_setup3(self, N, data):
        x, P, H = data.get_initial_data(setup=self.setup)

        enkf = EnsembleKalmanFilter(
            N=N, 
            x=x.copy(), 
            P=P.copy(), 
            H=H.copy())
        
        start = datetime.now()
        error = enkf.run(
            data=data, 
            setup=self.setup,
            measurement_type=self.measurement_type)
        
        end = datetime.now()
        processing_time = (end - start).total_seconds()

        return error, np.round(processing_time / data.N, Configs.processing_time_decimal_place)

    def run(self):

        mae_errors = []
        rmse_errors = []
        max_errors = []
        processing_times = []
        for dropout_ratio in self.dropout_ratios:
            print(f"Setting dropout ratio by {str(dropout_ratio)}")
            
            data = DataLoader(
                sequence_nr=self.kitti_drive, 
                vo_dropout_ratio=dropout_ratio,
                gps_dropout_ratio=dropout_ratio,
                vo_root_dir=self.vo_root_dir,
                kitti_root_dir=self.kitti_root_dir,
                visualize_data=False)

            mae_errors_ = []
            rmse_errors_ = []
            max_errors_ = []
            processing_times_ = []
            for N in tqdm(self.n_samples):
                # error, processing_time = self.run_dummy_filter(base_time=N)
                # mae_errors_.append(error)
                # rmse_errors_.append(error)
                # max_errors_.append(error)
                # processing_times_.append(processing_time)
                
                if self.setup is SetupEnum.SETUP_1:
                    error, processing_time = self.run_filter_setup1(N=N, data=data)
                    
                elif self.setup is SetupEnum.SETUP_2:
                    error, processing_time = self.run_filter_setup2(N=N, data=data)
                    
                else: #SetupEnum.SETUP_3
                    error, processing_time = self.run_filter_setup3(N=N, data=data)
                    
                mae_errors_.append(error[ErrorEnum.MAE])
                rmse_errors_.append(error[ErrorEnum.RMSE])
                max_errors_.append(error[ErrorEnum.MAX])
                processing_times_.append(processing_time)
                
            mae_errors.append(mae_errors_)
            rmse_errors.append(rmse_errors_)
            max_errors.append(max_errors_)
            processing_times.append(processing_times_)

        print("Experiment finished.")
        self.time_df = pd.DataFrame(processing_times, columns=self.n_samples, index=self.dropout_ratios)
        self.mae_error_df = pd.DataFrame(mae_errors, columns=self.n_samples, index=self.dropout_ratios)
        self.rmse_error_df = pd.DataFrame(rmse_errors, columns=self.n_samples, index=self.dropout_ratios)
        self.max_error_df = pd.DataFrame(max_errors, columns=self.n_samples, index=self.dropout_ratios)
        self.dump_df()

    def find_best_combination(
        self, 
        error_weight, 
        error_upper_limit=500):
        """
            weighted sum method to find the best combination of parameters.
        """       
        error_types = ErrorEnum.get_all()

        optimal_sample_size = {}
        best_n_samples = []
        for error_type in error_types:
            weight_for_errors = error_weight
            weight_for_time = 1. - error_weight

            if error_type is ErrorEnum.MAE:
                errors = self.mae_error_df.values
            elif error_type is ErrorEnum.RMSE:
                errors = self.rmse_error_df.values
            else: # ErrorEnum.MAX
                errors = self.max_error_df.values

            assert errors is not None, "Please load error data frame."

            n_samples = []
            for i, zipped in enumerate(zip(errors, self.time_df.values)):
                errors_, times_ = zipped
                errors_ = np.array(errors_)
                times_ = np.array(times_)
                max_time = np.max(times_)
                max_error = np.max(errors_)                
                if max_error > error_upper_limit:
                    max_error = error_upper_limit
                
                errors_[errors_ > error_upper_limit] = error_upper_limit
            
                normalized_errors = [error / max_error for error in errors_]
                normalized_times = [time / max_time for time in times_]
                # Calculate a combined score for each model
                scores = [weight_for_errors * error + weight_for_time * time 
                        for error, time in zip(normalized_errors, normalized_times)]
            
                # Find the index of the best model
                optimal_index = scores.index(min(scores))
                n_sample_index = optimal_index % len(self.n_samples)
                n_samples.append(self.n_samples[n_sample_index])
            
            optimal_sample_size[error_type.value] = n_samples
            best_n_samples.append(n_samples)

        header = pd.MultiIndex.from_product([ErrorEnum.get_names()], names=['Error types'])
        df = pd.DataFrame(np.array(best_n_samples).T, 
                    index=['No dropout', '10% drop', '20% drop', '30% drop', '40% drop'], 
                    columns=header)

        return optimal_sample_size, df
            
    def plot_results(self):        
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 10))
        (ax1, ax2) = ax[0]
        (ax3, ax4) = ax[1] 
        sns.heatmap(self.time_df,
                    ax=ax1,
                    annot=True,
                    linewidths=1,
                    fmt='.3f')
        ax1.set_title("Processing time (seconds)")
        ax1.set(xlabel="# of samples", ylabel="Dropout ratio for VO and GPS")
        
        sns.heatmap(self.mae_error_df,
                    ax=ax2,
                    cmap="crest",
                    annot=True,
                    linewidths=1,
                    fmt='.3g')
        ax2.set_title("Mean Absolute Error")
        ax2.set(xlabel="# of samples", ylabel="Dropout ratio for VO and GPS")

        sns.heatmap(self.rmse_error_df,
                    ax=ax3,
                    cmap="crest",
                    annot=True,
                    linewidths=1,
                    fmt='.3g')
        ax3.set_title("Root Mean Squared Error")
        ax3.set(xlabel="# of samples", ylabel="Dropout ratio for VO and GPS")

        sns.heatmap(self.max_error_df,
                    ax=ax4,
                    cmap="crest",
                    annot=True,
                    linewidths=1,
                    fmt='.3g')
        ax4.set_title("Maximum Error")
        ax4.set(xlabel="# of samples", ylabel="Dropout ratio for VO and GPS")

        fig.subplots_adjust(wspace=0.2)

    def dump_df(self):
        # saving the results
        self.time_df.to_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/time_df.json")

        self.mae_error_df.to_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/mae_error_df.json")
        self.rmse_error_df.to_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/rmse_error_df.json")
        self.max_error_df.to_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/max_error_df.json")

    def load_df(self):

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.time_df = pd.read_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/time_df.json")

            self.mae_error_df = pd.read_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/mae_error_df.json")
            self.rmse_error_df = pd.read_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/rmse_error_df.json")
            self.max_error_df = pd.read_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/max_error_df.json")

        self.time_df.set_index([self.dropout_ratios], inplace=True)
        self.mae_error_df.set_index([self.dropout_ratios], inplace=True)
        self.rmse_error_df.set_index([self.dropout_ratios], inplace=True)
        self.max_error_df.set_index([self.dropout_ratios], inplace=True)

if __name__ == '__main__':
    pass