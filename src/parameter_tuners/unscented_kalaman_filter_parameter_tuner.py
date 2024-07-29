import sys
if __name__ == "__main__":
    sys.path.append('../../src')

import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from data_loader import DataLoader
from configs import SetupEnum, MeasurementDataEnum, Configs, ErrorEnum
from kalman_filters.unscented_kalman_filter import UnscentedKalmanFilter
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

class UnscentedKalmanFilterParameterTuner:

    time_df = None
    mae_error_df = None
    rmse_error_df = None
    max_error_df = None
    
    def __init__(self, 
                 params, 
                 kitti_dataset,
                 file_export_path,
                 kitti_root_dir,
                 vo_root_dir,
                 setup=SetupEnum.SETUP_1, 
                 measurement_type=MeasurementDataEnum.ALL_DATA, 
                 vo_dropout_ratio=0., 
                 gps_dropout_ratio=0.):

        self.setup = setup
        self.file_export_path = file_export_path
        self.measurement_type = measurement_type
        self.alphas = params["alphas"]
        self.betas = params["betas"]
        self.kappa = params["kappa"]
        print(params)
        print(measurement_type)
        self.vo_dropout_ratio = vo_dropout_ratio
        self.gps_dropout_ratio = gps_dropout_ratio
        print(kitti_root_dir)
        print(vo_root_dir)
        self.kitti_drive = kitti_dataset
        self.data = DataLoader(sequence_nr=self.kitti_drive, 
                               vo_dropout_ratio=vo_dropout_ratio,
                               gps_dropout_ratio=gps_dropout_ratio,
                               kitti_root_dir=kitti_root_dir,
                               vo_root_dir=vo_root_dir,
                               visualize_data=False)

    def run_dummy_filter(self, base_time):
        return np.random.randn(), base_time + np.random.randint(0, 30)
        
    def run_filter_setup1(self, alpha, beta, kappa):
        x, P, H = self.data.get_initial_data(setup=self.setup)

        ukf = UnscentedKalmanFilter(x=x.copy(), 
                                    P=P.copy(), 
                                    H=H.copy(), 
                                    alpha=alpha, 
                                    beta=beta, 
                                    kappa=kappa)
        
        start = datetime.now()
        error = ukf.run(data=self.data, 
                        setup=self.setup,
                        measurement_type=self.measurement_type)
        end = datetime.now()
        processing_time = (end - start).total_seconds()

        return error, np.round(processing_time / self.data.N, Configs.processing_time_decimal_place)
        
    def run_filter_setup2(self, alpha, beta, kappa):
        x, P, H = self.data.get_initial_data(setup=self.setup)

        ukf = UnscentedKalmanFilter(x=x.copy(), 
                                    P=P.copy(), 
                                    H=H.copy(), 
                                    alpha=alpha, 
                                    beta=beta, 
                                    kappa=kappa)
        
        start = datetime.now()
        error = ukf.run(data=self.data, 
                        setup=self.setup,
                        measurement_type=self.measurement_type)
        end = datetime.now()
        processing_time = (end - start).total_seconds()

        return error, np.round(processing_time / self.data.N, Configs.processing_time_decimal_place)
        
    def run_filter_setup3(self, alpha, beta, kappa):
        x, P, H = self.data.get_initial_data(setup=self.setup)

        ukf = UnscentedKalmanFilter(x=x.copy(), 
                                    P=P.copy(), 
                                    H=H.copy(), 
                                    alpha=alpha, 
                                    beta=beta, 
                                    kappa=kappa)
        
        start = datetime.now()
        error = ukf.run(data=self.data, 
                        setup=self.setup,
                        measurement_type=self.measurement_type)
        end = datetime.now()
        processing_time = (end - start).total_seconds()

        return error, np.round(processing_time / self.data.N, Configs.processing_time_decimal_place)

    def run(self):
        
        processing_times = []
        mae_errors = []
        rmse_errors = []
        max_errors = []
        for alpha in self.alphas:
            mae_errors_ = []
            rmse_errors_ = []
            max_errors_ = []
            processing_times_ = []
            for beta in tqdm(self.betas):
                # error, processing_time = self.run_dummy_filter(base_time=beta)
                # mae_errors_.append(error)
                # rmse_errors_.append(error)
                # max_errors_.append(error)
                # processing_times_.append(processing_time)

                if self.setup is SetupEnum.SETUP_1:
                    error, processing_time = self.run_filter_setup1(alpha=alpha, beta=beta, kappa=self.kappa)
                    
                elif self.setup is SetupEnum.SETUP_2:
                    error, processing_time = self.run_filter_setup2(alpha=alpha, beta=beta, kappa=self.kappa)
                    
                else: # SetupEnum.SETUP_3
                    error, processing_time = self.run_filter_setup3(alpha=alpha, beta=beta, kappa=self.kappa)

                mae_errors_.append(error[ErrorEnum.MAE])
                rmse_errors_.append(error[ErrorEnum.RMSE])
                max_errors_.append(error[ErrorEnum.MAX])
                processing_times_.append(processing_time)
                
            mae_errors.append(mae_errors_)
            rmse_errors.append(rmse_errors_)
            max_errors.append(max_errors_)
            processing_times.append(processing_times_)

        print("Experiment finished.")
        beta_str = [str(int(beta)) for beta in self.betas]
        self.time_df = pd.DataFrame(processing_times, columns=beta_str, index=self.alphas)
        self.mae_error_df = pd.DataFrame(mae_errors, columns=beta_str, index=self.alphas)
        self.rmse_error_df = pd.DataFrame(rmse_errors, columns=beta_str, index=self.alphas)
        self.max_error_df = pd.DataFrame(max_errors, columns=beta_str, index=self.alphas)
        self.dump_df()

    def find_best_combination(self,
                              error_weight, 
                              error_upper_limit=500):
        """
            weighted sum method to find the best combination of parameters.
        """
        error_types = ErrorEnum.get_all()

        processing_times = self.time_df.values

        weight_for_errors = error_weight
        weight_for_time = 1. - error_weight
        max_time = np.max(processing_times)

        print("-"*20)
        for error_type in error_types:
            if error_type is ErrorEnum.MAE:
                errors = self.mae_error_df.values
            elif error_type is ErrorEnum.RMSE:
                errors = self.rmse_error_df.values
            else: # ErrorEnum.MAX
                errors = self.max_error_df.values

            max_error = np.max(errors)
                            
            if max_error > error_upper_limit:
                max_error = error_upper_limit
                
            processing_times_ = np.array(processing_times).flatten()
            errors_ = np.array(errors).flatten()
            errors_[errors_ > error_upper_limit] = error_upper_limit
            
            normalized_times = [time / max_time for time in processing_times_]
            normalized_errors = [error / max_error for error in errors_]
            # Calculate a combined score for each model
            scores = [weight_for_errors * error + weight_for_time * time 
                    for error, time in zip(normalized_errors, normalized_times)]
            
            # Find the index of the best model
            optimal_index = scores.index(min(scores))
            alpha_index = optimal_index // len(self.betas)
            beta_index = optimal_index % len(self.betas)
            print(f"For {error_type.name} error:")
            print(f"Minimum model error: {errors_[optimal_index]:.3f}")
            print(f"Processing time: {processing_times_[optimal_index]:.1f}")
            print(f"Alpha (k={self.kappa}): {self.alphas[alpha_index]}")
            print(f"Beta (k={self.kappa}): {self.betas[beta_index]}")
            print("-"*20)
        
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
        ax1.set(xlabel="Beta", ylabel="Alpha")
        
        sns.heatmap(self.mae_error_df,
                    ax=ax2,
                    cmap="crest",
                    annot=True,
                    linewidths=1,
                    fmt='.3g')
        ax2.set_title("Mean Absolute Error")
        ax2.set(xlabel="Beta", ylabel="Alpha")

        sns.heatmap(self.rmse_error_df,
                    ax=ax3,
                    cmap="crest",
                    annot=True,
                    linewidths=1,
                    fmt='.3g')
        ax3.set_title("Root Mean Squared Error")
        ax3.set(xlabel="Beta", ylabel="Alpha")

        sns.heatmap(self.max_error_df,
                    ax=ax4,
                    cmap="crest",
                    annot=True,
                    linewidths=1,
                    fmt='.3g')
        ax4.set_title("Maximum Error")
        ax4.set(xlabel="Beta", ylabel="Alpha")

        fig.subplots_adjust(wspace=0.2)

    def plot_combined_result(self, df, error_type=ErrorEnum.MAE):
        fig, ax = plt.subplots(nrows=1, figsize=(16, 6))
        fig.subplots_adjust(wspace=0.2)
        
        sns.heatmap(df,
                    ax=ax,
                    cmap="crest",
                    annot=True,
                    fmt='.4g')
        ax.set_title(error_type.name)
        ax.set(xlabel="Beta", ylabel="Alpha")

    def get_error_df(self, error_type=ErrorEnum.MAE):
        if error_type is ErrorEnum.MAE:
            return self.mae_error_df
        elif error_type is ErrorEnum.RMSE:
            return self.rmse_error_df
        
        return self.max_error_df
        
    def dump_df(self):
        vo_dropout_percentage = str(int(self.vo_dropout_ratio * 100))
        gps_dropout_percentage = str(int(self.gps_dropout_ratio * 100))
        # Saving the results
        self.time_df.to_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{vo_dropout_percentage}_{gps_dropout_percentage}_{self.kappa}_time_df.json")

        self.mae_error_df.to_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{vo_dropout_percentage}_{gps_dropout_percentage}_{self.kappa}_mae_error_df.json")
        self.rmse_error_df.to_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{vo_dropout_percentage}_{gps_dropout_percentage}_{self.kappa}_rmse_error_df.json")
        self.max_error_df.to_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{vo_dropout_percentage}_{gps_dropout_percentage}_{self.kappa}_max_error_df.json")

    def load_df(self):
        vo_dropout_percentage = str(int(self.vo_dropout_ratio * 100))
        gps_dropout_percentage = str(int(self.gps_dropout_ratio * 100))
        # Loading results
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.time_df = pd.read_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{vo_dropout_percentage}_{gps_dropout_percentage}_{self.kappa}_time_df.json")
            
            self.mae_error_df = pd.read_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{vo_dropout_percentage}_{gps_dropout_percentage}_{self.kappa}_mae_error_df.json")
            self.rmse_error_df = pd.read_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{vo_dropout_percentage}_{gps_dropout_percentage}_{self.kappa}_rmse_error_df.json")
            self.max_error_df = pd.read_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{vo_dropout_percentage}_{gps_dropout_percentage}_{self.kappa}_max_error_df.json")

        self.time_df.set_index([self.alphas], inplace=True)
        self.mae_error_df.set_index([self.alphas], inplace=True)
        self.rmse_error_df.set_index([self.alphas], inplace=True)
        self.max_error_df.set_index([self.alphas], inplace=True)



if __name__ == '__main__':
    pass