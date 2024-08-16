import sys
if __name__ == "__main__":
    sys.path.append('../../src')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import sys
from tqdm import tqdm
from configs import SetupEnum, MeasurementDataEnum, SamplingEnum, Configs, ErrorEnum, FilterEnum, NoiseTypeEnum
from kalman_filters.particle_filter import ResamplingAlgorithms, ParticleFilter


class ParticleFilterParameterTuner:

    algorithm_str = {
        ResamplingAlgorithms.MULTINOMIAL: "MULTINOMIAL", 
        ResamplingAlgorithms.RESIDUAL: "RESIDUAL", 
        ResamplingAlgorithms.STRATIFIED: "STRATIFIED", 
        ResamplingAlgorithms.SYSTEMATIC: "SYSTEMATIC"
    }

    time_df = None
    mae_error_df = None
    rmse_error_df = None
    max_error_df = None
    
    def __init__(
        self, 
        setup,
        params, 
        data,
        kitti_drive,
        file_export_path
        ):

        self.setup = setup
        self.file_export_path = file_export_path
        self.n_samples = params["n_samples"]
        self.algorithms = params["algorithms"]
        self.vo_dropout_ratio = data.vo_dropout_ratio
        self.gps_dropout_ratio = data.gps_dropout_ratio
        
        print(params)
        print(self.vo_dropout_ratio)
        print(self.gps_dropout_ratio)
        self.importance_resampling = True
        self.kitti_drive = kitti_drive
        self.data = data
    
    def change_data_sampling(self, sampling=SamplingEnum.DEFAULT_DATA, upsampling_factor=10, downsampling_ratio=0.1):
        if sampling is SamplingEnum.UPSAMPLED_DATA:
            self.data.set_upsampling_factor(upsampling_factor)
        elif sampling is SamplingEnum.DOWNSAMPLED_DATA:
            self.data.set_downsampling_ratio(downsampling_ratio)
        
        self.data.set_data_sampling(sampling=sampling)
        
    def run_dummy_filter(self, base_time):
        return {
                ErrorEnum.MAE: 100.,
                ErrorEnum.RMSE: 100.,
                ErrorEnum.MAX: 1000.,
            }, base_time + np.random.randint(0, 30)
    
    def run_filter(self, algorithm, N):
        x, P, H, q, r_vo, r_gps = self.data.get_initial_data(
            setup=self.setup, 
            filter_type=FilterEnum.PF,
            noise_type=NoiseTypeEnum.CURRENT
        )
        pf = ParticleFilter(
            N=N, 
            x_dim=x.shape[0], 
            H=H.copy(), 
            q=q,
            r_vo=r_vo,
            r_gps=r_gps,
            setup=self.setup,
            resampling_algorithm=algorithm
        )
        pf.create_gaussian_particles(mean=x.copy(), var=P.copy())

        start = datetime.now()
        error = pf.run(
            data=self.data, 
            importance_resampling=self.importance_resampling,
            measurement_type=MeasurementDataEnum.DROPOUT, 
            debug_mode=True,
            show_graph=False,
        )
        end = datetime.now()
        processing_time = (end - start).total_seconds()
        return error, np.round(processing_time / self.data.N, Configs.processing_time_decimal_place)
    
    def run(self):
        mae_errors = []
        rmse_errors = []
        max_errors = []
        processing_times = []
        for algorithm in self.algorithms:
            print(f"Resampled by: {self.algorithm_str[algorithm]}")
            mae_errors_ = []
            rmse_errors_ = []
            max_errors_ = []
            processing_times_ = []
            for N in self.n_samples:
                # error, processing_time = self.run_dummy_filter(base_time=N)
                # mae_errors_.append(error)
                # rmse_errors_.append(error)
                # max_errors_.append(error)
                # processing_times_.append(processing_time)
                
                error, processing_time = self.run_filter(algorithm=algorithm, N=N)
                
                mae_errors_.append(error[ErrorEnum.MAE])
                rmse_errors_.append(error[ErrorEnum.RMSE])
                max_errors_.append(error[ErrorEnum.MAX])
                processing_times_.append(processing_time)
            mae_errors.append(mae_errors_)
            rmse_errors.append(rmse_errors_)
            max_errors.append(max_errors_)
            processing_times.append(processing_times_)

        print("Experiment finished.")
        algorithm_list = [self.algorithm_str[al] for al in self.algorithms]
        self.time_df = pd.DataFrame(processing_times, columns=self.n_samples, index=algorithm_list)
        self.mae_error_df = pd.DataFrame(mae_errors, columns=self.n_samples, index=algorithm_list)
        self.rmse_error_df = pd.DataFrame(rmse_errors, columns=self.n_samples, index=algorithm_list)
        self.max_error_df = pd.DataFrame(max_errors, columns=self.n_samples, index=algorithm_list)
        self.dump_df()

    def find_best_combination(
        self, 
        error_weight, 
        error_upper_limit=500):
        """
            weighted sum method to find the best combination of parameters.
        """
        error_types = ErrorEnum.get_all()
        processing_times = self.time_df.values
        weight_for_errors = error_weight
        weight_for_time = 1. - error_weight
        optimal_params = {}
        optimal_params_list = []
        for error_type in error_types:

            if error_type is ErrorEnum.MAE:
                errors = self.mae_error_df.values
            elif error_type is ErrorEnum.RMSE:
                errors = self.rmse_error_df.values
            else: # ErrorEnum.MAX
                errors = self.max_error_df.values


            max_time = np.max(processing_times)
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
            algorithm_index = optimal_index // len(self.n_samples)
            n_sample_index = optimal_index % len(self.n_samples)

            optimal_params[error_type.value] = {
                'n_sample': self.n_samples[n_sample_index],
                'resampling_algorithm': self.algorithms[algorithm_index]
            }
            optimal_params_list.append([self.n_samples[n_sample_index], self.algorithms[algorithm_index].name])

        df = pd.DataFrame(np.array(optimal_params_list), 
                    index=ErrorEnum.get_names(), 
                    columns=["# of samples", "resampling algorithm"])
        
        return optimal_params, df
        
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
        ax1.set(xlabel="# of samples", ylabel="Resampling algorithms")
        
        sns.heatmap(self.mae_error_df,
                    ax=ax2,
                    cmap="crest",
                    annot=True,
                    linewidths=1,
                    fmt='.3g')
        ax2.set_title("Mean Absolute Error")
        ax2.set(xlabel="# of samples", ylabel="Resampling algorithms")

        sns.heatmap(self.rmse_error_df,
                    ax=ax3,
                    cmap="crest",
                    annot=True,
                    linewidths=1,
                    fmt='.3g')
        ax3.set_title("Root Mean Squared Error")
        ax3.set(xlabel="# of samples", ylabel="Resampling algorithms")

        sns.heatmap(self.max_error_df,
                    ax=ax4,
                    cmap="crest",
                    annot=True,
                    linewidths=1,
                    fmt='.3g')
        ax4.set_title("Maximum Error")
        ax4.set(xlabel="# of samples", ylabel="Resampling algorithms")

        fig.subplots_adjust(wspace=0.2)

    def dump_df(self):
        # saving the results
        self.time_df.to_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{str(self.vo_dropout_ratio)}_{str(self.gps_dropout_ratio)}_time_df.json")

        self.mae_error_df.to_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{str(self.vo_dropout_ratio)}_{str(self.gps_dropout_ratio)}_mae_error_df.json")
        self.rmse_error_df.to_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{str(self.vo_dropout_ratio)}_{str(self.gps_dropout_ratio)}_rmse_error_df.json")
        self.max_error_df.to_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{str(self.vo_dropout_ratio)}_{str(self.gps_dropout_ratio)}_max_error_df.json")

    def load_df(self):
        self.time_df = pd.read_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{str(self.vo_dropout_ratio)}_{str(self.gps_dropout_ratio)}_time_df.json")

        self.mae_error_df = pd.read_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{str(self.vo_dropout_ratio)}_{str(self.gps_dropout_ratio)}_mae_error_df.json")
        self.rmse_error_df = pd.read_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{str(self.vo_dropout_ratio)}_{str(self.gps_dropout_ratio)}_rmse_error_df.json")
        self.max_error_df = pd.read_json(f"{self.file_export_path}/{str(self.setup.value)}/{self.kitti_drive}/{str(self.vo_dropout_ratio)}_{str(self.gps_dropout_ratio)}_max_error_df.json")

if __name__ == '__main__':
    pass