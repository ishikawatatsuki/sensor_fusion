import sys
if __name__ == "__main__":
    sys.path.append('../../src')
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

from configs import ErrorEnum, Configs

def get_error_report(ground_truth, estimated_result):
    
    absolute_errors = np.absolute(np.subtract(ground_truth, estimated_result))

    return {
        ErrorEnum.MAE: np.round(np.mean(absolute_errors), Configs.decimal_place),
        ErrorEnum.RMSE: np.round(np.sqrt(mean_squared_error(ground_truth, estimated_result)), Configs.decimal_place),
        ErrorEnum.MAX: np.round(np.max(absolute_errors), Configs.decimal_place)
    }

def get_error_from_list(errors, e_type=ErrorEnum.MAE):
    return [error[e_type] for error in errors]

def get_dummy_error_report():
    n = (np.random.random() * 100) ** (1/2)
    return {
        ErrorEnum.MAE: n,
        ErrorEnum.RMSE: n,
        ErrorEnum.MAX: n + np.random.randint(1, 10)
    }
    
def print_error_report(report, title):
    print(f"----- {title} -----")
    print(f"Mean Absolute Error: {report[ErrorEnum.MAE]}")
    print(f"Root Mean Squared Error: {report[ErrorEnum.RMSE]}")
    print(f"Maximum Error: {report[ErrorEnum.MAX]}")
    print("")