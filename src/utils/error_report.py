import sys
if __name__ == "__main__":
    sys.path.append('../../src')
import numpy as np
from sklearn.metrics import mean_squared_error

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
    t = f"----- {title} -----"
    print("")
    print(t)
    print(f"Mean Absolute Error: {report[ErrorEnum.MAE]} m")
    print(f"Root Mean Squared Error: {report[ErrorEnum.RMSE]} m")
    print(f"Maximum Error: {report[ErrorEnum.MAX]} m")
    print("-" * len(t))
    print("")
    
