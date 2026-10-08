import diimpy.bayesian_inversion as bayes
from diimpy.settings import DEFAULT_SETTINGS_DIR
import torch
import numpy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    output = bayes.main(device='cpu',
                  precision=torch.float64,
                  data_path= DEFAULT_SETTINGS_DIR / 'npy_data',
                  x_data_filename='buossoi_x_data.pt',
                  y_data_filename='buossoi_y_data.pt',
                  dates_filename='dates.pt',
                  N=4000,
                  batch_size=5,
                  lambdas_path=DEFAULT_SETTINGS_DIR / 'lambdas',
                  lambdas_column_name='lambdas',
                  cte_path=DEFAULT_SETTINGS_DIR / 'cte.csv',
                  perturbations_path=DEFAULT_SETTINGS_DIR / 'perturbations_diim.pt',
                  verbose=True,
                  output_path = None,
                  prefix = '',
                  calc_kd = True,
                  calc_bbp = True,
                  num_threads = 60)
    chla = torch.exp(output['chla_hat'][:,0,0,0,0,0] +  (output['chla_hat'][:,0,0,0,0,1]**2)/2)
    print(chla)

    
    
    
