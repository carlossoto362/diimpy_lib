#!/usr/bin/env python

import torch
import numpy as np
from scipy import stats
import os
import sys
import diimpy.read_bbc as rb
import pandas as pd
from numba import njit

from diimpy.settings import DEFAULT_SETTINGS_DIR
from pathlib import Path
from typing import Union
from os import PathLike


@torch.jit.script
class AbsorptionParams:
    def __init__(self,
                 absortion_PH: torch.Tensor,
                 linear_regression_slope_s: torch.Tensor,
                 linear_regression_intercept_s: torch.Tensor,
                 linear_regression_slope_b: torch.Tensor,
                 linear_regression_intercept_b: torch.Tensor,
                 dCDOM: torch.Tensor,
                 sCDOM: torch.Tensor,
                 Qa: torch.Tensor,
                 Qb: torch.Tensor,
                 Theta_min: torch.Tensor,
                 Theta_o: torch.Tensor,
                 beta: torch.Tensor,
                 sigma: torch.Tensor,
                 dNAP: torch.Tensor,
                 sNAP: torch.Tensor,
                 absortion_w: torch.Tensor,
                 eNAP: torch.Tensor,
                 fNAP: torch.Tensor,
                 br_nap: torch.Tensor,
                 backscattering_w: torch.Tensor,
                 scattering_w: torch.Tensor,
                 vs: torch.Tensor,
                 rs: torch.Tensor,
                 vu: torch.Tensor,
                 ru: torch.Tensor,
                 rd: torch.Tensor,
                 T: torch.Tensor,
                 gammaQ: torch.Tensor,
                 lambdas: torch.Tensor
                 ):
        
        self.original_parameters = [absortion_PH,
            linear_regression_slope_s,
            linear_regression_intercept_s,
            linear_regression_slope_b,
            linear_regression_intercept_b,
            dCDOM,
            sCDOM,
            Qa,
            Qb,
            Theta_min,
            Theta_o,
            beta,
            sigma,
            br_nap,
            dNAP,
            sNAP,
            absortion_w,
            eNAP,
            fNAP,
            backscattering_w,
            scattering_w,
            vs,
            rs,
            vu,
            ru,
            rd,
            T,
            gammaQ
        ]
        
        self.all_parameters = [parameter.clone() for parameter in self.original_parameters]
        
        self.absortion_PH = self.all_parameters[0]
        self.linear_regression_slope_s = self.all_parameters[1]
        self.linear_regression_intercept_s = self.all_parameters[2]
        self.linear_regression_slope_b = self.all_parameters[3]
        self.linear_regression_intercept_b = self.all_parameters[4]
        self.dCDOM = self.all_parameters[5]
        self.sCDOM = self.all_parameters[6]
        self.Qa = self.all_parameters[7]
        self.Qb = self.all_parameters[8]
        self.Theta_min = self.all_parameters[9]
        self.Theta_o = self.all_parameters[10]
        self.beta = self.all_parameters[11]
        self.sigma = self.all_parameters[12]
        self.br_nap = self.all_parameters[13]
        self.dNAP = self.all_parameters[14]
        self.sNAP = self.all_parameters[15]
        self.absortion_w = self.all_parameters[16]
        self.eNAP = self.all_parameters[17]
        self.fNAP = self.all_parameters[18]
        self.backscattering_w = self.all_parameters[19]
        self.scattering_w = self.all_parameters[20]
        self.vs = self.all_parameters[21]
        self.rs = self.all_parameters[22]
        self.vu = self.all_parameters[23]
        self.ru = self.all_parameters[24]
        self.rd = self.all_parameters[25]
        self.T = self.all_parameters[26]
        self.gammaQ = self.all_parameters[27]
        
        self.lambdas = lambdas
        
    def perturb_parameters(self,perturbation_factors: torch.Tensor):
        len_ = len(perturbation_factors)
        for i in range(len_):
            self.all_parameters[i][0] = self.original_parameters[i][0] * perturbation_factors[i]




class customTensorData(torch.nn.Module):

    def __init__(self,data_path: Union[PathLike , None] = None,x_data_filename: str = 'x_data.pt',y_data_filename: str = 'y_data.pt',dates_filename: str = 'dates.pt',precision: torch.dtype = torch.float32 , device: Union[ torch.device , str] = 'cpu'):

        if data_path is None:
            data_path = DEFAULT_SETTINGS_DIR / "npy_data"
        else:
            data_path = Path(data_path)

        
        super().__init__()
        self.x_data = torch.load(data_path / x_data_filename,weights_only=True).to(precision).to(device)
        self.dates = torch.load(data_path / dates_filename,weights_only=True).to(precision).to(device)
        self.y_data = torch.load(data_path / y_data_filename,weights_only=True).to(precision).to(device)
        self.device = device
        self.precision = precision

    def __len__(self):
        return len(self.dates)
        
    def __getitem__(self, idx):

        return self.x_data[idx],self.y_data[idx]


def fill_params(constant,device,precision):
    params = AbsorptionParams(
        
        absortion_PH = constant['absortion_PH'].unsqueeze(0).to(precision).to(device),
        linear_regression_slope_s = torch.tensor([constant['linear_regression_slope_s']],dtype=precision).unsqueeze(0).to(device),
        linear_regression_intercept_s = torch.tensor([constant['linear_regression_intercept_s']],dtype=precision).unsqueeze(0).to(device),
        linear_regression_slope_b = torch.tensor([constant['linear_regression_slope_b']],dtype=precision).unsqueeze(0).to(device),
        linear_regression_intercept_b = torch.tensor([constant['linear_regression_intercept_b']],dtype=precision).unsqueeze(0).to(device),
        dCDOM = torch.tensor([constant['dCDOM']],dtype=precision).unsqueeze(0).to(device),
        sCDOM = torch.tensor([constant['sCDOM']],dtype=precision).unsqueeze(0).to(device),
        Qa = torch.tensor([constant['Qa']],dtype=precision).unsqueeze(0).to(device),
        Qb = torch.tensor([constant['Qb']],dtype=precision).unsqueeze(0).to(device),
        Theta_min = torch.tensor([constant['Theta_min']],dtype=precision).unsqueeze(0).to(device),
        Theta_o = torch.tensor([constant['Theta_o']],dtype=precision).unsqueeze(0).to(device),
        beta = torch.tensor([constant['beta']],dtype=precision).unsqueeze(0).to(device),
        sigma = torch.tensor([constant['sigma']],dtype=precision).unsqueeze(0).to(device),
        dNAP = torch.tensor([constant['dNAP']],dtype=precision).unsqueeze(0).to(device),
        sNAP = torch.tensor([constant['sNAP']],dtype=precision).unsqueeze(0).to(device),
        absortion_w = constant['absortion_w'].unsqueeze(0).unsqueeze(0).to(precision).to(device),
        eNAP = torch.tensor([constant['eNAP']],dtype=precision).unsqueeze(0).to(device),
        fNAP = torch.tensor([constant['fNAP']],dtype=precision).unsqueeze(0).to(device),
        br_nap = torch.tensor([constant['br_nap']],dtype=precision).unsqueeze(0).to(device),
        backscattering_w = constant['backscattering_w'].unsqueeze(0).to(precision).to(device),
        scattering_w = constant['scattering_w'].unsqueeze(0).to(precision).to(device),
        vs = torch.tensor([constant['vs']],dtype=precision).unsqueeze(0).to(device),
        rs = torch.tensor([constant['rs']],dtype=precision).unsqueeze(0).to(device),
        vu = torch.tensor([constant['vu']],dtype=precision).unsqueeze(0).to(device),
        ru = torch.tensor([constant['ru']],dtype=precision).unsqueeze(0).to(device),
        rd = torch.tensor([constant['rd']],dtype=precision).unsqueeze(0).to(device),
        T = torch.tensor([constant['T']],dtype=precision).unsqueeze(0).to(device),
        gammaQ = torch.tensor([constant['gammaQ']],dtype=precision).unsqueeze(0).to(device),
        lambdas = torch.tensor(constant['lambdas'],dtype=precision).unsqueeze(0).to(device)
        )
    return params

def read_constants(lambdas_path: Union[ PathLike , None] = None,lambdas_column_name: str = 'lambdas',cte_path: Union[PathLike , None] = None,tensor: bool = True,my_device: Union[ torch.device , str ] = 'cpu',dict: bool = False,precision: torch.dtype = torch.float32):

    if lambdas_path is None:
        lambdas_path = DEFAULT_SETTINGS_DIR / 'lambdas'
        cte_path = DEFAULT_SETTINGS_DIR / 'cte.csv'
    else:
        settings_dir = Path(lambdas_path)
        cte_path = Path(cte_path)


    cts_lambda = rb.read_cte_lambda(lambdas_path=lambdas_path,lambdas_column_name=lambdas_column_name,print_values=False)
    constant = {}

    for key in list(cts_lambda.keys())[1:]:
        if tensor == True:
            constant[key] = torch.tensor(cts_lambda[key],dtype=precision)
        else:
            constant[key] = cts_lambda[key]
            
    cts = pd.read_csv(cte_path)
        
    for i in range(len(cts['name'])):
        constant[cts['name'].iloc[i]] = cts['value'].iloc[i]
        
    lambdas = np.array(cts_lambda['lambda']).astype(float)
    constant['lambdas'] = lambdas

    linear_regression=stats.linregress(lambdas,constant['scattering_PH'])
    linear_regression_slope = linear_regression.slope
    linear_regression_intercept = linear_regression.intercept
    constant['linear_regression_slope_s'] = linear_regression_slope

    constant['linear_regression_intercept_s'] = linear_regression_intercept

    linear_regression=stats.linregress(lambdas,constant['backscattering_PH'])
    linear_regression_slope = linear_regression.slope
    linear_regression_intercept = linear_regression.intercept
    constant['linear_regression_slope_b'] = linear_regression_slope
    constant['linear_regression_intercept_b'] = linear_regression_intercept
    if dict:
        return constant
    else:
        constant = fill_params(constant,my_device,precision)
        return constant

if __name__ == '__main__':
        
    pass
