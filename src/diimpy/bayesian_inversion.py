
import numpy as np
import torch
import pandas as pd
import os
from torch.utils.data import DataLoader
import time
import sys
import diimpy.Forward_module as fm
import diimpy.read_data_module as rdm
from torch import nn
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from diimpy.settings import DEFAULT_SETTINGS_DIR
from pathlib import Path
from typing import Union
from os import PathLike

def approximate_jacobian(x,epsilon: float,model_eval,constant,dtype=torch.float64,device='cpu'):
    y_1 = model_eval(x,constant)
    jacobian = torch.zeros((  *(y_1.shape[:-1]) , y_1.shape[-1], x.shape[-1] ),dtype=dtype).to(device)
    
    for i in range(x.shape[-1]):
        x_2 = x.detach().clone()
        x_2[...,i] += epsilon
        y_2 = model_eval(x_2,constant)
        jacobian[...,i] += (y_2 - y_1)/(x_2[...,i] - x[...,i])

        
        x_2 = x.detach().clone()
        x_2[...,i] -= epsilon
        y_2 = model_eval(x_2,constant)
        jacobian[...,i] += (y_1 - y_2)/(x[...,i] - x_2[...,i])

    return jacobian/2

def train_loop(data_i,model,loss_fn,optimizer,N,lat_size=1,lon_size=1,num_days=1,my_device = 'cpu',constant = None,calc_kd=True,calc_bbp=True,verbose=True):
    """
    """
    
    past_pred=torch.empty((N,num_days,lat_size,lon_size,1,1,3)) # (batch_size,lat_size,lon_size,1,lambdas_dim,5)

    criterium = 1
    criterium_2 = 0
    i=0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    X = data_i[0]
    Y = data_i[1]
    s_a = (loss_fn.s_a)
    s_e = (loss_fn.s_e)
    s_e_inverse = (loss_fn.s_e_inverse)
    s_a_inverse = (loss_fn.s_a_inverse)

    dR = (1e-13)
    ls_val=[]
    init_time = time.time()
    consecutive_equal = [0,0]
    while (((criterium >dR ) & (i<N)) or ((criterium_2 < 100)&(i<N))):
        pred = model(X,constant = constant)
        loss = loss_fn(Y,pred,model.state_dict()['chparam'])
        
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        ls_val.append(loss.item())
        past_pred[i] = model.state_dict()['chparam']
        
        if i != 0:
            criterium = ls_val[-2] - ls_val[-1]
        if criterium <=dR:
            criterium_2+=1
            consecutive_equal[0] = consecutive_equal[1]
            consecutive_equal[1] = i
            if consecutive_equal[1] != (consecutive_equal[0]+1) :
                criterium_2 = 0
        if verbose == True:
            if (i+1) % 100 == 0:
                print('loss_value:',ls_val[-1],'for iteration:',i,criterium_2)
                
        i+=1
        scheduler.step(loss)
    last_rrs = pred.detach().to('cpu').clone()
    last_i = i
    
    parameters_eval = (list(model.parameters())[0].clone().detach())

    evaluate_model = fm.evaluate_model_class(model=model,X=X)

    K_x = approximate_jacobian(x=parameters_eval,epsilon=1e-6,model_eval=evaluate_model.model_der,constant=constant,dtype=s_e_inverse.dtype,device = my_device)
    S_hat = torch.einsum('...ia,...ij,...jb->...ab', K_x, s_e_inverse.unsqueeze(-3), K_x) #(Kx^T * S * Kx)_{ab}
    S_hat += s_a_inverse
    S_hat = torch.inverse( S_hat)

    
    X_hat = torch.empty((  *(K_x.shape[:-2]),1,2*K_x.shape[-1] ),dtype=s_e_inverse.dtype).to(my_device)

    X_hat[...,::2] = past_pred[last_i-1].detach().to('cpu').clone()
    d_X_hat = torch.sqrt(torch.diagonal(S_hat,dim1=-2,dim2=-1).clone().detach()).unsqueeze(-2)
    X_hat[...,1::2] = d_X_hat.detach().to('cpu').clone()
    output = {'chla_hat':X_hat,'RRS_hat':last_rrs}

    
    if calc_kd:

        kd_values = evaluate_model.kd_der(parameters_eval,constant)
        kd_derivative = approximate_jacobian(x=parameters_eval,epsilon=1e-6,model_eval=evaluate_model.kd_der,constant=constant,dtype=s_e_inverse.dtype,device=my_device)
        kd_hat = torch.empty((  *(kd_derivative.shape[:-2]),2*kd_derivative.shape[-2] ),dtype=s_e_inverse.dtype).to(my_device)
        kd_delta = fm.error_propagation(kd_derivative,S_hat)
        kd_hat[...,::2] = kd_values.detach().to('cpu').clone()
        kd_hat[...,1::2] = torch.sqrt(kd_delta).detach().to('cpu').clone()
        output['kd_hat'] = kd_hat
    if calc_bbp:

        bbp_values = evaluate_model.bbp_der(parameters_eval,constant)
        bbp_derivative = approximate_jacobian(x=parameters_eval,epsilon=1e-6,model_eval=evaluate_model.bbp_der,constant=constant,dtype=s_e_inverse.dtype,device=my_device)
        bbp_hat = torch.empty((  *(bbp_derivative.shape[:-2]),2*bbp_derivative.shape[-2] ),dtype=s_e_inverse.dtype).to(my_device)
        bbp_delta = fm.error_propagation(bbp_derivative,S_hat)
        bbp_hat[...,::2] = bbp_values.detach().to('cpu').clone()
        bbp_hat[...,1::2] = torch.sqrt(bbp_delta).detach().to('cpu').clone()
        output['bbp_hat'] = bbp_hat
    
    return output    

def main(device='cpu',
         precision=torch.float64,
         data_path=DEFAULT_SETTINGS_DIR / 'npy_data',
         x_data_filename='buossoi_x_data.pt',
         y_data_filename='buossoi_y_data.pt',
         dates_filename='dates.pt',
         y_data_std_filename = None,
         N=500,
         batch_size='all',
         lambdas_path=DEFAULT_SETTINGS_DIR / 'lambdas',
         lambdas_column_name='lambdas',
         cte_path=DEFAULT_SETTINGS_DIR / 'cte.csv',
         perturbations_path=DEFAULT_SETTINGS_DIR / 'perturbations_diim.pt',
         verbose=True,
         output_path = None,
         prefix = '',
         calc_kd = True,
         calc_bbp = True,
         num_threads = None,
         lr = 0.029853826189179603
):
    if num_threads:
        torch.set_num_threads(num_threads)

    data = rdm.customTensorData(data_path=data_path,x_data_filename=x_data_filename,y_data_filename=y_data_filename,dates_filename=dates_filename,device = device).to(device)

    if batch_size == 'all':
        batch_size = len(data)

    

    #batch_size = 3
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    lat_size = data.x_data.shape[1]
    lon_size = data.x_data.shape[2]

    if verbose == True:
        print('data loaded, lat_size: {}, lon_size: {}, dates_size: {}, batch_size: {}, lambdas_size: {}'.format(lat_size,lon_size,data.dates.shape[0],batch_size,data.x_data.shape[4]))



    params = rdm.read_constants(lambdas_path=lambdas_path,lambdas_column_name=lambdas_column_name,cte_path=cte_path,tensor = True,my_device = device,dict=False,precision=precision)
    lambdas = params.lambdas
    len_lambdas = lambdas.shape[-1]
    
    assert data.x_data.shape[4] == len_lambdas, 'inconsistency, input data have {} wavelenghts, while parameters are compute for {} wavelenghts.'.format(data.x_data.shape[4],lambdas.shape[-1])

    s_a = torch.eye(3)*1.13
    x_a = torch.zeros((lat_size,lon_size,1,1,3))
    if y_data_std_filename == None:
        rrs_std = (torch.eye(5)*torch.tensor([1.5e-3,1.2e-3,1e-3,8.6e-4,5.7e-4]))**(2) #validation rmse from https://catalogue.marine.copernicus.eu/documents/QUID/CMEMS-OC-QUID-009-141to144-151to154.pdf
        rrs_std = rrs_std.unsqueeze(-3).unsqueeze(-3).unsqueeze(-3)
        s_e = rrs_std.repeat(batch_size,1,1,1,1).to(precision).to(device)
    else:
        rrs_std = torch.load(data_path + '/' + y_data_std_filename)
        s_e = ((torch.eye(len_lambdas)*rrs_std)**2).to(precision).to(device)

    perturbations = torch.load(perturbations_path).to(precision).to(device)
    params.perturb_parameters(perturbations)

    model = fm.Forward_Model(num_days = batch_size,precision=precision).to(device)
    loss = fm.RRS_loss(x_a,s_a,s_e,num_days=batch_size,my_device = device,precision=precision)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    
    if verbose == True:
        init_time = time.time()
        print('starting minimization loop with {} max iterations'.format(N))
        
    output = train_loop(next(iter(dataloader)),model,loss,optimizer,N=N,lat_size=1,lon_size=1,num_days=batch_size,my_device = device,constant = params,calc_kd=calc_bbp,calc_bbp=calc_bbp,verbose = verbose)

    if verbose == True:
        print('inversion ended, time for inversion:',time.time() - init_time)
        
    if output_path:
        for key in output.keys():
            torch.save(output[key],output_path + '/' + prefix + '_' + key + '.pt')
        if verbose == True:
            print('outputs_saved')
            
    output['x_data'] = data.x_data.detach().to('cpu').clone()
    output['y_data'] = data.y_data.detach().to('cpu').clone()
    output['dates'] = data.dates.detach().to('cpu').clone()
            
    return output
            


if __name__ == '__main__':
    pass

    

    
