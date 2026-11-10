from diimpy.settings import DEFAULT_SETTINGS_DIR
from pathlib import Path
from typing import Union
from os import PathLike
import torch
import scipy
import numpy as np
    
from diimpy.read_data_module import read_constants
import diimpy.Forward_module_v2 as fm
import diimpy.Forward_Multilayer as mult
import time

if __name__ == "__main__":
    
    data_path = DEFAULT_SETTINGS_DIR / "npy_data"
    
    batch_size = 1
    lat_size = 1
    lon_size = 1
    lambdas_dim = 5
    
    N = 6 #number of layers
    H = 1 #maximum depth
    dz = torch.zeros(H)+0.4 
    z = torch.linspace(0.1,H,steps=N+1) 
    dz = z[1:] - z[:-1]#depth of each layer 
    z = z[:-1]#depth of the top part of each layer
    z = z.reshape((1,1,1,N,1))
    dz = dz.reshape((1,1,1,N,1))

    #a following the beta distribution
    a = torch.tensor(scipy.stats.beta.ppf(np.linspace(0.1,1,N), 2, 5)).reshape((1,1,1,N,1))
    b = a*0.8
    bb = b*0.05

    a = torch.ones((1,1,1,N,lambdas_dim))*a #batch_size,lat_size,lon_size,depth_size,lambdas_size
    b = torch.ones((1,1,1,N,lambdas_dim))*b
    bb = torch.ones((1,1,1,N,lambdas_dim))*bb
    zenith = torch.tensor(torch.pi/2)

    Ediro = torch.ones((1,1,1,1,5))
    Edifo = torch.ones((1,1,1,1,5))
    
    params = read_constants(lambdas_path= data_path / '../lambdas',lambdas_column_name='lambdas',cte_path = data_path / '../cte.csv',tensor = True,my_device = 'cpu',dict=False,precision=torch.float32)
    perturbations = torch.load(DEFAULT_SETTINGS_DIR / 'perturbations_diim.pt')
    params.perturb_parameters(perturbations)
    
    
    Bs,Bu,Bd,Cs,Cu,cd,Fd = fm.second_layer(a,b,bb,zenith,params)
    Rplus,Rminus,Explus,Exminus,xplus,xminus,yplus,yminus,Edir,cplus,cminus = mult.multilayer_parameters(Ediro,Edifo,z,dz,Bs,Bu,Bd,Cs,Cu,cd,Fd)
    Edif = mult.Edif_multilayer(Rminus,Explus,xplus,Edir,cplus,cminus,Edifo)
    Eu = mult.Eu_multilayer(Rplus,Explus,yplus,Edir,cplus,cminus,Ediro,Exminus[...,0,:],yminus[...,0,:])
    print(Eu)
    Rplus,Rminus,Explus,Exminus,xplus,xminus,yplus,yminus,Edir,cplus,cminus = mult.multilayer_parameters_reflectance(Ediro,Edifo,z,dz,Bs,Bu,Bd,Cs,Cu,cd,Fd)
    Edif = mult.Edif_multilayer(Rminus,Explus,xplus,Edir,cplus,cminus,Edifo)
    Eu = mult.Eu_multilayer(Rplus,Explus,yplus,Edir,cplus,cminus,Ediro,Exminus[...,0,:],yminus[...,0,:])
    print(Eu)
    
