#!/usr/bin/env python
import numpy as np
import torch
from torch import nn
import os
import sys
from typing import Union,Optional
from diimpy.read_data_module import AbsorptionParams
import diimpy.Forward_module_v2 as fm
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



@torch.jit.script
def multilayer_parameters(Ediro,Edifo,z,dz,Bs,Bu,Bd,Cs,Cu,cd,Fd):
    """
    """
    #Exponentials evaluated with the dz's, Edir, xminus and yminus evaluated with z = [0,z1,...,zn-1]. Rplus and minus dont requiere z. 
    Rplus,Rminus,Explus,Exminus,xplus,yplus,Edir = fm.third_layer(Bs,Bu,Bd,Cs,Cu,cd,Fd,Ediro,z+dz,dz,flags=True)
    xplus,flags = xplus
    

    if not any(flags):
        xminus,yminus = xplus,yplus
    else:
        xminus,yminus = xplus.clone(),yplus.clone()
        xminus[flags] = (z)[flags]*Fd[flags]
        yminus[flags] = -(z)[flags]*Bd[flags]
        
    N = Bs.shape[:-2] #number of layers
    
    zeta = torch.zeros((Bs.shape))
    eta = torch.zeros((Bs.shape))
    gamma = torch.zeros((Bs.shape))
    theta = torch.zeros((Bs.shape))
    beta = torch.zeros((Bs.shape))

    zeta[...,0,:] = 1
    eta[...,0,:] = Rminus[...,0,:]*Exminus[...,0,:]
    eta[...,1:,:] = -Exminus[...,1:,:]*(1-Rplus[...,:-1,:]*Rminus[...,1:,:])
    gamma[...,1:,:] = -(1 - Rminus[...,1:,:]*Rplus[...,1:,:])

    theta[...,0,:] = Edifo - Ediro*xminus[...,0,:]
    if any(flags[...,0,:]).any():
        theta[...,0,:][flags[...,0,:]] = Edifo
        
    for i in range(N-1):
        w = (Explus[...,i,:]*(1-Rplus[...,i,:]*Rminus[...,i+1,:]))/zeta[...,i,:]
        beta[...,i,:], delta[...,i,:] = (Rminus[...,i,:]-Rminus[...,i+1,:]) - w * eta[...,i] , \
            (xminus[...,i+1,:] - xplus[...,i,:] - (yminus[...,i+1,:] - yplus[...,i,:])*Rminus[...,i+1,:])*Edir[...,i,:] - w * theta[...,i,:]
        
        w =  (1-Rminus[...,i,:]*Rplus[...,i,:])/beta[...,i,:]
        zeta[...,i+1,:],theta[...,i+1,:] = (Rplus[...,i,:]-Rplus[...,i+1,:]) - w*gamma[...,i,:],\
            (yminus[...,i+1,:] - yplus[...,i,:] - (xminus[...,i+1,:] - xplus[...,i,:])*Rplus[...,i,:])*Edir[...,i,:] - w * delta[...,i,:]

    
    cplus = torch.zeros((Bs.shape))
    cminus = torch.zeros((Bs.shape))

    cplus[...,-1,:],cminus[...,-1,:] = theta[...,-1,:]/zeta[...,-1,:],0
    for j in range(N-2,-1,-1):
        cminus[...,j,:] = (delta[...,j] + (1-Rplus[...,j+1,:]*Rminus[...,j+1,:])*cplus[...,j+1,:])/beta[...,j,:]
        cplus[...,j,:] = (theta[...,j] - eta[...,j,:]*cminus[...,j,:])/beta[...,j,:]
    
    return Rplus,Rminus,Explus,Exminus,xplus,yplus,Edir,cplus,cminus

def Edif_multilayer(Rminus,Explus,xplus,Edir,cplus,cminus):
    return cplus * Explus + cminus * Rminus + xplus * Edir

def Eu_multilayer(Rplus,Explus,yplus,Edir,cplus,cminus):
    return cplus * Rplus * Explus + cminus + yplus * Edir
    
    
