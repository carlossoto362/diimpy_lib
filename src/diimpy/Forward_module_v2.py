#!/usr/bin/env pythonA
import torch
from torch import nn
import os
import sys
from typing import Union,Optional
from diimpy.read_data_module import AbsorptionParams
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
###################################################################################################################################################################################################
############################################################################FUNCTIONS NEEDED TO DEFINE THE FORWARD MODEL###########################################################################
###################################################################################################################################################################################################

################Functions for the absortion coefitient####################
@torch.jit.script
def absorption_CDOM(lambda_,params: AbsorptionParams):
    """
    Function that returns the mass-specific absorption coefficient of CDOM, function dependent of the wavelength lambda. 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return params.dCDOM*torch.exp(-(params.sCDOM)*(lambda_ - 450.))

@torch.jit.script
def absorption_NAP(lambda_,params: AbsorptionParams):
    """
    Mass specific absorption coefficient of NAP.
    See Gallegos et al., 2011.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return params.dNAP*torch.exp(-params.sNAP*(lambda_ - 440.))

@torch.jit.script
def absorption(lambda_,chla,NAP,CDOM,params: AbsorptionParams):
    """
    Total absortion coeffitient.
    aW,λ (values used from Pope and Fry, 1997), aP H,λ (values averaged and interpolated from
    Alvarez et al., 2022).
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return params.absortion_w + (params.absortion_PH )*chla + \
                (absorption_CDOM(lambda_,params))*CDOM + absorption_NAP(lambda_,params)*NAP
   
    
##############Functions for the scattering coefitient########################

@torch.jit.script
def Carbon(chla,PAR,params: AbsorptionParams):
    """
    defined from the carbon to Chl-a ratio. 
    theta_o, sigma, beta, and theta_min constants (equation and values computed from Cloern et al., 1995), and PAR
    the Photosynthetically available radiation, obtained from the OASIM model, see Lazzari et al., 2021.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    nominator = chla
    beta =  params.beta
    sigma = params.sigma
    exponent = -(PAR - beta)/sigma
    denominator = params.Theta_o*(torch.exp(exponent)/(1+torch.exp(exponent))) + params.Theta_min
    return nominator/denominator

@torch.jit.script
def scattering_ph(lambda_,params: AbsorptionParams):
    """
    The scattering_ph is defined initially as a linear regression between the diferent scattering_ph for each lambda, and then, I
    change the slope and the intercept gradually. 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    
    return params.linear_regression_slope_s *lambda_ + params.linear_regression_intercept_s

@torch.jit.script
def backscattering_ph(lambda_,params: AbsorptionParams):
    """
    The scattering_ph is defined initially as a linear regression between the diferent scattering_ph for each lambda, and then, I
    change the slope and the intercept gradually. 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    
    return params.linear_regression_slope_b *lambda_ + params.linear_regression_intercept_b

@torch.jit.script
def scattering_NAP(lambda_,params: AbsorptionParams):
    """
    NAP mass-specific scattering coefficient.
    eNAP and fNAP constants (equation and values used from Gallegos et al., 2011)
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return params.eNAP*(550./lambda_)**params.fNAP
    
@torch.jit.script
def scattering(lambda_,PAR,chla,NAP,params: AbsorptionParams):
    """
    Total scattering coefficient.
    bW,λ (values interpolated from Smith and Baker, 1981,), bP H,λ (values used
    from Dutkiewicz et al., 2015)
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return params.scattering_w + scattering_ph(lambda_,params) * \
            Carbon(chla,PAR,params) + scattering_NAP(lambda_,params) * NAP

    
#################Functions for the backscattering coefitient#############

@torch.jit.script
def backscattering(lambda_,PAR,chla,NAP,params: AbsorptionParams):
    """
    Total backscattering coefficient.
     Gallegos et al., 2011.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return params.backscattering_w + backscattering_ph(lambda_,params) * \
                Carbon(chla,PAR,params) + params.br_nap * scattering_NAP(lambda_,params) * NAP


@torch.jit.script
def bbp(lambda_,PAR,chla,NAP,CDOM,params: AbsorptionParams):
    """
    """
    return backscattering_ph(lambda_,params) * \
                Carbon(chla,PAR,params) + params.br_nap * scattering_NAP(lambda_,params) * NAP

@torch.jit.script
def a_b_bb(lambda_,PAR,chla,NAP,CDOM,params: AbsorptionParams):
    """
    """
    return absorption(lambda_,chla,NAP,CDOM,params),scattering(lambda_,PAR,chla,NAP,params),backscattering(lambda_,PAR,chla,NAP,params)

###############Functions for the end solution of the equations###########
#The final result is written in terms of these functions, all the functions are functions of absorption, scattering and backscattering only. 

@torch.jit.script
def c_d(a,b, zenith): 
    """
    """
    return (a + b)/torch.cos(zenith*3.1416/180)


@torch.jit.script
def F_d(b,bb, zenith, params: AbsorptionParams):
    """
    """
    return (b - params.rd * bb)/torch.cos(zenith*3.1416/180.)

@torch.jit.script
def B_d(bb,zenith,params: AbsorptionParams):
    """
    """
    return  params.rd*bb/torch.cos(zenith*3.1416/180) 

@torch.jit.script
def C_s(a, bb, params: AbsorptionParams):
    """
    """
    return (a + params.rs * bb )/params.vs

@torch.jit.script
def B_u(bb, params: AbsorptionParams):
    """
    """
    return (params.ru * bb)/params.vu

@torch.jit.script
def B_s(bb, params: AbsorptionParams):
    """
    """
    return (params.rs * bb)/params.vs

@torch.jit.script
def C_u(a, bb, params: AbsorptionParams):
    """
    """
    return (a + params.ru * bb)/ params.vu

@torch.jit.script
def second_layer(a,b,bb,zenith,params:AbsorptionParams):
    """
    """
    return B_s(bb, params),B_u(bb, params),B_d(bb,zenith, params),\
        C_s(a, bb, params),C_u(a, bb, params),c_d(a,b, zenith),F_d(b,bb, zenith, params)
        

#######################THIRD LAYER########################################
##########################################################################

@torch.jit.script
def D_(Bs,Bu,Cs,Cu,flags=False):
    """
    """
    return (0.5) * (Cs + Cu + ( (Cs + Cu)**2 - 4. * Bs * Bu )**(0.5))

@torch.jit.script
def x_(Bs,Bu,Bd,Cs,Cu,cd,Fd,z=torch.tensor(0),flags:bool=False):
    """
    """
    denominator = (cd - Cs) * (cd + Cu) + Bs * Bu
    nominator = -(Cu + cd) * Fd - Bu * Bd
    flag = (denominator == 0)
    if any(flag):
        denominator[denominator==0] = 1
        nominator[denominator == 0] = (z*Fd)[denominator==0]
    if flags == True:
        return [nominator/denominator,flag]

    return nominator/denominator

@torch.jit.script
def y_(Bs,Bu,Bd,Cs,Cu,cd,Fd,z=torch.tensor(0),flags:bool=False):
    """
    """
    denominator = (cd - Cs) * (cd + Cu) + Bs * Bu
    nominator = -Bs * Fd + (cd - Cs) * Bd
    flag = (denominator == 0)
    if any(flag):
        denominator[denominator==0] = 1
        nominator[denominator == 0] = (-z[denominator==0]*Bd[denominator==0])
    if flags == True:
        return [nominator/denominator,flag]
    return nominator/denominator

@torch.jit.script
def R_plus(Bs,Bu,Cs,Cu):
    """
    """
    return Bs / D_(Bs,Bu,Cs,Cu)

@torch.jit.script
def R_minus(Bs,Bu,Cs,Cu):
    """
    """
    return Bu / D_(Bs,Bu,Cs,Cu)


@torch.jit.script
def Exp_plus(Bs,Bu,Cs,Cu,dz=torch.tensor(0)):
    """
    """
    if dz == 0:
        return torch.tensor(1)
    return torch.exp(- (D_(Bs,Bu,Cs,Cu) - Cu) * dz)

@torch.jit.script
def Exp_minus(Bs,Bu,Cs,Cu,dz=torch.tensor(0)):
    """
    """
    if dz == 0:
        return torch.tensor(1)
    return torch.exp(- (D_(Bs,Bu,Cs,Cu) - Cs) * dz)

@torch.jit.script
def E_dir(cd, Ediro, z=torch.tensor(0)):
    """
    """
    return Ediro*torch.exp(-z * cd)

@torch.jit.script
def third_layer(Bs,Bu,Bd,Cs,Cu,cd,Fd,Ediro,z=torch.tensor(0),dz=torch.tensor(0),flags:bool=False):
    """
    """
    Rplus,Rminus = R_plus(Bs,Bu,Cs,Cu),R_minus(Bs,Bu,Cs,Cu)
    Explus,Exminus = Exp_plus(Bs,Bu,Cs,Cu,dz),Exp_minus(Bs,Bu,Cs,Cu,dz)
    y,Edir = y_(Bs,Bu,Bd,Cs,Cu,cd,Fd,z),E_dir(cd, Ediro, z)
    x = x_(Bs,Bu,Bd,Cs,Cu,cd,Fd,z,flags=flags)
    return Rplus,Rminus,Explus,Exminus,x,y,Edir



##################################################################################
##################################################################################
@torch.jit.script
def three_layers_composition(Ediro,lambda_,zenith,PAR,chla,NAP,CDOM,params: AbsorptionParams,z=torch.tensor(0),dz=torch.tensor(0),flags:bool=False):
    a,b,bb = a_b_bb(lambda_,PAR,chla,NAP,CDOM,params)
    Bs,Bu,Bd,Cs,Cu,cd,Fd = second_layer(a,b,bb,zenith,params)
    Rplus,Rminus,Explus,Exminus,x,y,Edir = third_layer(Bs,Bu,Bd,Cs,Cu,cd,Fd,Ediro,z,dz,flags)
    return Rplus,Rminus,Explus,Exminus,x,y,Edir


#######################PARTICULAR SOLUTION########################################
#######################INFINIT LAYER##############################################

@torch.jit.script
def C_plus(x, Ediro, Edifo):
    """
    """
    return Edifo - x * Ediro

@torch.jit.script
def E_u(Cplus,y,Explus,Rplus,Edir):
    """
    """
    return Cplus * Rplus * Explus  + y * Edir

@torch.jit.script
def E_dif(Cplus,x,Explus,Edir):
    """
    """
    return Cplus*Explus + x * Edir
        

    

                                                  

##########################from the bio-optical model to RRS(Remote Sensing Reflectance)##############################

@torch.jit.script
def Q_rs(zenith,params: AbsorptionParams):
    """
    Empirical result for the Radiance distribution function, 
    equation from Aas and Højerslev, 1999, 
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return (params.Qa)*torch.exp(-(params.Qb)*torch.sin((3.1416/180.0)*(90.0-zenith)))

@torch.jit.script
def Rrs_minus(Rrs,params: AbsorptionParams):
    """
    Empirical solution for the effect of the interface Atmosphere-sea.
     Lee et al., 2002
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return Rrs/(params.T+params.gammaQ*Rrs)

@torch.jit.script
def Rrs_plus(Rrs,params: AbsorptionParams):
    """
    Empirical solution for the effect of the interface Atmosphere-sea.
     Lee et al., 2002
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    return Rrs*params.T/(1-params.gammaQ*Rrs)


########################Compositions#############################

@torch.jit.script
def kd(Ediro,Edifo,lambda_,zenith,PAR,chla,NAP,CDOM,params: AbsorptionParams,z=torch.tensor(0)):
    """
    """
    Rplus,Rminus,Explus,Exminus,x,y,Edir = three_layers_composition(Ediro,lambda_,zenith,PAR,chla,NAP,CDOM,params,z)
    Edif = E_dif(C_plus(x, Ediro, Edifo),x,Explus,Edir)
    return (z**-1)*torch.log((Ediro + Edifo)/(Edir + Edif))


@torch.jit.script
def Rrs_MODEL(Ediro,Edifo,lambda_,zenith,PAR,chla,NAP,CDOM,params: AbsorptionParams,z=torch.tensor(0)):
    """
    Remote Sensing Reflectance.
    Aas and Højerslev, 1999.
    perturbation_factors is a dictionary with all the parameters that are been optimices in this Model. Are multiplicative factors between 0 and 2, that modify the value
    of some of the constants of the bio quimical model. 
    """
    Rplus,Rminus,Explus,Exminus,x,y,Edir = three_layers_composition(Ediro,lambda_,zenith,PAR,chla,NAP,CDOM,params,z)
    Eu,Qrs = E_u(C_plus(x, Ediro, Edifo),y,Explus,Rplus,Edir),Q_rs(zenith,params)
    Rrs = Eu  / (Qrs*(Ediro + Edifo)   )
    return Rrs_plus( Rrs ,params)


class Forward_Model(nn.Module):
    """
    Bio-Optical model plus corrections, in order to have the Remote Sensing Reflectance, in terms of the inversion problem. 
    Forward_Model(x) returns a tensor, with each component being the Remote Sensing Reflectance for each given wavelength. 
    if the data has 5 rows, each with a different wavelength, RRS will return a vector with 5 components.  RRS has tree parameters, 
    self.chla is the chlorophil-a, self.NAP the Non Algal Particles, and self.CDOM the Colored Dissolved Organic Mather. 
    According to the invention problem, we want to estimate them by making these three parameters have two constraints,
    follow the equations of the bio-optical model, plus, making the RRS as close as possible to the value
    measured by the satellite.
    
    """
    def __init__(self,precision: torch.Tensor = torch.float32,num_days: int = 1,lat_size: int = 1, lon_size: int = 1 , \
                 learning_chla: bool = True,learning_perturbation_factors: bool = False):
        super().__init__()
        
        self.precision = precision
        self.learning_chla = learning_chla
        self.learning_perturbations = learning_perturbation_factors
        
        if learning_chla == True:
            #self.chparam = nn.Parameter(torch.rand((num_days,lat_size,lon_size,1,1,3), dtype=precision), requires_grad=True) #numbers between -1 and 1, prior mean zero and std ~ n1
            self.chparam = nn.Parameter(torch.ones((num_days,lat_size,lon_size,1,1,3), dtype=precision), requires_grad=True)
        if learning_perturbation_factors == True:
            self.perturbation_factors =  nn.Parameter(torch.ones(14, dtype=precision), requires_grad=True)
            self.perturbation_factors_names = [
                '$\\epsilon_{a,ph}$',
                '$\\epsilon_{slope,s,ph}$',
                '$\\epsilon_{intercept,s,ph}$',
                '$\\epsilon_{slope,b,ph}$',
                '$\\epsilon_{intercept,b,ph}$',
                '$\\epsilon_{d,cdom}$',
                '$\\epsilon_{s,cdom}$',
                '$\\epsilon_{q,a}$',
                '$\\epsilon_{q,b}$',
                '$\\epsilon_{theta,min}$',
                '$\\epsilon_{theta,o}$',
                '$\\epsilon_\\beta$',
                '$\\epsilon_\\sigma$',
                '$\\epsilon_{br,nap}$',
            ]
            
    def forward(self,x_data,constant:AbsorptionParams, chla: Optional[torch.Tensor] = None, axis : Optional[int]  = None ):
        """
        x_data: torch tensor, with shape (batch_size,lat_size,lon_size,1,lambdas_dim,5), such that x_data[...,i] is interpreted as a row vector.  
           where lambdas_dim is the number of wavelenghts.
           x_data[...,0] : E_s(lambda,0)
           x_data[...,1] : E_d(lambda,0)
           x_data[...,2] : lambda
           x_data[...,3] : zenith
           x_data[...,4] : PAR

        chla: torch tensor, with shape (...,1,3) or None. Default: None.
           if self.learning_chla is set to False, please provide this argument, with a tensor of shape (...,1,3), where 
           chla[...,0] : chlorophyll
           chla[...,1] : Non Algal Particles (Detritus)
           chla[...,2] : Colored Disolved Organic Matter
        
        constant: instance of class AbsorptionParams. This is a required parameter.
        """
        if self.learning_perturbations == True:
            constant.perturb_parameters(self.perturbation_factors)
        

        if self.learning_chla == True:
            Rrs = Rrs_MODEL(x_data[...,0],x_data[...,1],x_data[...,2],\
                            x_data[...,3],x_data[...,4],torch.exp(self.chparam[...,0]),torch.exp(self.chparam[...,1]),torch.exp(self.chparam[...,2]),constant)
            
            if axis is not None:
                Rrs = Rrs[:,axis]
            return Rrs.to(self.precision)
            
        else:
            if chla is None:
                print("self.learning_chla set to false, please provide values for chla, nap and cdom.")
            Rrs = Rrs_MODEL(x_data[...,0],x_data[...,1],x_data[...,2],\
                                x_data[...,3],x_data[...,4],torch.exp(chla[...,0]),torch.exp(chla[...,1]),torch.exp(chla[...,2]),constant)
            if axis is not None:
                Rrs = Rrs[:,axis]
            return Rrs.to(self.precision)
            
######Functions for error propagation########

@torch.jit.script
def error_propagation(df: torch.Tensor,sigma: torch.Tensor):
    error_ = df @  sigma @ torch.transpose(df,-2,-1)
    error = torch.diagonal(error_,dim1=-2,dim2=-1)
    return error

class evaluate_model_class():
    """
    class to evaluate functions needed to compute the uncerteinty. 
    """
    def __init__(self,model: Forward_Model, X: torch.Tensor, axis: Optional[int] = None,\
                 which_parameters: str = 'chla',chla : Optional[torch.Tensor] = None):
        self.axis = axis
        self.model = model
        self.X = X
        self.chla = chla
        self.which_parameters = which_parameters
        
        
    def model_der(self,parameters_eval: torch.Tensor, constant:AbsorptionParams ):
            
        if self.which_parameters == 'chla':
            learning_chla = self.model.learning_chla
            self.model.learning_chla = False
            rrs_ =  self.model(self.X,constant = constant, chla = parameters_eval,axis = self.axis)
            self.model.learning_chla = learning_chla
        elif self.which_parameters == 'perturbations':
            constant.perturb_parameters(parameters_eval)    
            rrs_ = self.model(self.X,parameters = self.chla,axis = self.axis,constant = constant)
        if self.axis is not None:
            return rrs_[...,axis]
        else:
            return rrs_
        
    def kd_der(self,parameters_eval: torch.Tensor, constant:AbsorptionParams):
        
        if self.which_parameters == 'chla': 
            kd_values = kd(9,self.X[...,0],self.X[...,1],self.X[...,2],\
                           self.X[...,3],self.X[...,4],torch.exp(parameters_eval[...,0]),torch.exp(parameters_eval[...,1]),torch.exp(parameters_eval[...,2]),constant)
        elif self.which_parameters == 'perturbations':
            constant.perturb_parameters(parameters_eval)
            kd_values = kd(9,self.X[...,0],self.X[...,1],self.X[...,2],self.X[...,3],self.X[...,4],\
                           torch.exp(self.chla[...,0]),torch.exp(self.chla[...,1]),torch.exp(self.chla[...,2]),constant)
        if self.axis is not None:
            kd_values = kd_values[...,axis]
        return kd_values

    def bbp_der(self,parameters_eval: torch.Tensor, constant:AbsorptionParams):
            
        if self.which_parameters == 'chla': 
            bbp_values = bbp(self.X[...,2],self.X[...,4],\
                             torch.exp(parameters_eval[...,0]),torch.exp(parameters_eval[...,1]),torch.exp(parameters_eval[...,2]),constant)
        elif self.which_parameters == 'perturbations':
            constant.perturb_parameters(parameters_eval)
            bbp_values = bbp(self.X[...,2],self.X[...,4],\
                             torch.exp(self.chla[...,0]),torch.exp(self.chla[...,1]),torch.exp(self.chla[...,2]),constant)
        if self.axis is not None:
            return bbp_values[...,axis]
        return bbp_values[...,[1,2,4]]
        

class RRS_loss(nn.Module):

    def __init__(self,x_a,s_a,s_e,precision = torch.float32,num_days:int =1,my_device:str = 'cpu'):
        super(RRS_loss, self).__init__()
        """
        Class to evaluate the loss function RRS_loss = -2log(p(z|x,y)), minus the log posterior distribution of the latent variable z=(chla,nap,cdom). 
        p(z|x,y) uses a Gaussian likelihood, and a Gaussian prior. 
        parameters:
          x_a: mean value of the prior values for chla, nap and cdom, with dimension (3).
          s_a: covariance matrix of the prior for chla, nap and cdom, dimension (3,3).
          s_e: covariance matrix of RRS. Dimension (5,5).
        """
        self.x_a = torch.stack([x_a for _ in range(num_days)]).to(my_device).to(precision)
        self.x_a
        self.s_a = s_a.to(my_device).to(precision)
        self.s_e = s_e.to(my_device).to(precision)
        self.s_e_inverse = torch.inverse(self.s_e)
        self.s_a_inverse = torch.inverse(self.s_a)
        self.precision = precision

    def forward(self,y,f_x,x):
        diff_y = (y - f_x)                  # (num_days,lat_size,lon_size,1,lambdas)
        diff_x = x - self.x_a    # (num_days,lat_size,lon_size,1,1,3)

        # Compute batch-wise quadratic forms

        term1 = torch.einsum('...bi,...ij,...bj->...b', diff_y, self.s_e_inverse, diff_y)  # (num_days,)
        term2 = torch.einsum('...bi,...ij,...bj->...b', diff_x, self.s_a_inverse, diff_x)  # (num_days,)
        
        total = term1 + term2
        loss = total.mean()

        return loss

class OBS_loss(nn.Module):

    def __init__(self,precision = torch.float32,my_device:str = 'cpu',normalization_values = torch.tensor([[1,1,1,1,1,1,1,1,1]])):
        super(OBS_loss, self).__init__()
        self.precision = precision
        self.normalization_values = torch.tensor(normalization_values).to(my_device).to(precision)


    def forward(self,Y_l,pred_l,nan_array):
        
        custom_array = ((Y_l-pred_l)/self.normalization_values)**2
        lens = torch.tensor([len(element[~element.isnan()]) for element in nan_array])

        means_output = torch.sum(custom_array,dim=1)/lens
        return means_output.mean().to(self.precision)


if __name__ == "__main__":

    pass
