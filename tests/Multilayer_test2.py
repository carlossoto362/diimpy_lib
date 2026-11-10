import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt

import diimpy.Forward_module_v2 as fm
import diimpy.Forward_Multilayer as mult
from diimpy.read_data_module import read_constants
from diimpy.settings import DEFAULT_SETTINGS_DIR
data_path = DEFAULT_SETTINGS_DIR

batch_size = 1
lat_size = 1
lon_size = 1
lambdas_dim = 5

params = read_constants(lambdas_path= DEFAULT_SETTINGS_DIR / 'lambdas',lambdas_column_name='lambdas',cte_path = DEFAULT_SETTINGS_DIR / 'cte.csv',tensor = True,my_device = 'cpu',dict=False,precision=torch.float32)
perturbations = torch.load(DEFAULT_SETTINGS_DIR / 'perturbations_diim.pt')
params.perturb_parameters(perturbations)


N=500
z = np.linspace(0.001,1,N+1)
dz = z[1:] - z[:-1]#depth of each layer
z = z[:-1]#depth of the top part of each layer

chlorophyll_density = (scipy.stats.beta.pdf(z, 2, 5))*0.5
nap_density = (scipy.stats.beta.pdf(z, 3, 4)) * 6
cdom_density = (scipy.stats.beta.pdf(z, 2, 5))*2
PAR = (scipy.stats.beta.pdf(z, 1, 15))*10
z= z*500

plt.plot(chlorophyll_density,z,label='generated chl')
plt.plot(nap_density,z,label='generated nap')
plt.plot(cdom_density,z,label='generated cdom')
plt.ylim(500,-10)
plt.xlim(0,15)
plt.legend()
plt.xlabel("density [$\\mathrm{mg} \\quad \\mathrm{m}^{-3}$]")
plt.ylabel("depth $[\\mathrm{m}]$")
plt.show()

plt.plot(PAR,z,'--',label='generated PAR')
plt.ylim(500,-10)
plt.xlim(0,200)
plt.legend()
plt.xlabel("PAR [$\\mathrm{W} \\quad \\mathrm{m}^{-2}$]")
plt.ylabel("depth $[\\mathrm{m}]$")
plt.show()


chlorophyll_density = chlorophyll_density.reshape((1,1,1,N,1))
nap_density = nap_density.reshape((1,1,1,N,1))
cdom_density = cdom_density.reshape((1,1,1,N,1))
PAR = PAR.reshape((1,1,1,N,1))
lambda_ = torch.tensor([412.5,442.5,490,510,555]).reshape((1,1,1,1,5))

Ediro = torch.ones((1,1,1,1,5))*5
Edifo = torch.ones((1,1,1,1,5))*2
zenith = torch.tensor(51)

z = z.reshape((1,1,1,N,1))
#z[...,-1,0] = 1000
dz = dz.reshape((1,1,1,N,1))
#dz[...,-1,0] = 500

a,b,bb = fm.a_b_bb(lambda_,PAR,chlorophyll_density,nap_density,cdom_density,params)
Bs,Bu,Bd,Cs,Cu,cd,Fd = fm.second_layer(a,b,bb,zenith,params)
Rplus,Rminus,Explus,Exminus,xplus,xminus,yplus,yminus,Edir,cplus,cminus = mult.multilayer_parameters(Ediro,Edifo,z,dz,Bs,Bu,Bd,Cs,Cu,cd,Fd)
Edif = mult.Edif_multilayer(Rminus,Explus,xplus,Edir,cplus,cminus,Edifo)
Eu = mult.Eu_multilayer(Rplus,Explus,yplus,Edir,cplus,cminus,Ediro,Exminus[...,0,:],yminus[...,0,:])
print(torch.exp(- (fm.D_(Bs,Bu,Cs,Cu) - Cs) * dz))
print((fm.D_(Bs,Bu,Cs,Cu) - Cs))
#print(cminus)
Edir_ = torch.zeros((Edif.shape))
Edir_[...,1:,:] = Edir
Edir_[...,0,:] = Ediro
Edir = Edir_


plt.plot(a[0,0,0,:,0],z[0,0,0,:,0],label='a')
plt.plot(b[0,0,0,:,0],z[0,0,0,:,0],label='b')
plt.plot(bb[0,0,0,:,0],z[0,0,0,:,0],label='bb')
plt.ylim(500,-10)
plt.xlim(0,15)
plt.legend()
plt.xlabel("density [$\\mathrm{mg} \\quad \\mathrm{m}^{-3}$]")
plt.ylabel("depth $[\\mathrm{m}]$")
plt.show()


plt.plot(Eu[0,0,0,:-1,2],z[0,0,0,:,0],label='Eu output')
plt.plot(Edif[0,0,0,:-1,2],z[0,0,0,:,0],label='Edif output')
plt.plot(Edir[0,0,0,:-1,2],z[0,0,0,:,0],label='Edir output')


plt.ylim(1000,-10)
plt.xlim(-1,15)
plt.legend()
plt.xlabel("Irradiance [$\\mathrm{W} \\quad \\mathrm{m}^{-2}$]")
plt.ylabel("depth $[\\mathrm{m}]$")
plt.show()
