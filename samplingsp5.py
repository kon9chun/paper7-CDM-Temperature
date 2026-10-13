import numpy as np
# embdata=np.load('embdata.npy')
ydata_norm=np.load('ydata_norm.npy')
sampled_point=np.load('sp.npy')
sp5index=np.array([1,3,5,7,9])-1
sp5=sampled_point[:,sp5index]

def normalize(x,x_max,x_min):
    # if x_max==x_min:
    #     return np.zeros_like(x)
    return (x-x_min)/(x_max-x_min)*2-1

def denormalize(x,x_max,x_min):
    drange=x_max-x_min
    return (x+1)/2*drange+x_min

spmax=sp5.max(axis=1)
spmin=sp5.min(axis=1)

sp_n=normalize(sp5,spmax[:,np.newaxis],spmin[:,np.newaxis])
spmax_max=spmax.max()
spmax_min=spmax.min()
spmin_max=spmin.max()
spmin_min=spmin.min()

spmax_n=normalize(spmax,spmax_max,spmax_min)
spmin_n=normalize(spmin,spmin_max,spmin_min)

embdata=np.concatenate((sp_n,spmax_n[:,np.newaxis],spmin_n[:,np.newaxis]),axis=1)

array=np.arange(0,ydata_norm.shape[0])
index=np.random.choice(array,10000,replace=False)

v_ydata=ydata_norm[index]
v_emb=embdata[index]
t_ydata=np.delete(ydata_norm,index,0)
t_emb=np.delete(embdata,index,0)

from torch.utils.data import Dataset,DataLoader
class MyDataset(Dataset):
    def __init__(self,embdata,ydata):
        self.embdata=embdata
        self.ydata=ydata
    def __len__(self):
        return len(self.embdata)
    def __getitem__(self,idx):
        return self.embdata[idx],self.ydata[idx]

train_dataset=MyDataset(t_emb,t_ydata)
val_dataset=MyDataset(v_emb,v_ydata)
train_loader=DataLoader(train_dataset,batch_size=32)
val_loader=DataLoader(val_dataset,batch_size=50)

import torch
import torch.nn as nn
from tqdm import tqdm
from diffusion_model import DiffusionModel
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device="cuda:1"

diffusion=DiffusionModel(
    image_size=(40,240),
    channels=1,
    cond_dim=7,
    timesteps=1000,
    device=device
)

diffusion.load('final_model_sp5.pt')

diffusion.model.eval()
y_true=np.empty((0,40,240))
y_pred=np.empty((0,40,240))
x_in=np.empty((0,7))
for veb,vyb in val_loader:
    x_in=np.concatenate((x_in,veb.numpy()),axis=0)
    y_true=np.concatenate((y_true,vyb.numpy()),axis=0)
    veb=veb.to(device)
    # vyb=vyb.unsqueeze(1).to(device)
    sample=diffusion.generate(veb, num_samples=vyb.size(0))
    y_pred=np.concatenate((y_pred,sample[:,0,:,:]),axis=0)

spmax_vector=denormalize(x_in[:,5],spmax_max,spmax_min).reshape(-1,1,1)
spmin_vector=denormalize(x_in[:,6],spmin_max,spmin_min).reshape(-1,1,1)

true=denormalize(y_true,spmax_vector,spmin_vector)
pred=denormalize(y_pred,spmax_vector,spmin_vector)

def calculate_rmse(matrix1, matrix2):
    error = matrix1 - matrix2    
    # 计算均方误差 (MSE)
    mse = np.mean(error ** 2)    
    # 计算均方根误差 (RMSE)
    rmse = np.sqrt(mse)    
    return rmse
sp5_rmse=[]
for i in range(true.shape[0]):
    rmse=calculate_rmse(true[i],pred[i])
    sp5_rmse.append(rmse)

np.save('sp5_pred.npy',pred)
np.save('sp5_rmse.npy', np.array(sp5_rmse))
print(np.array(sp5_rmse).shape)