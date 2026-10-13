import numpy as np
import torch
torch.manual_seed(42)
# embdata=np.load('embdata.npy')
# ydata_norm=np.load('ydata_norm.npy')
np.random.seed(42)
ydata=np.load('ydata.npy')
sampled_point=np.load('sp.npy')
sp5index=np.array([1,3,5,7,9])-1
sp3index=np.array([1,5,9])-1
sp5=sampled_point[:,sp5index]
sp3=sampled_point[:,sp3index]

def normalize(x,x_max,x_min):
    # if x_max==x_min:
    #     return np.zeros_like(x)
    return (x-x_min)/(x_max-x_min)*2-1

def denormalize(x,x_max,x_min):
    drange=x_max-x_min
    return (x+1)/2*drange+x_min

sp5max=np.max(sp5,axis=1)
sp5min=np.min(sp5,axis=1)

sp5max_max=sp5max.max()
sp5max_min=sp5max.min()
sp5min_max=sp5min.max()
sp5min_min=sp5min.min()

sp5max_n=normalize(sp5max,sp5max_max,sp5max_min)
sp5min_n=normalize(sp5min,sp5min_max,sp5min_min)

sp5n=normalize(sp5,sp5max.reshape(-1,1),sp5min.reshape(-1,1))

embdata=np.concatenate((sp5n,sp5max_n.reshape(-1,1),sp5min_n.reshape(-1,1)),axis=1)
ydata_norm=normalize(ydata,sp5max.reshape(-1,1,1),sp5min.reshape(-1,1,1))

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


import torch.nn as nn
from tqdm import tqdm
from diffusion_model import DiffusionModel
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

diffusion=DiffusionModel(
    image_size=(40,240),
    channels=1,
    cond_dim=7,
    timesteps=1000,
    device=device
)

epochs=250
optimizer=torch.optim.AdamW(diffusion.model.parameters(),lr=1e-5,weight_decay=1e-5)
scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-7)

for epoch in range(epochs):
    diffusion.model.train()
    epoch_loss=0
    progress_bar=tqdm(train_loader,desc=f"Epoch {epoch+1}/{epochs}")
    for eb,yb in progress_bar:
        eb=eb.to(device)
        yb=yb.unsqueeze(1).to(device)
        loss=diffusion.train_step(yb,eb,optimizer)
        epoch_loss+=loss
        progress_bar.set_postfix({'loss':loss})
    avg_loss=epoch_loss/len(train_loader)
    scheduler.step()

diffusion.save('final_model_sp5_0319.pt')