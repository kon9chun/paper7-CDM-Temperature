import numpy as np
embdata=np.load('embdata.npy')
ydata_norm=np.load('ydata_norm.npy')

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

diffusion=DiffusionModel(
    image_size=(40,240),
    channels=1,
    cond_dim=11,
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

diffusion.save('final_model.pt')