import sys
folder = '2D_FNO'
sys.path.append(f"/home/{folder}") #formatted string literals


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
#get_ipython().run_line_magic('matplotlib', 'inline')
import yaml
import os
import math
import torch
from torch.utils.data import DataLoader

#from functorch import vmap, grad  (functorch replaced in new version)
#from models import FNN2d, FNN3d
from train_utils import Adam

import torch.nn.functional as F
import torch.nn as nn
#matplotlib.use('TKagg')
import numpy as np
import traceback
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from tqdm import tqdm
from train_utils.utils import  get_grid3d, load_checkpoint, load_config, update_config
from train_utils.losses import LpLoss
#from train_utils.datasets import DataLoader1D_coupled

try:
    import wandb
except ImportError:
    wandb = None



# configuration file
config_file = f'/home/{folder}/configs/config.yaml'
config = load_config(config_file)


ETA = torch.load("/home/2D10p64_suffled_eta_random6.pt",weights_only=False)
batch = config['data']['total_num']
ETA= torch.reshape(ETA,(batch,201,64,64))
#print(ETA.shape)

C = torch.load("/home/2D10p64_suffled_c_random6.pt",weights_only=False)
C= torch.reshape(C,(batch,201,64,64))
#print(C.shape)



C = C.float()
ETA = ETA.float()


data = torch.stack([C,ETA], dim=-1)
class DataLoader2D_coupled(object):
    def __init__(self, data, nx=64, nt=100, sub=1, sub_t=1, nfields=2):

        self.sub = sub
        self.sub_t = sub_t
        self.nfields = nfields
        s = nx
        # if nx is odd
        if (s % 2) == 1:
            s = s - 1
        self.S = s // sub
        self.T = nt // sub_t

        data = data[:, 0:self.T:sub_t, 0:s:sub, 0:s:sub]
        self.data = data


    def make_loader(self, n_sample, batch_size, start=0, train=True):
        a_data = self.data[start:start + n_sample, 0,:]
    
        u_data = self.data[start:start + n_sample].reshape(n_sample, self.T, self.S, self.S, self.nfields)
    
        a_data = a_data.reshape(n_sample, 1,self.S,self.S, self.nfields).repeat([1, self.T, 1, 1,1])
    
        print('u_data',u_data.shape) 
        print('a_data',a_data.shape) 

        dataset = torch.utils.data.TensorDataset(a_data, u_data)
        if train:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader
    

dataset = DataLoader2D_coupled(data, config['data']['nx'], config['data']['nt'], config['data']['sub'], config['data']['sub_t'])

#%%
train_loader = dataset.make_loader(config['data']['n_train'], config['train']['batchsize'], start=0, train=True)

#%%
test_loader = dataset.make_loader(config['data']['n_test'], config['test']['batchsize'], start=config['data']['n_train'],train=False )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from neuralop.models import FNO, FNO3d
from neuralop.utils import count_model_params
model = FNO(n_modes=(20, 20, 20),
             in_channels=2,
             out_channels=2,
             hidden_channels=4,
             n_layers=4,
             projection_channel_ratio=2)
model = model.to(device)

n_params = count_model_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()

from neuralop.training import AdamW
optimizer = AdamW(model.parameters(),
                                lr=8e-3,
                                weight_decay=1e-7)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)


def save_checkpoint(path, name, model, optimizer=None):
    ckpt_dir = 'checkpoints/%s/' % path
    os.makedirs(ckpt_dir)
    try:
        model_state_dict = model.module.state_dict()
    except AttributeError:
        model_state_dict = model.state_dict()

    if optimizer is not None:
        optim_dict = optimizer.state_dict()
    else:
        optim_dict = 0.0

    torch.save({
        'model': model_state_dict,
        'optim': optim_dict
    }, ckpt_dir + name)
    print('Checkpoint is saved at %s' % ckpt_dir + name)

import pickle as pkl
def train(model,
                             dataset,
                             train_loader,
                             optimizer, scheduler,
                             config,
                             M=1.0, L=1.0, padding=0,
                             rank=0, log=False,
                             project='PINO-2d-default',
                             group='default',
                             tags=['default'],
                             use_tqdm=True):

    if rank == 0 and wandb and log:
        run = wandb.init(project=project,
                         entity='shawngr2',
                         group=group,
                         config=config,
                         tags=tags, reinit=True,
                         settings=wandb.Settings(start_method="fork"))


    dataweight_CH = config['train']['Dc_loss']
    dataweight_AC = config['train']['Deta_loss']
    ckpt_freq = config['train']['ckpt_freq']
    model.train()
    myloss = LpLoss(size_average=True)
    S, T = dataset.S, dataset.T
    batch_size = config['train']['batchsize']
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    Tloss =[]
    Dloss_CH =[]
    Dloss_AC =[]
    for e in pbar:
        model.train()
        # Initialize variables to store different types of losses
        data_CH = 0.0
        data_AC = 0.0
        train_loss = 0.0

        for x, y in train_loader:

            x, y = x.to(rank), y.to(rank)
     
            x_in = F.pad(x, (0,0,0,0, 0, 0,0, padding), "constant", 0)

            x_in = x_in.permute(0, 4, 1, 2,3)

            out = model(x_in)
            out = out[..., :-padding,:,:]        
            out = out.permute(0, 2,3,4,1)

            dataloss_CH = myloss(out[...,0], y[...,0])
            dataloss_AC = myloss(out[...,1], y[...,1])


            total_loss = dataloss_CH * dataweight_CH +dataloss_AC* dataweight_AC

            optimizer.zero_grad()

            total_loss.backward()
            optimizer.step()
            data_CH += dataloss_CH.item()
            data_AC += dataloss_AC.item()
            train_loss += total_loss.item()

        scheduler.step()
        data_CH /= len(train_loader)
        data_AC /= len(train_loader)
        train_loss /= len(train_loader)

        Tloss.append(train_loss)
        Dloss_CH.append(data_CH)
        Dloss_AC.append(data_AC)
        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch {e}, train loss: {train_loss:.5f} '
                    f'data_CH error: {data_CH:.5f}; '
                    f'data_AC error: {data_AC:.5f}\n'
                )
            )
        if wandb and log:
            wandb.log(
                {
                    'Train Dloss_CH error': data_CH,
                    'Train Dloss_AC error': data_AC,
                    'Train loss': train_loss,
                }
            )

        if e % ckpt_freq == 0:
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)
    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model, optimizer)
    print('Done!')
    with open(f'/home/ggangmei/my-vast/NO_paper/FNO_paper/{folder}/Train_loss.pkl','wb') as f:
        pkl.dump(Tloss,f)
    with open(f'/home/ggangmei/my-vast/NO_paper/FNO_paper/{folder}/D_CH.pkl','wb') as f:
        pkl.dump(Dloss_CH,f)
    with open(f'/home/ggangmei/my-vast/NO_paper/FNO_paper/{folder}/D_AC.pkl','wb') as f:
        pkl.dump(Dloss_AC,f)

    return



import time
from train_utils.losses import LpLoss
log = False
st = time.time()
train(model,
    dataset,
    train_loader,
    optimizer, 
    scheduler,
    config,
    M=config['data']['M'],
    L=config['data']['L'],
    padding=17,
    rank=0,
    log=log,
    project=config['log']['project'],   
    group=config['log']['group'])
et = time.time()
elapsed_time = et - st

f = open(f'/home/{folder}/training_time.txt', 'w')
f.write(str(elapsed_time))
f.close()



def eval_burgers2D_vec_pad(model,
                       dataloader,
                       config,
                       M=1.0, L=1.0, padding=0,
                       use_tqdm=True):
    model.eval()
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    testc_err = []
    testeta_err = []
    testTotal_err = []

    
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            x_in = F.pad(x, (0,0,0,0, 0, 0,0, padding), "constant", 0)
            x_in = x_in.permute(0, 4, 1, 2,3)
            out = model(x_in)
            out = out[..., :-padding,:,:]       
            out = out.permute(0, 2,3,4,1)
   
            dataloss_CH = myloss(out[...,0], y[...,0])
            dataloss_AC = myloss(out[...,1], y[...,1])
            total_loss = dataloss_CH +dataloss_AC

            testc_err.append(dataloss_CH.item())
            testeta_err.append(dataloss_AC.item())
            testTotal_err.append(total_loss.item())

    with open(f'/home/ggangmei/my-vast/NO_paper/FNO_paper/{folder}/testc_err.pkl','wb') as f:
        pkl.dump(testc_err,f)
    with open(f'/home/ggangmei/my-vast/NO_paper/FNO_paper/{folder}/testeta_err.pkl','wb') as f:
        pkl.dump(testeta_err,f)

    meanc_err = np.mean(testc_err)
    stdc_err = np.std(testc_err, ddof=1) / np.sqrt(len(testc_err))
    meaneta_err = np.mean(testeta_err)
    stdeta_err = np.std(testeta_err, ddof=1) / np.sqrt(len(testeta_err))
    meanTotal_err = np.mean(testTotal_err)
    stdTotal_err = np.std(testTotal_err, ddof=1) / np.sqrt(len(testTotal_err))

    with open(f'/home/{folder}/meanTotal_err.pkl','wb') as f:
        pkl.dump(meanTotal_err,f)
    with open(f'/home/{folder}/stdTotal_err.pkl','wb') as f:
        pkl.dump(stdTotal_err,f)
    with open(f'/home/{folder}/meanc_err.pkl','wb') as f:
        pkl.dump(meanc_err,f)
    with open(f'/home/{folder}/stdc_err.pkl','wb') as f:
        pkl.dump(stdc_err,f)
    with open(f'/home/{folder}/meaneta_err.pkl','wb') as f:
        pkl.dump(meaneta_err,f)
    with open(f'/home/{folder}/stdeta_err.pkl','wb') as f:
        pkl.dump(stdeta_err,f)




eval_burgers2D_vec_pad(model,
                       test_loader,
                       config,
                       M=config['data']['M'],
                       L=config['data']['L'],
                       padding=17,
                       use_tqdm=True)



#%%# Generate Test Predictions  with Padding
padding = 17
batch_size = config['test']['batchsize']
Nx = config['data']['nx']
Nt = config['data']['nt'] 
Ntest = config['data']['n_test']
Ntrain = config['data']['n_train']
# Ntest = Ntrain
in_dim = 2
out_dim = 2

model.eval()
# model.to('cpu')
test_x = np.zeros((Ntest,Nt,Nx, Nx, in_dim))
#preds_y = np.zeros((0,128,101,2))
preds_y = np.zeros((Ntest,Nt,Nx, Nx, out_dim))
test_y0 = np.zeros((Ntest,Nx, Nx, out_dim))
test_y = np.zeros((Ntest,Nt,Nx, Nx, out_dim))

with torch.no_grad():
    for i, data in enumerate(test_loader):
#     for i, data in enumerate(train_loader):
        data_x, data_y = data
        data_x, data_y = data_x.to(device), data_y.to(device)
        #print(data_y[:,:,0,1])
        data_x_pad = F.pad(data_x, (0,0,0,0, 0, 0,0, padding), "constant", 0)
        #print('data_x_pred',data_x_pad.shape)
        data_x_pad=data_x_pad.permute(0, 4, 1, 2,3)
        pred_y_pad = model(data_x_pad)  
        pred_y_pad = pred_y_pad[..., :-padding,:,:]    
        pred_y = pred_y_pad.permute(0, 2,3,4,1)
  
        test_x[i] = data_x.cpu().numpy()
        test_y[i] = data_y.cpu().numpy()
        test_y0[i] = data_x[:, 0, :, :,-out_dim:].cpu().numpy() # same way as in training code
        preds_y[i] = pred_y.cpu().numpy()





#%% save and load data
def save_data(data_path, test_x, test_y, preds_y):
    data_dir, data_filename = os.path.split(data_path)
    os.makedirs(data_dir, exist_ok=True)
    np.savez(data_path, test_x=test_x, test_y=test_y, preds_y=preds_y)

def load_data(data_path):
    data = np.load(data_path)
    test_x = data['test_x']
    test_y = data['test_y']
    preds_y = data['preds_y']
    return test_x, test_y, preds_y

data_dir = f'home/{folder}/Data'
data_filename = 'data1.npz'
data_path = os.path.join(data_dir, data_filename)
save_data(data_path, test_x, test_y, preds_y)



#%%
data_dir = f'home/{folder}'
data_filename = 'model.pth'
data_path = os.path.join(data_dir, data_filename)
torch.save(model.state_dict(),data_path)



