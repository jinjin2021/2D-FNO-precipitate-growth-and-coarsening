import sys
<<<<<<< HEAD

folder = 'LocalNO/layer/4l16w26m_LocalNO'
sys.path.append(f"/home/ggangmei/my-vast/{folder}") #formatted string literals


#from IPython.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))
=======
folder = '2D_FNO'
sys.path.append(f"/home/{folder}") #formatted string literals


from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
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
<<<<<<< HEAD
config_file = f'/home/ggangmei/my-vast/{folder}/configs/config.yaml'
config = load_config(config_file)


ETA = torch.load("/home/ggangmei/my-vast/FNO_EAAI/400samples_EAAI_2D10p64_suffled_eta.pt",weights_only=False)
batch = config['data']['total_num']
ETA= torch.reshape(ETA,(batch,100,64,64))
#print(ETA.shape)

C = torch.load("/home/ggangmei/my-vast/FNO_EAAI/400samples_EAAI_2D10p64_suffled_c.pt",weights_only=False)
C= torch.reshape(C,(batch,100,64,64))
=======
config_file = f'/home/{folder}/configs/config.yaml'
config = load_config(config_file)


ETA = torch.load("/home/2D10p64_suffled_eta_random6.pt",weights_only=False)
batch = config['data']['total_num']
ETA= torch.reshape(ETA,(batch,201,64,64))
#print(ETA.shape)

C = torch.load("/home/2D10p64_suffled_c_random6.pt",weights_only=False)
C= torch.reshape(C,(batch,201,64,64))
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
#print(C.shape)



C = C.float()
ETA = ETA.float()


data = torch.stack([C,ETA], dim=-1)
class DataLoader2D_coupled(object):
    def __init__(self, data, nx=64, nt=100, sub=1, sub_t=1, nfields=2):
<<<<<<< HEAD
#         dataloader = MatReader(datapath)
=======

>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
        self.sub = sub
        self.sub_t = sub_t
        self.nfields = nfields
        s = nx
        # if nx is odd
        if (s % 2) == 1:
            s = s - 1
        self.S = s // sub
        self.T = nt // sub_t
<<<<<<< HEAD
        #self.T += 1
        data = data[:, 0:self.T:sub_t, 0:s:sub, 0:s:sub]
        self.data = data#.permute(0, 2, 1, 3)#Rearranges the dimensions of the sliced data tensor
        #The permutation changes the order of dimensions from (batch(0), time(1), spatial1(2), field(3)) to (batch, spatial1, spatial2, time, field).


    def make_loader(self, n_sample, batch_size, start=0, train=True):
        a_data = self.data[start:start + n_sample, 0,:]#.reshape(n_sample, self.S, self.nfields)#slicing starts from the left. If it does not mentioned the last element, it will take the origianal value. In this case a_data is the intial value of both the fields
        print('a_data', a_data.shape)
        #torch.Size([196, 128, 128, 101, 2]); data = a[:, :, :,0]= torch.Size([196, 128, 128, 2])
        u_data = self.data[start:start + n_sample].reshape(n_sample, self.T, self.S, self.S, self.nfields)
    
        a_data = a_data.reshape(n_sample, 1,self.S,self.S, self.nfields).repeat([1, self.T, 1, 1,1])#This line reshapes a_data, likely another dataset, into a 5-dimensional tensor similar to u_data.
=======

        data = data[:, 0:self.T:sub_t, 0:s:sub, 0:s:sub]
        self.data = data


    def make_loader(self, n_sample, batch_size, start=0, train=True):
        a_data = self.data[start:start + n_sample, 0,:]
    
        u_data = self.data[start:start + n_sample].reshape(n_sample, self.T, self.S, self.S, self.nfields)
    
        a_data = a_data.reshape(n_sample, 1,self.S,self.S, self.nfields).repeat([1, self.T, 1, 1,1])
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
    
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
<<<<<<< HEAD
from neuralop.models import LocalNO 
from neuralop.utils import count_model_params
model = LocalNO(n_modes=(26, 26, 26),
             in_channels=2,
             out_channels=2,
             hidden_channels=16,
             default_in_shape=(100,64,64),
             n_layers=4,
             disco_layers=False,
=======
from neuralop.models import FNO, FNO3d
from neuralop.utils import count_model_params
model = FNO(n_modes=(20, 20, 20),
             in_channels=2,
             out_channels=2,
             hidden_channels=4,
             n_layers=4,
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
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
<<<<<<< HEAD
    #ckpt_dir = 'checkpoints/%s/' % path
    ckpt_dir = '/scratch/ggangmei/LocalNO/checkpoints/%s/' % path
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
=======
    ckpt_dir = 'checkpoints/%s/' % path
    os.makedirs(ckpt_dir)
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
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
<<<<<<< HEAD
    # fc_weight = config['train']['fc_loss']
    # feta_weight = config['train']['feta_loss']
    ckpt_freq = config['train']['ckpt_freq']
    #nfields = config['model']['out_dim']
#     nfields = model.out_dim
=======
    ckpt_freq = config['train']['ckpt_freq']
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
    model.train()
    myloss = LpLoss(size_average=True)
    S, T = dataset.S, dataset.T
    batch_size = config['train']['batchsize']
    pbar = range(config['train']['epochs'])
    if use_tqdm:
        pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)

    Tloss =[]
<<<<<<< HEAD
    # PDE_ACloss =[]
    # PDE_CHloss =[]
=======
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
    Dloss_CH =[]
    Dloss_AC =[]
    for e in pbar:
        model.train()
        # Initialize variables to store different types of losses
<<<<<<< HEAD
        # train_pinoAC = 0.0
        # train_pinoCH = 0.0
        data_CH = 0.0
        data_AC = 0.0
        train_loss = 0.0
        dataloss_Ca =0.0
        dataloss_Cb = 0.0

        cases_ca = torch.zeros((0,1)).to(rank)
        cases_cb = torch.zeros((0,1)).to(rank)
        cases_c = torch.zeros((0,1)).to(rank)
        boxplot_C = torch.zeros((0,2)).to(rank)

        cases_ETAa = torch.zeros((0,1)).to(rank)
        cases_ETAb = torch.zeros((0,1)).to(rank)
        cases_ETA = torch.zeros((0,1)).to(rank)
        boxplot_ETA = torch.zeros((0,2)).to(rank)

        cases_Ta = torch.zeros((0,1)).to(rank)
        cases_Tb = torch.zeros((0,1)).to(rank)
        cases_T = torch.zeros((0,1)).to(rank)
        boxplot_T = torch.zeros((0,2)).to(rank)
        for x, y in train_loader:
            #print('rank', rank)
            #x, y = x.to('cuda:0'), y.to('cuda:1')
            x, y = x.to(rank), y.to(rank)
            # print('x', x.shape) #torch.Size([1, 11, 64, 64, 5])
            # print('y', y.shape) #torch.Size([1, 11, 64, 64, 2])
            x_in = F.pad(x, (0,0,0,0, 0, 0,0, padding), "constant", 0)
            #print('x_in',x_in.shape) #torch.Size([2, 100, 64, 64, 5])
            default_in_shape=x_in.shape
            x_in = x_in.permute(0, 4, 1, 2,3)
            #print('x_in',x_in.shape) #torch.Size([2, 5, 100, 64, 64])
            out = model(x_in)#.reshape(batch_size,T + padding,S,S, nfields)
            #print('out',out.shape)    #torch.Size([2, 2, 100, 64, 64])   
            out = out[..., :-padding,:,:]        
            out = out.permute(0, 2,3,4,1)
            #print('out',out.shape) #torch.Size([2, 100, 64, 64, 2])
            for t in range(0,100,1):
                dataloss_Ca = myloss(out[0,t,:,:,0], y[0,t,:,:,0])
                #print('dataloss_Ca', dataloss_Ca.shape)
                dataloss_Cb = myloss(out[1,t,:,:,0], y[1,t,:,:,0])
                
                cases_ca=torch.vstack([cases_ca,dataloss_Ca])
    
                cases_cb=torch.vstack([cases_cb,dataloss_Cb])

                dataloss_ETAa = myloss(out[0,t,:,:,1], y[0,t,:,:,1])
                #print('dataloss_Ca', dataloss_Ca.shape)
                dataloss_ETAb = myloss(out[1,t,:,:,1], y[1,t,:,:,1])

                cases_ETAa=torch.vstack([cases_ETAa,dataloss_ETAa])
                cases_ETAb=torch.vstack([cases_ETAb,dataloss_ETAb])

                Tloss_a= dataloss_Ca + dataloss_ETAa
                Tloss_b= dataloss_Cb + dataloss_ETAb

                
                cases_Ta=torch.vstack([cases_Ta,Tloss_a])
                cases_Tb=torch.vstack([cases_Tb,Tloss_b])

            cases_c=torch.vstack([cases_ca,cases_cb])
            #print('cases_c', cases_c.shape)
            cases_ETA=torch.vstack([cases_ETAa,cases_ETAb])
            cases_T=torch.vstack([cases_Ta,cases_Tb])
            dataloss_CH = myloss(out[...,0], y[...,0])
            dataloss_AC = myloss(out[...,1], y[...,1])

            #loss_fc, loss_feta = PINO_loss_burgers2D_vec(out, M=M, L=L)
            # total_loss = loss_fc * fc_weight + loss_feta * feta_weight \
            #     + dataloss_CH * dataweight_CH +dataloss_AC* dataweight_AC
=======
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


>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
            total_loss = dataloss_CH * dataweight_CH +dataloss_AC* dataweight_AC

            optimizer.zero_grad()

            total_loss.backward()
            optimizer.step()
            data_CH += dataloss_CH.item()
            data_AC += dataloss_AC.item()
<<<<<<< HEAD
            # train_pinoCH += loss_fc.item()
            # train_pinoAC += loss_feta.item()
            train_loss += total_loss.item()
#Each of these variables holds the sum of the corresponding loss over all batches.
        N = config['data']['n_train']
        boxplot_C = torch.reshape(cases_c,(N,100))
        #print('boxplot_C', boxplot_C.shape)
        boxplot_ETA = torch.reshape(cases_ETA,(N,100))
        #print('boxplot_ETA', boxplot_ETA.shape)
        boxplot_T = torch.reshape(cases_T,(N,100))
        #print('boxplot_T', boxplot_T.shape)
=======
            train_loss += total_loss.item()
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503

        scheduler.step()
        data_CH /= len(train_loader)
        data_AC /= len(train_loader)
<<<<<<< HEAD
        # train_pinoCH /= len(train_loader)
        # train_pinoAC /= len(train_loader)
        train_loss /= len(train_loader)

        Tloss.append(train_loss)
        # PDE_CHloss.append(train_pinoCH)
        # PDE_ACloss.append(train_pinoAC)
=======
        train_loss /= len(train_loader)

        Tloss.append(train_loss)
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
        Dloss_CH.append(data_CH)
        Dloss_AC.append(data_AC)
        if use_tqdm:
            pbar.set_description(
                (
                    f'Epoch {e}, train loss: {train_loss:.5f} '
<<<<<<< HEAD
                    # f'train fc error: {train_pinoCH:.5f}; '
                    # f'train feta error: {train_pinoAC:.5f}; '
=======
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
                    f'data_CH error: {data_CH:.5f}; '
                    f'data_AC error: {data_AC:.5f}\n'
                )
            )
        if wandb and log:
            wandb.log(
                {
<<<<<<< HEAD
                    # 'Train fc error': train_pinoCH,
                    # 'Train feta error': train_pinoAC,
=======
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
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
<<<<<<< HEAD
    with open(f'/home/ggangmei/my-vast/{folder}/Train_loss.pkl','wb') as f:
        pkl.dump(Tloss,f)
    # with open(f'/home/ggangmei/my-vast/NO_paper/{folder}/PDE_CHloss.pkl','wb') as f:
    #     pkl.dump(PDE_CHloss,f)
    # with open(f'/home/ggangmei/my-vast/NO_paper/{folder}/PDE_ACloss.pkl','wb') as f:
    #     pkl.dump(PDE_ACloss,f)
    with open(f'/home/ggangmei/my-vast/{folder}/D_CH.pkl','wb') as f:
        pkl.dump(Dloss_CH,f)
    with open(f'/home/ggangmei/my-vast/{folder}/D_AC.pkl','wb') as f:
        pkl.dump(Dloss_AC,f)
    with open(f'/home/ggangmei/my-vast/{folder}/boxplot_C.pkl','wb') as f:
        pkl.dump(boxplot_C,f)
    with open(f'/home/ggangmei/my-vast/{folder}/boxplot_ETA.pkl','wb') as f:
        pkl.dump(boxplot_ETA,f)
    with open(f'/home/ggangmei/my-vast/{folder}/boxplot_T.pkl','wb') as f:
        pkl.dump(boxplot_T,f)
=======
    with open(f'/home/ggangmei/my-vast/NO_paper/FNO_paper/{folder}/Train_loss.pkl','wb') as f:
        pkl.dump(Tloss,f)
    with open(f'/home/ggangmei/my-vast/NO_paper/FNO_paper/{folder}/D_CH.pkl','wb') as f:
        pkl.dump(Dloss_CH,f)
    with open(f'/home/ggangmei/my-vast/NO_paper/FNO_paper/{folder}/D_AC.pkl','wb') as f:
        pkl.dump(Dloss_AC,f)
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503

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

<<<<<<< HEAD
f = open(f'/home/ggangmei/my-vast/{folder}/training_time.txt', 'w')
=======
f = open(f'/home/{folder}/training_time.txt', 'w')
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
f.write(str(elapsed_time))
f.close()



def eval_burgers2D_vec_pad(model,
                       dataloader,
                       config,
                       M=1.0, L=1.0, padding=0,
                       use_tqdm=True):
    model.eval()
<<<<<<< HEAD
    #nfields = config['model']['out_dim']
#     nfields = model.nfields
=======
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
    myloss = LpLoss(size_average=True)
    if use_tqdm:
        pbar = tqdm(dataloader, dynamic_ncols=True, smoothing=0.05)
    else:
        pbar = dataloader

    testc_err = []
    testeta_err = []
    testTotal_err = []
<<<<<<< HEAD
    dataloss_Ca =0.0
    dataloss_Cb = 0.0

    cases_ca = torch.zeros((0,1)).to(device)
    boxplot_C = torch.zeros((0,2)).to(device)

    cases_ETAa = torch.zeros((0,1)).to(device)
    boxplot_ETA = torch.zeros((0,2)).to(device)

    cases_Ta = torch.zeros((0,1)).to(device)
    boxplot_T = torch.zeros((0,2)).to(device)
=======

>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
    
    with torch.no_grad():
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
<<<<<<< HEAD
            # print('x', x.shape) #torch.Size([1, 11, 64, 64, 5])
            # print('y', y.shape) #torch.Size([1, 11, 64, 64, 2])
            x_in = F.pad(x, (0,0,0,0, 0, 0,0, padding), "constant", 0)
            #print('x_in',x_in.shape) #torch.Size([2, 100, 64, 64, 5])
            x_in = x_in.permute(0, 4, 1, 2,3)
            #print('x_in',x_in.shape) #torch.Size([2, 5, 100, 64, 64])
            out = model(x_in)#.reshape(batch_size,T + padding,S,S, nfields)
            out = out[..., :-padding,:,:]
            #print('out',out.shape)    #torch.Size([2, 2, 100, 64, 64])           
            out = out.permute(0, 2,3,4,1)
            #print('out',out.shape)
            #out = out[..., :-padding, :,:,:]#torch.Size([1, 11, 64, 64, 2])
            #print('out_padding',out.shape)
            for t in range(0,100,1):
                dataloss_Ca = myloss(out[0,t,:,:,0], y[0,t,:,:,0])
                #print('dataloss_Ca', dataloss_Ca.shape)
                
                cases_ca=torch.vstack([cases_ca,dataloss_Ca])
    

                dataloss_ETAa = myloss(out[0,t,:,:,1], y[0,t,:,:,1])
                #print('dataloss_Ca', dataloss_Ca.shape)

                Tloss_a= dataloss_Ca + dataloss_ETAa

                cases_ETAa=torch.vstack([cases_ETAa,dataloss_ETAa])
                cases_Ta=torch.vstack([cases_Ta,Tloss_a])

=======
            x_in = F.pad(x, (0,0,0,0, 0, 0,0, padding), "constant", 0)
            x_in = x_in.permute(0, 4, 1, 2,3)
            out = model(x_in)
            out = out[..., :-padding,:,:]       
            out = out.permute(0, 2,3,4,1)
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
   
            dataloss_CH = myloss(out[...,0], y[...,0])
            dataloss_AC = myloss(out[...,1], y[...,1])
            total_loss = dataloss_CH +dataloss_AC

<<<<<<< HEAD
            #loss_fc, loss_feta = PINO_loss_burgers2D_vec(out, M=M, L=L)

            testc_err.append(dataloss_CH.item())
            testeta_err.append(dataloss_AC.item())
            testTotal_err.append(total_loss.item())
            # fc_err.append(loss_fc.item())
            # feta_err.append(loss_feta.item())
        #print('cases_ca',cases_ca.shape)
        N=config['data']['n_test']
        boxplot_Ctest = torch.reshape(cases_ca,(N,100))
        #print('boxplot_C', boxplot_C.shape)
        boxplot_ETAtest = torch.reshape(cases_ETAa,(N,100))
        #print('boxplot_ETA', boxplot_ETA.shape)
        boxplot_Ttest = torch.reshape(cases_Ta,(N,100))
        #print('boxplot_T', boxplot_T.shape)
    #with open(f'/home/ggangmei/my-vast/FNO_EAAI/{folder}/testc_err.pkl','wb') as f:
     #   pkl.dump(testc_err,f)
   # with open(f'/home/ggangmei/my-vast/FNO_EAAI/{folder}/testeta_err.pkl','wb') as f:
    #    pkl.dump(testeta_err,f)
    #with open(f'/home/ggangmei/my-vast/FNO_EAAI/{folder}/testTotal_err.pkl','wb') as f:
     #    pkl.dump(testTotal_err,f)
    # with open(f'/home/ggangmei/my-vast/NO_paper/{folder}/fc_err.pkl','wb') as f:
    #     pkl.dump(fc_err,f)
    # with open(f'/home/ggangmei/my-vast/NO_paper/{folder}/feta_err.pkl','wb') as f:
    #     pkl.dump(feta_err,f)
    # mean_fc_err = np.mean(fc_err)
    # std_fc_err = np.std(fc_err, ddof=1) / np.sqrt(len(fc_err))
    
    # mean_feta_err = np.mean(feta_err)
    # std_feta_err = np.std(feta_err, ddof=1) / np.sqrt(len(feta_err))
=======
            testc_err.append(dataloss_CH.item())
            testeta_err.append(dataloss_AC.item())
            testTotal_err.append(total_loss.item())

    with open(f'/home/ggangmei/my-vast/NO_paper/FNO_paper/{folder}/testc_err.pkl','wb') as f:
        pkl.dump(testc_err,f)
    with open(f'/home/ggangmei/my-vast/NO_paper/FNO_paper/{folder}/testeta_err.pkl','wb') as f:
        pkl.dump(testeta_err,f)
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503

    meanc_err = np.mean(testc_err)
    stdc_err = np.std(testc_err, ddof=1) / np.sqrt(len(testc_err))
    meaneta_err = np.mean(testeta_err)
    stdeta_err = np.std(testeta_err, ddof=1) / np.sqrt(len(testeta_err))
    meanTotal_err = np.mean(testTotal_err)
    stdTotal_err = np.std(testTotal_err, ddof=1) / np.sqrt(len(testTotal_err))

<<<<<<< HEAD


   # print(f'==Averaged relative L2 error mean: {mean_err}, std error: {std_err}==\n'
    #      f'==Averaged equation error mean: {mean_fc_err}, std error: {std_fc_err}==\n'
     #     f'==Averaged equation error mean: {mean_feta_err}, std error: {std_feta_err}==\n')
    # with open(f'/home/ggangmei/my-vast/NO_paper/{folder}/mean_fc_err.pkl','wb') as f:
    #     pkl.dump(mean_fc_err,f)
    # with open(f'/home/ggangmei/my-vast/NO_paper/{folder}/std_fc_err.pkl','wb') as f:
    #     pkl.dump(std_fc_err,f)
    # with open(f'/home/ggangmei/my-vast/NO_paper/{folder}/mean_feta_err.pkl','wb') as f:
    #     pkl.dump(mean_feta_err,f)
    # with open(f'/home/ggangmei/my-vast/NO_paper/{folder}/std_feta_err.pkl','wb') as f:
    #     pkl.dump(std_feta_err,f)
    with open(f'/home/ggangmei/my-vast/{folder}/meanTotal_err.pkl','wb') as f:
        pkl.dump(meanTotal_err,f)
    with open(f'/home/ggangmei/my-vast/{folder}/stdTotal_err.pkl','wb') as f:
        pkl.dump(stdTotal_err,f)
    with open(f'/home/ggangmei/my-vast/{folder}/meanc_err.pkl','wb') as f:
        pkl.dump(meanc_err,f)
    with open(f'/home/ggangmei/my-vast/{folder}/stdc_err.pkl','wb') as f:
        pkl.dump(stdc_err,f)
    with open(f'/home/ggangmei/my-vast/{folder}/meaneta_err.pkl','wb') as f:
        pkl.dump(meaneta_err,f)
    with open(f'/home/ggangmei/my-vast/{folder}/stdeta_err.pkl','wb') as f:
        pkl.dump(stdeta_err,f)
    with open(f'/home/ggangmei/my-vast/{folder}/boxplot_Ctest.pkl','wb') as f:
        pkl.dump(boxplot_Ctest,f)
    with open(f'/home/ggangmei/my-vast/{folder}/boxplot_ETAtest.pkl','wb') as f:
        pkl.dump(boxplot_ETAtest,f)
    with open(f'/home/ggangmei/my-vast/{folder}/boxplot_Ttest.pkl','wb') as f:
        pkl.dump(boxplot_Ttest,f)
=======
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
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503




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
<<<<<<< HEAD
        #pred_y= pred_y_pad[..., :-padding, :,:,:].reshape(data_y.shape)
        #pred_y = pred_y.cpu().numpy()
        
        
        #print(pred_y.shape) 
        #preds_y=np.vstack([preds_y,pred_y])
       #torch.Size([1, 128, 101, 2])
        #print(preds_y.shape)
        #print(pred_y[:,:,0,1])
        #print(pred_y.dtype)
=======
  
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
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

<<<<<<< HEAD
data_dir = f'my-vast/{folder}/Data_output'
=======
data_dir = f'home/{folder}/Data'
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
data_filename = 'data1.npz'
data_path = os.path.join(data_dir, data_filename)
save_data(data_path, test_x, test_y, preds_y)



#%%
<<<<<<< HEAD
data_dir = f'my-vast/{folder}'
=======
data_dir = f'home/{folder}'
>>>>>>> e5128782df310be638c747bcb8ed90690ab96503
data_filename = 'model.pth'
data_path = os.path.join(data_dir, data_filename)
torch.save(model.state_dict(),data_path)



