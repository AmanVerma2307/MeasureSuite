import random
import torch
import numpy as np
from model import *
from epochs.epoch import *
from loss.icgd import *
from utils.parser import parse
from utils.lrScheduler import getScheduler
from data import dataLoader
from utils.optimizer import getOptimizer

args = parse()

if(args.reproducibility == 1):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

if(args.trackMetrics == 1):
    wandb.init(project='dgbqaCodebase',name=args.exp_name)

if(args.dataset == 'soli'):
    input_dim = 32
    patch_size = (5,5,5)
    T = 40
    H = 32
    W = 32
    C = 4
    d_model = args.d_model
    dff = args.dff
    num_heads = args.res3dvivit_heads
    num_encoders = args.num_encoders
    rate = 0.1
    G = 11
    I = 10

if(args.dataset in ['ntu_60', 'ntu_120']):
    T = args.ntu_numFrames
    H = None
    W = None
    C = None

    if(args.dataset == 'ntu_60'):
        G = 6
        I = 40

    if(args.dataset == 'ntu_120'):
        G = 4
        I = 69

train_dataLoader, test_dataLoader = dataLoader(args)

model = quantModel(args,
                   T=T,
                   H=H,
                   W=W,
                   C=C,
                   G=G,
                   I=I)

if(args.multi_gpu == 0):
    device = torch.device(args.device)
criterion_hgr = torch.nn.CrossEntropyLoss()
criterion_id = torch.nn.CrossEntropyLoss()
criterion_icgd = icgdLossIterator(G,I)
criterion_icgd.requires_grad_ = False

model = model.to(device)
wandb.watch(model,criterion_id,log="all",log_freq=1)

optimizer = getOptimizer(args, model)
scheduler = getScheduler(optimizer,args)

train_metrics, val_metrics = train_val(train_dataLoader,
                                       test_dataLoader,
                                       model,
                                       optimizer,
                                       criterion_hgr,
                                       criterion_id,
                                       criterion_icgd,
                                       args,
                                       scheduler)

np.savez_compressed('./_store/modelHistory/'+args.exp_name+'_trainMetrics.npz',np.array(train_metrics))
np.savez_compressed('./_store/modelHistory/'+args.exp_name+'_valMetrics.npz',np.array(val_metrics))
 