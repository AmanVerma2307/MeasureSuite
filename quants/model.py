import torch
import torch.nn.functional as F
from models.vivit import *
from models.i3d import *
from models.fstanet import *
from models.videomae import *
from models.skateformer import *
from models.hcn import *

def getModel(args,
             T,
             H,
             W,
             C):
    """
    Function to get model
    """

    if(args.dataset not in ['ntu_60', 'ntu_120']):

        if(args.model == 'vivit'):
            model = res3dViViT(input_dim=args.input_dim,
                            patch_size=[args.patchSizeT,args.patchSizeH,args.patchSizeW],
                            T=T,
                            H=H,
                            W=W,
                            C=C,
                            d_model=args.d_model,
                            num_heads=args.numHeads,
                            dff=args.dff,
                            rate=args.vivit_rate,
                            num_encoders=args.numEncoders,
                            )
            
        if(args.model == 'i3d'):
            model = i3d()

        if(args.model == 'fstanet'):
            model = fsta(T)

        if(args.model == 'videomae'):
            model = videomae(modelStyle="B")

    if(args.dataset in ['ntu_60', 'ntu_120']):

        if(args.model == 'skateformer'):
            model = SkateFormer_(T=T)

        if(args.model == 'hcn'):
            model = HCN(window_size=T)

    total_params = sum(p.numel() for p in model.parameters())
    print('++++++++++++++++++')
    print('Model: '+str(args.model))
    print(model)
    print('Total parameters: '+str(total_params)) 
    print('++++++++++++++++++')

    return model


class quantModel(torch.nn.Module):
    
    """
    Model defined for pretrained motion maps
    """

    def __init__(self,
                 args,
                 T,
                 H,
                 W,
                 C,
                 G,
                 I):
        
        super().__init__()
        self.args = args # Input arguments
        self.G = G # Number of gestures
        self.I = I # Number of identities
        self.T = T # Number op frames
        self.H = H # Input height dimensions
        self.W = W # Input width dimensions
        self.C = C # Number of channels in the input

        if(self.args.RGB == 1 or self.args.dataset in ['ntu_60','nntu_120']):
            self.backbone = getModel(self.args,
                                     T=T,
                                     H=H,
                                     W=W,
                                     C=3)
        else:
            self.normConv = torch.nn.Conv3d(in_channels=C,out_channels=3,kernel_size=(3,3,3))
            if(self.args.motionModel == 1):
                self.backbone = getModel(self.args,
                                         T=(T-1),
                                         H=H,
                                         W=W,
                                         C=3)
            else:
                self.backbone = getModel(self.args,
                                     T=T,
                                     H=H,
                                     W=W,
                                     C=3)
                
        self.dense_hgr = torch.nn.Linear(self.backbone.embedDims,G)
        self.dense_id = torch.nn.Linear(self.backbone.embedDims,I)

    def forward(self,x):
        if(self.args.RGB == 0):
            x = F.pad(x,(1,1,1,1,1,1))
            x = self.normConv(x)
            x = F.relu(x)
        x = self.backbone(x)
        f_hgr = self.dense_hgr(x)
        f_id = self.dense_id(x)
        return f_hgr, f_id, x

    def predict(self,
                dataLoader,
                args):
        
        """
        Function to predict embeddings and outputs

        INPUTS:-
        1) dataLoader: The testSet loader with N samples
        2) args: Parsed arguments

        OUPUTS:-
        1) y_hgr_preds: Predicted HGR labels of shape (N,)
        2) y_hgr_preds: Predicted ID labels of shape (N,)
        3) f_theta: Predicted embeddings of shape (N,d)
        """

        y_hgr_preds = []
        y_id_preds = []
        embeddings = []
        
        if(args.multi_gpu == 0):
            device = torch.device(args.device)

        for batch_idx, dataSample in enumerate(tqdm.tqdm(dataLoader,colour='yellow')):
            
            self.eval()
            with torch.set_grad_enabled(False):
                dense_hgr, dense_id, f_theta = self.forward(dataSample['data'].to(device))

            for elemPreds_hgr, elemPreds_id, elemEmbeddings in zip(torch.argmax(dense_hgr,dim=-1).detach().cpu().numpy(),
                                                                   torch.argmax(dense_id,dim=-1).detach().cpu().numpy(),
                                                                   f_theta.detach().cpu().numpy()):
                y_hgr_preds.append(elemPreds_hgr)
                y_id_preds.append(elemPreds_id)
                embeddings.append(elemEmbeddings)                

        return np.array(y_hgr_preds), np.array(y_id_preds), np.array(embeddings)
    

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default="fstanet")
    parser.add_argument('--dataset',
                        type=str,
                        default='soli')
    parser.add_argument('--RGB',
                        type=int,
                        default=1,
                        help="If True, then the input has RGB channels. Default: True")
    parser.add_argument('--motionModel',
                        type=int,
                        default=0,
                        help="If True, then the input has motion maps. Default: False")
    parser.add_argument('--embedDims',
                        type=int,
                        default=32,
                        help="Output embedding dimensions of the backbone+FC")

    
    args = parser.parse_args()
    device = torch.device('cuda:0')

    if(args.dataset not in ['ntu_60', 'ntu_120']):
        model = quantModel(args,64,128,128,3,10,11).to(device)
        input = torch.randn(size=(2,3,64,128,128)).to(device)
        op1, op2, op3 = model(input)
        print(op1.size(),op2.size(),op3.size())

    if(args.dataset in ['ntu_60', 'ntu_120']):
        model = quantModel(args,T=120,H=None,W=None,C=3,G=6,I=40).to(device)
        input = torch.randn(size=(100,3,120,25,1)).to(device)
        op1, op2, op3 = model(input)
        print(op1.size(),op2.size(),op3.size())
