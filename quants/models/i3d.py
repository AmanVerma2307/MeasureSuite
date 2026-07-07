import tqdm
import torch
import torch.nn.functional as F
import numpy as np

class i3d(torch.nn.Module):

    """
    I3D Class: Pretrained weights from KINETICS-400
    """

    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/pytorchvideo',
                                       model="i3d_r50",
                                       pretrained=True)
        
        modelList = [child for child in ([child for child in self.backbone.children()][0]).children()]
        self.backbone = torch.nn.Sequential(*modelList[:-1])
        self.pool = torch.nn.AdaptiveAvgPool3d(1)
        self.embedDims = 2048       

    def forward(self,x):
        x = self.backbone(x)
        x = self.pool(x).squeeze((2,3,4))
        #x = F.relu(self.dense(x))
        return x
    
    def predict(self,
                dataLoader,
                args):
        
        """
        Function to predict embeddings and outputs

        INPUTS:-
        1) dataLoader: The testSet loader with N samples
        2) args: Parsed arguments

        OUPUTS:-
        1) f_theta: Predicted embeddings of shape (N,d)
        """

        embeddings = []
        
        if(args.multi_gpu == 0):
            device = torch.device(args.device)

        for batch_idx, dataSample in enumerate(tqdm.tqdm(dataLoader,colour='yellow')):

            x = dataSample['data'].to(device)
            y_id = dataSample['label'].to(device)

            self.eval()
            with torch.set_grad_enabled(False):
                f_theta = self.forward(x)
            
            for elemEmbeddings in f_theta.detach().cpu().numpy():
                embeddings.append(elemEmbeddings)                

        return np.array(embeddings)

        
if  __name__ == "__main__":
    
    device = torch.device('cuda:0')
    model = i3d(143).to(device)
    input = torch.randn(size=(5,3,64,128,128)).to(device)
    op1 = model(input)
    print(op1.shape)

    # model = torch.hub.load('facebookresearch/pytorchvideo',
    #                                    model="i3d_r50",
    #                                    pretrained=True)
    # totalChildren = 0
    # for idx, child in enumerate(model.children()):
    #     print('==================================')
    #     print('#'+str(idx)+': '+str(child))
    #     totalChildren = totalChildren + 1

    #     totalChildrenCurr = 0
    #     for idx, node in enumerate(child.children()):
    #         print('##'+str(idx)+': '+str(node))
    #         totalChildrenCurr = totalChildrenCurr + 1

    #     print('Total nodes in this child: '+str(totalChildrenCurr))
    # print('Total children in the model: '+str(totalChildren))

    # devModel = [child for child in model.children()]
    # devModel1 = [child for child in devModel[0].children()]
    # print(devModel1)

