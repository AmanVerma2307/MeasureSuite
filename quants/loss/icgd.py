import torch
import numpy as np

class icgdLoss(torch.nn.Module):

    def __init__(self,G,I):
        super().__init__()
        self.G = G # Total number of gestures
        self.I = I # Total number of Identities

    def get_HGRmask(self,y_hgr):
        return (y_hgr.unsqueeze(-1) != y_hgr).to(dtype=torch.float32)

    def get_IDmask(self,y_id):

        """
        ID mask of ICGD loss

        INPUTS:-
        1) y_id: Input ID labels of shape (B,)
        2) device: Processing device

        OUTPUTS:-
        1) id_mask: ID mask of shape (B,B). High if different samples but same identities.
        """

        ##### Similar identity mask
        id_mask = (y_id.unsqueeze(-1) == y_id).to(dtype=torch.float32)

        ##### Distinct position mask
        B = y_id.size(0) # batch_size
        device_id = torch.device(id_mask.get_device())
        id_mask_dist = torch.logical_not(torch.eye(B).to(torch.bool)).to(device_id,dtype=torch.float32) # Except diagonal entries everything turned high
        id_mask_dist.requires_grad = False # Making this a non-parametric model.

        ##### ID mask
        return torch.logical_and(id_mask,id_mask_dist).to(dtype=torch.float32)
    
    def forward(self,y_hgr,y_id,f_theta):

        """
        ICGD Loss

        INPUTS:-
        1) y_hgr: HGR labels of shape (B,)
        2) y_id: ID labels of shape (B,)
        3) f_theta: Output embeddings of shape (B,d)

        OUTPUTS:-
        1) loss: ICGD Loss for the batch
        """

        ##### Gram matrix formation
        f_theta = torch.nn.functional.normalize(f_theta,dim=-1) # L2 normalization
        #f_theta*(1/torch.norm(f_theta,dim=-1).unsqueeze(-1)) # L2 normalization
        G_bar = torch.matmul(f_theta,torch.permute(f_theta,(1,0))) # Gram matrix -> (B,B)

        ##### HGR mask
        hgr_mask = self.get_HGRmask(y_hgr) # HGR mask, shape -> (B,B)

        ##### ID mask
        id_mask = self.get_IDmask(y_id) # ID mask, shape -> (B,B)

        #print(torch.sum(hgr_mask).item())
        #print(torch.sum(id_mask).item())
        #print(torch.sum((G_bar*hgr_mask))/torch.sum(hgr_mask)) # -> ICGD Score

        ##### Negative mask
        gamma = (G_bar > 0).to(dtype=torch.float32) # Masks negative values to zero. Shape -> (B,B)

        ##### Masking the Gram matrix
        G_bar_total = torch.triu(gamma*hgr_mask*id_mask*G_bar,diagonal=1) # Shape -> (B,B)
        norm_factor = torch.triu(gamma*hgr_mask*id_mask,diagonal=1) # Shape -> (B,B)

        ##### Loss computation
        #print('Sum: '+str(torch.sum(G_bar).item()))
        #print('Factor: '+str((torch.sum(norm_factor)+1).item()))
        return (torch.sum(G_bar_total)/(torch.sum(norm_factor)+1))

class icgdLossIterator(torch.nn.Module):

    def __init__(self,G,I):
        super().__init__()
        self.G = G # Total number of gestures
        self.I = I # Total number of Identities

    def get_HGRmask(self,y_hgr):
        return (y_hgr.unsqueeze(-1) != y_hgr).to(dtype=torch.float32)

    def get_IDmask(self,y_id):

        """
        ID mask of ICGD loss

        INPUTS:-
        1) y_id: Input ID labels of shape (B,)
        2) device: Processing device

        OUTPUTS:-
        1) id_mask: ID mask of shape (B,B). High if different samples but same identities.
        """

        ##### Similar identity mask
        id_mask = (y_id.unsqueeze(-1) == y_id).to(dtype=torch.float32)
        id_mask_curr = (y_id.unsqueeze(-1) == 1).to(dtype=torch.float32)

        ##### Distinct position mask
        B = y_id.size(0) # batch_size
        device_id = torch.device(id_mask.get_device())
        id_mask_dist = torch.logical_not(torch.eye(B).to(torch.bool)).to(device_id,dtype=torch.float32) # Except diagonal entries everything turned high
        id_mask_dist.requires_grad = False # Making this a non-parametric model.

        ##### ID mask
        return torch.logical_and(id_mask,id_mask_dist*id_mask_curr).to(dtype=torch.float32)
    
    def forward(self,y_hgr,y_id,f_theta):

        """
        ICGD Loss

        INPUTS:-
        1) y_hgr: HGR labels of shape (B,)
        2) y_id: ID labels of shape (B,I)
        3) f_theta: Output embeddings of shape (B,d)

        OUTPUTS:-
        1) loss: ICGD Loss for the batch
        """

        ##### One-hot ID labels
        y_id_ohot = torch.nn.functional.one_hot(y_id.to(dtype=torch.int64),
                                                num_classes=self.I)

        ##### Gram matrix formation
        f_theta = torch.nn.functional.normalize(f_theta,dim=-1) # L2 normalization
        G_bar = torch.matmul(f_theta,torch.permute(f_theta,(1,0))) # Gram matrix -> (B,B)

        ##### Negative mask
        gamma = (G_bar > 0).to(dtype=torch.float32) # Masks negative values to zero. Shape -> (B,B)

        ##### HGR mask
        hgr_mask = self.get_HGRmask(y_hgr) # HGR mask, shape -> (B,B)

        ##### Loss computation
        #### Defining essentials
        lossVal = 0.0

        for sub_idx in range(self.I):

            y_id_curr = y_id_ohot[:,sub_idx]
            id_mask = self.get_IDmask(y_id_curr)

            entanglemenVal = torch.sum(torch.triu(gamma*hgr_mask*id_mask*G_bar,diagonal=1))
            normVal = torch.sum(torch.triu(gamma*hgr_mask*id_mask,diagonal=1))
            lossVal = lossVal + (entanglemenVal/(normVal+1))

        return lossVal/self.I

if __name__ == "__main__":

    device = torch.device("cuda:0")

    f_theta = torch.Tensor(np.load('./data/soli/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz')['arr_0']).to(device)
    y_hgr = torch.Tensor(np.load('./data/soli/y_dev_DeltaDistance_SOLI.npz')['arr_0']).to(device)
    y_id = torch.Tensor(np.load('./data/soli/y_dev_id_DeltaDistance_SOLI.npz')['arr_0']).to(device)

    icgd_loss = icgdLossIterator(11,10).to(device)
    loss = icgd_loss(y_hgr,y_id,f_theta)
    print(loss.item())
    
    ##### Gramian matrix
    #a = torch.normal(0,1,size=(10,32))
    #b = torch.norm(a,dim=-1)
    #c = a*(1/b.unsqueeze(-1))
    #d = torch.matmul(c,torch.permute(c,(1,0)))

    ##### Positive mask
    #gamma = (d > 0).to(dtype=torch.float32)
    
    ##### Lower triangular masking
    #L_d = torch.triu(gamma,diagonal=1)

    ##### HGR mask
    #y_hgr = torch.randint(low=0,
    #                   high=11,
    #                    size=(32,))
    #print(y_hgr)
    #print(hgr_mask(y_hgr))

    ##### ID mask
    #y_id = torch.randint(low=0,high=11,size=(10,))
    #print(y_id)
    #print(id_mask(y_id))




