import tqdm
import torch
import numpy as np

def eval(dataloader,
         model,
         device,
         args,
         writer=False):
    
    """
    Returns predictions and embeddings for the model
    
    OUTPUTS:-
    1) g_hgr: HGR predictions of shape (N,G)
    2) g_id: ID predictions of shape (N,I)
    3) f_theta: Output embeddings of shape (N,d_model)
    4) resultFile: File with HGR accuracy, ID accuracy, and confusion matrix values
    """

    g_hgr = []
    g_id = []
    f_theta = []
    acc_hgr = 0.0
    acc_id = 0.0
    total_samples = 0.0

    for batch_idx, (x,y_hgr,y_id) in enumerate(tqdm.tqdm(dataloader,colour='green')):

        x = x.to(device)
        y_hgr = y_hgr.to(device)
        y_id = y_id.to(device)

        model.eval()
        with torch.set_grad_enabled(False):
            dense_hgr, dense_id, embeddings = model(x)

        for hgr_item, id_item, f_theta_item in zip(dense_hgr.detach().cpu().numpy(),
                                                   dense_id.detach().cpu().numpy(),
                                                   embeddings.detach().cpu().numpy()):
            
            g_hgr.append(hgr_item)
            g_id.append(id_item)
            f_theta.append(f_theta_item)
            
        acc_hgr = acc_hgr + (torch.sum(y_hgr == torch.argmax(dense_hgr,dim=-1))).detach().item()
        acc_id = acc_id + (torch.sum(y_id == torch.argmax(dense_id,dim=-1))).detach().item()
        total_samples = total_samples + x.size(0)

    acc_hgr = acc_hgr/total_samples 
    acc_id = acc_id/total_samples

    if(writer == True):
        print('HGR Acc: '+str(acc_hgr)) # HGR Accuracy
        print('ID Acc: '+str(acc_id)) # ID Accuracy

        result_file = open('./_store/resultFiles/'+args.exp_name+'.txt','w')
        result_file.write('HGR Acc: '+str(acc_hgr)+"\n")
        result_file.write('ID Acc: '+str(acc_id)+"\n")
        result_file.close()

    return np.array(hgr_item), np.array(id_item), np.array(f_theta)
