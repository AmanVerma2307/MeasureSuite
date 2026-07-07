import time
import tqdm
import wandb
import torch

def train_epoch(dataloader, 
                model, 
                optimizer, 
                obj_hgr, 
                obj_id,
                obj_icgd,
                device,
                args):
    
    """
    Function to train the netwok on a single epoch
    """

    loss_hgr = 0.0
    loss_id = 0.0
    loss_icgd = 0.0
    loss = 0.0
    acc_hgr = 0.0
    acc_id = 0.0
    total_samples = 0.0

    model.train()
    
    for batch_idx, (x,y_hgr,y_id) in enumerate(tqdm.tqdm(dataloader,colour='blue')):
        
        x = x.to(device)
        y_hgr = y_hgr.type(torch.LongTensor).to(device)
        y_id = y_id.type(torch.LongTensor).to(device)
    
        #optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            dense_hgr, dense_id, f_theta = model.forward(x)

            loss_hgr_batch = obj_hgr(dense_hgr,y_hgr) # HGR Loss
            loss_id_batch = obj_id(dense_id,y_id) # ID Loss
            loss_icgd_batch = obj_icgd(y_hgr,y_id,f_theta) # ICGD Loss

            #print(loss_hgr_batch.item(),loss_id_batch.item(),loss_icgd_batch.item())

            loss_batch = args.lambda_hgr*loss_hgr_batch + args.lambda_id*loss_id_batch + args.lambda_icgd*loss_icgd_batch
            loss_batch.backward()
            
            optimizer.step()
            optimizer.zero_grad()

        loss_hgr = loss_hgr + loss_hgr_batch.detach().item()*x.size(0)
        loss_id = loss_id + loss_id_batch.detach().item()*x.size(0)
        loss_icgd = loss_icgd + loss_icgd_batch.detach().item()*x.size(0)
        loss = loss + (args.lambda_hgr*loss_hgr_batch + args.lambda_id*loss_id_batch + args.lambda_icgd*loss_icgd_batch).detach().item()*x.size(0)
        acc_hgr = acc_hgr + (torch.sum(y_hgr == torch.argmax(dense_hgr,dim=-1))).detach().item()
        acc_id = acc_id + (torch.sum(y_id == torch.argmax(dense_id,dim=-1))).detach().item()
        total_samples = total_samples + x.size(0)

    return loss_hgr/total_samples, loss_id/total_samples, loss_icgd/total_samples, loss/total_samples, acc_hgr/total_samples, acc_id/total_samples

def val_epoch(dataloader, 
                model, 
                obj_hgr, 
                obj_id,
                obj_icgd,
                device,
                args):
    
    """
    Function to test the netwok on a single epoch
    """

    loss_hgr = 0.0
    loss_id = 0.0
    loss_icgd = 0.0
    loss = 0.0
    acc_hgr = 0.0
    acc_id = 0.0
    total_samples = 0.0

    for batch_idx, (x,y_hgr,y_id) in enumerate(tqdm.tqdm(dataloader,colour='green')):
        
        x = x.to(device)
        y_hgr = y_hgr.type(torch.LongTensor).to(device)
        y_id = y_id.type(torch.LongTensor).to(device)

        model.eval()
        with torch.set_grad_enabled(False):

            dense_hgr, dense_id, f_theta = model.forward(x)

            loss_hgr_batch = obj_hgr(dense_hgr,y_hgr) # HGR Loss
            loss_id_batch = obj_id(dense_id,y_id) # ID Loss
            loss_icgd_batch = obj_icgd(y_hgr,y_id,f_theta) # ICGD Loss

            #print(loss_hgr_batch.item(),loss_id_batch.item(),loss_icgd_batch.item())

            #loss_batch = loss_hgr_batch + args.lambda_id*loss_id_batch + args.lambda_icgd*loss_icgd_batch

        loss_hgr = loss_hgr + loss_hgr_batch.detach().item()*x.size(0)
        loss_id = loss_id + loss_id_batch.detach().item()*x.size(0)
        loss_icgd = loss_icgd + loss_icgd_batch.detach().item()*x.size(0)
        loss = loss + (args.lambda_hgr*loss_hgr_batch + args.lambda_id*loss_id_batch + args.lambda_icgd*loss_icgd_batch).detach().item()*x.size(0)
        acc_hgr = acc_hgr + (torch.sum(y_hgr == torch.argmax(dense_hgr,dim=-1))).detach().item()
        acc_id = acc_id + (torch.sum(y_id == torch.argmax(dense_id,dim=-1))).detach().item()
        total_samples = total_samples + x.size(0)

    return loss_hgr/total_samples, loss_id/total_samples, loss_icgd/total_samples, loss/total_samples, acc_hgr/total_samples, acc_id/total_samples

def train_val(train_loader,
              val_loader, 
              model, 
              optimizer, 
              obj_hgr, 
              obj_id,
              obj_icgd,
              args,
              scheduler=None):
    
    """
    Function to train and validate in a single GPU setting
    """

    model_path = './_store/weights/'+args.exp_name+'.pth'
    loss_best = 1e+6
    device = torch.device(args.device)

    loss_hgr = []
    loss_id = []
    loss_icgd = []
    loss = []
    acc_hgr = []
    acc_id = []
    val_loss_hgr = []
    val_loss_id = []
    val_loss_icgd = []
    val_loss = []
    val_acc_hgr = []
    val_acc_id = []
    train_metrics = []
    val_metrics = []

    for epoch in range(args.num_epochs):

        time_start = time.time()
        print(f'Epoch {epoch+1}/{args.num_epochs}')
        print('-' * 10)

        ##### Training
        loss_hgr_train_curr, loss_id_train_curr, loss_icgd_train_curr, loss_train_curr, acc_hgr_train_curr, acc_id_train_curr = train_epoch(train_loader,
                                                                                                                                model,
                                                                                                                                optimizer,
                                                                                                                                obj_hgr,
                                                                                                                                obj_id,
                                                                                                                                obj_icgd,
                                                                                                                                device,
                                                                                                                                args)
        loss_hgr.append(loss_hgr_train_curr)
        loss_id.append(loss_id_train_curr)
        loss_icgd.append(loss_icgd_train_curr)
        loss.append(loss_train_curr)
        acc_hgr.append(acc_hgr_train_curr)
        acc_id.append(acc_id_train_curr)

        ##### Validation
        loss_hgr_val_curr, loss_id_val_curr, loss_icgd_val_curr, loss_val_curr, acc_hgr_val_curr, acc_id_val_curr = val_epoch(val_loader,
                                                                                                                            model,
                                                                                                                            obj_hgr,
                                                                                                                            obj_id,
                                                                                                                            obj_icgd,
                                                                                                                            device,
                                                                                                                            args)
        
        if(args.lrScheduler == 1):
            if(args.lrScheduler_mode == 'plateau'):
                scheduler.step(loss_val_curr)
            else:
                scheduler.step()

        if(loss_val_curr < loss_best):
            loss_best = loss_val_curr
            torch.save(model.state_dict(),model_path)

        val_loss_hgr.append(loss_hgr_val_curr)
        val_loss_id.append(loss_id_val_curr)
        val_loss_icgd.append(loss_icgd_val_curr)
        val_loss.append(loss_val_curr)
        val_acc_hgr.append(acc_hgr_val_curr)
        val_acc_id.append(acc_id_val_curr)

        ##### Outputs
        print('Total time:'+str(time.time() - time_start))
        print('Loss: '+str(loss_train_curr))
        print('Validation Loss: '+str(loss_val_curr))
        print('HGR Accuracy: '+str(acc_hgr_train_curr))
        print('ID Accuracy: '+str(acc_id_train_curr))
        print('Validation HGR Accuracy: '+str(acc_hgr_val_curr))
        print('Validation ID Accuracy: '+str(acc_id_val_curr))

        wandb.log({'epoch':epoch,
                   'loss_hgr':loss_hgr_train_curr,
                   'loss_id':loss_id_train_curr,
                   'loss_icgd':loss_icgd_train_curr,
                   'loss':loss_train_curr,
                   'acc_hgr':acc_hgr_train_curr,
                   'acc_id':acc_id_train_curr,
                   'val_loss_hgr':loss_hgr_val_curr,
                   'val_loss_id':loss_id_val_curr,
                   'val_loss_icgd':loss_icgd_val_curr,
                   'val_loss':loss_val_curr,
                   'val_acc_hgr':acc_hgr_val_curr,
                   'val_acc_id':acc_id_val_curr,
                   })

    ##### Storing
    train_metrics.append(loss_hgr)
    train_metrics.append(loss_id)
    train_metrics.append(loss_icgd)
    train_metrics.append(loss)
    train_metrics.append(acc_hgr)
    train_metrics.append(acc_id)

    val_metrics.append(val_loss_hgr)
    val_metrics.append(val_loss_id)
    val_metrics.append(val_loss_icgd)
    val_metrics.append(val_loss)
    val_metrics.append(val_acc_hgr)
    val_metrics.append(val_acc_id)

    return train_metrics, val_metrics
