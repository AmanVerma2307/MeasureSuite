import torch

def getScheduler(optimizer,args):

    """
    Function to get scheduler
    """

    if(args.lrScheduler == 1):
        if(args.lrScheduler_mode == 'step'):
            return torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=args.lrScheduler_stepGamma)
        if(args.lrScheduler_mode == 'multiStep'):
            return torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[25,60,80,90], gamma=args.lrScheduler_stepGamma)
        if(args.lrScheduler_mode == 'exponential'):
            return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        if(args.lrScheduler_mode == 'plateau'):
            return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                              mode='min', 
                                                              factor=args.lrScheduler_stepGamma, 
                                                              patience=10, 
                                                              cooldown=2)
    else:
        return None
    