from types import SimpleNamespace
from timm.optim.optim_factory import create_optimizer

def getOptimizer(args, model):

    """
    Function to select optimizer

    INPUTS:-
    1) args: The training/testing arguments
    2) model: The trainig model

    OUTPUTS:-
    1) optim: The optimzer
    """

    argsOptim = SimpleNamespace()

    if(args.optimizer == 'adam'):
        argsOptim.weight_decay = 0
        argsOptim.lr = args.lr
        argsOptim.opt = 'adam'
        argsOptim.momentum = 0.9
        argsOptim.eps = 1e-7

    if(args.optimizer == 'nadam'):
        argsOptim.weight_decay = 0
        argsOptim.lr = args.lr
        argsOptim.opt = 'nadam'
        argsOptim.momentum = 0.9
        argsOptim.eps = 1e-7

    if(args.optimizer == 'adamp'):
        argsOptim.lr = args.lr
        argsOptim.weight_decay = 0
        argsOptim.opt = 'adamp'
        argsOptim.momentum = 0.9
        argsOptim.eps = 1e-7

    if(args.optimizer == 'novograd'):
        argsOptim.lr = args.lr
        argsOptim.weight_decay = 0
        argsOptim.opt = 'novograd'
        argsOptim.momentum = 0.9
        argsOptim.eps = 1e-7

    if(args.optimizer == 'radam'):
        argsOptim.lr = args.lr
        argsOptim.weight_decay = 0
        argsOptim.opt = 'radam'
        argsOptim.momentum = 0.9
        argsOptim.eps = 1e-7

    if(args.optimizer == 'adamw'):
        argsOptim.lr = args.lr
        argsOptim.weight_decay = 0
        argsOptim.opt = 'adamw'
        argsOptim.momentum = 0.9
        argsOptim.eps = 1e-7    

    if(args.optimizer == 'adafactor'):
        argsOptim.lr = args.lr
        argsOptim.weight_decay = 0
        argsOptim.opt = 'adafactor'
        argsOptim.momentum = 0.9
        argsOptim.eps = 1e-7      

    if(args.optimizer == 'adahessian'):
        argsOptim.lr = args.lr
        argsOptim.weight_decay = 0
        argsOptim.opt = 'adahessian'
        argsOptim.momentum = 0.9
        argsOptim.eps = 1e-7

    optim = create_optimizer(argsOptim, model)
    print(optim)
    return optim
