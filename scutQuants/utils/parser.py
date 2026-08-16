import argparse

def parseArgs():

    parser = argparse.ArgumentParser()

    parser.add_argument("--lambda_id",
                        type=float,
                        help="Scaling Value of ID Loss")
    parser.add_argument("--lambda_cgid",
                        type=float,
                        help="Scaling Value of CGID Loss")
    parser.add_argument("--local_batch_size",
                        type=int,
                        help="Batch Size to used for a device")
    parser.add_argument("--exp_name",
                        type=str,
                        help="Name of the Experiment being run, will be used saving the model and correponding outputs")
    parser.add_argument("--numEpochs",
                        type=int,
                        default=100,
                        help="Number of epochs to run")
    parser.add_argument("--modelChoice",
                        type=str,
                        default='vivit',
                        help="Number of epochs to run")

    args = parser.parse_args()

    return args
