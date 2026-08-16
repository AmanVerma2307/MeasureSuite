import argparse

def parse():
    parser = argparse.ArgumentParser()

    #### General arguments
    parser.add_argument('--reproducibility',
                        type=int,
                        default=1,
                        help="If 1, then random seeding")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="Random seed to ensure reproducibility")
    parser.add_argument('--trackMetrics',
                        type=int,
                        default=1,
                        help="If 1, then metrics will be tracked with wandb")
    parser.add_argument('--multi_gpu',
                        type=bool,
                        default=False,
                        help="If True, then multi GPU training/testing")
    parser.add_argument('--device',
                        type=str,
                        default='cuda:0',
                        help="Device to be used in single gpu setting")
    parser.add_argument('--exp_name',
                        type=str,
                        help="Name of the experiment")

    #### Data arguments
    parser.add_argument('--dataset',
                        type=str,
                        default='soli',
                        help="Dataset to be used for train/test: [soli,tiny,handLogin,scut]")
    parser.add_argument('--shuffle',
                        type=bool,
                        default=False,
                        help="Shuffling in the dataset")

    #### NTU arguments
    parser.add_argument('--ntu_numFrames',
                        type=int,
                        default=120,
                        help="Total number of frames in ntu dataset")

    #### Model arguments
    parser.add_argument('--model',
                        type=str,
                        default='res3dViViT',
                        help="Model to be used")
    parser.add_argument('--RGB',
                        type=int,
                        default=1,
                        help="If True, then the input has RGB channels. Default: True")
    parser.add_argument('--motionModel',
                        type=int,
                        default=0,
                        help="If True, then the input has motion maps. Default: False")

    ### Res3D-ViViT arguments
    parser.add_argument('--res3dvivit_heads',
                        type=int,
                        default=2,
                        help="Number of heads to be used")
    parser.add_argument('--num_encoders',
                        type=int,
                        default=2,
                        help="Number of encoders to be used")
    parser.add_argument('--d_model',
                        type=int,
                        default=32,
                        help="Embedding dimensions")
    parser.add_argument('--dff',
                        type=int,
                        default=256,
                        help="Embedding hidden dimensions")
    
    #### Training arguments
    parser.add_argument('--num_epochs',
                        type=int,
                        default=50,
                        help="Number of training epochs")
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help="Batch size")
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help="Optimizer to be used for training.")
    parser.add_argument('--lr',
                        type=float,
                        default=1e-4,
                        help="The learning rate to be used in the optimizer.")
    parser.add_argument('--lambda_hgr',
                        type=float,
                        default=1.0,
                        help="Weight of HGR loss")
    parser.add_argument('--lambda_id',
                        type=float,
                        default=1.0,
                        help="Weight of ID loss")
    parser.add_argument('--lambda_icgd',
                        type=float,
                        default=1.0,
                        help="Weight of ICGD loss")
    
    #### Scheduler arguments
    parser.add_argument('--lrScheduler',
                        type=int,
                        default=0,
                        help="If True, then lrScheduler is used. Default: False")
    parser.add_argument('--lrScheduler_mode',
                        type=str,
                        default='step',
                        help="Mode of lrScheduler")
    parser.add_argument('--lrScheduler_stepGamma',
                        type=float,
                        default=0.5,
                        help="The reduction factor for learning rate with each step's iteration")
    
    
    args = parser.parse_args()
    return args