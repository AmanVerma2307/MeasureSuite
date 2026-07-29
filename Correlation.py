import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from selector import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',
                    type=str,
                    default='soli',
                    help="Dataset to be analyzed")
parser.add_argument('--mode',
                    type=str,
                    default='corrPlots',
                    help="Mode of correlation analysis")
args = parser.parse_args()

if(args.dataset == 'soli'):
    embeddingList = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-ViViT_1-pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-1_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-ViViT_1pt5-1pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_pt5-pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_pt5-1_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_pt5-1pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_1-pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_1-1_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_1-1pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-pt5_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-1_SOLI.npz',
                        './Embeddings/DGBQA_CGID_Res3D-MF_1pt5-1pt5_SOLI.npz',
                        './Embeddings/MS_TPN_pt5-pt5_SOLI.npz',
                        './Embeddings/MS_TPN_pt5-1_SOLI.npz',
                        './Embeddings/MS_TPN_pt5-1pt5_SOLI.npz',
                        './Embeddings/MS_TPN_1-pt5_SOLI.npz',
                        './Embeddings/MS_TPN_1-1_SOLI.npz',
                        './Embeddings/MS_TPN_1-1pt5_SOLI.npz',
                        './Embeddings/MS_TPN_1pt5-pt5_SOLI.npz',
                        './Embeddings/MS_TPN_1pt5-1_SOLI.npz',
                        './Embeddings/MS_TPN_1pt5-1pt5_SOLI.npz',
                        './Embeddings/MS_TAM_pt5-pt5_SOLI.npz',
                        './Embeddings/MS_TAM_1-pt5_SOLI.npz',
                        './Embeddings/MS_TAM_1-1_SOLI.npz',
                        './Embeddings/MS_MViT_pt5-pt5_SOLI.npz',
                        './Embeddings/MS_MViT_pt5-1_SOLI.npz',
                        './Embeddings/MS_MViT_pt5-1pt5_SOLI.npz',
                        './Embeddings/MS_MViT_1-pt5_SOLI.npz',
                        './Embeddings/MS_MViT_1-1_SOLI.npz',
                        './Embeddings/MS_MViT_1-1pt5_SOLI.npz',
                        './Embeddings/MS_MViT_1pt5-pt5_SOLI.npz',
                        './Embeddings/MS_MViT_1pt5-1_SOLI.npz',
                        './Embeddings/MS_MViT_1pt5-1pt5_SOLI.npz']
    datasetList = ['Soli']*39

if(args.dataset == 'handLogin'):
    embeddingList = ['./Embeddings/MS_ViViT_pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_ViViT_pt5-1_HandLogin.npz',
                        './Embeddings/MS_ViViT_pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_ViViT_pt5-2pt5_HandLogin.npz',
                        './Embeddings/MS_ViViT_1-pt5_HandLogin.npz',
                        './Embeddings/MS_ViViT_1-1_HandLogin.npz',
                        './Embeddings/MS_ViViT_1-1pt5_HandLogin.npz',
                        './Embeddings/MS_ViViT_1-2pt5_HandLogin.npz',
                        './Embeddings/MS_ViViT_1pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_ViViT_1pt5-1_HandLogin.npz',
                        './Embeddings/MS_ViViT_1pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_ViViT_1pt5-2pt5_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1-pt5_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1-1_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1-1pt5_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1-2pt5_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1pt5-pt5_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1pt5-1_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1pt5-1pt5_HandLogin.npz',
                        './Embeddings/Test/DGBQA_CGID_Res3D-MF_1pt5-2pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_pt5-1_HandLogin.npz',
                        './Embeddings/MS_TPN_pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_pt5-2pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_1-pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_1-1pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_1-2pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_1pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_1pt5-1_HandLogin.npz',
                        './Embeddings/MS_TPN_1pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_TPN_1pt5-2pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_pt5-1_HandLogin.npz',
                        './Embeddings/MS_TAM_pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_pt5-2pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_1-pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_1-1_HandLogin.npz',
                        './Embeddings/MS_TAM_1-1pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_1-2pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_1pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_1pt5-1_HandLogin.npz',
                        './Embeddings/MS_TAM_1pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_TAM_1pt5-2pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_pt5-1_HandLogin.npz',
                        './Embeddings/MS_MViT_pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_pt5-2pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_1-pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_1-1_HandLogin.npz',
                        './Embeddings/MS_MViT_1-1pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_1-2pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_1pt5-pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_1pt5-1_HandLogin.npz',
                        './Embeddings/MS_MViT_1pt5-1pt5_HandLogin.npz',
                        './Embeddings/MS_MViT_1pt5-2pt5_HandLogin.npz']
    datasetList = ['HandLogin']*55

if(args.dataset == 'tiny'):
    embeddingList = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz',
                    './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1pt5_Tiny.npz',
                    './Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-2pt5_Tiny.npz',
                    './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1_Tiny.npz',
                    './Embeddings/DGBQA_CGID_Res3D-ViViT_1-1pt5_Tiny.npz',
                    './Embeddings/DGBQA_CGID_Res3D-ViViT_1-2pt5_Tiny.npz',
                    './Embeddings/MS_MF_1-1_Tiny.npz',
                    './Embeddings/MS_MF_1-1pt5_Tiny.npz',
                    './Embeddings/MS_MF_1-2pt5_Tiny.npz',
                    './Embeddings/MS_TAM_1-1_Tiny.npz',
                    './Embeddings/MS_TAM_1-1pt5_Tiny.npz',
                    './Embeddings/MS_TAM_1-2pt5_Tiny.npz']
    datasetList = ['Tiny']*12




if(args.mode == 'corrPlots'):

    measureVal = get_params(embeddingList,
                            datasetList,
                            'full')
    df = make_df(np.array(measureVal))

    labels = ['$\\mathcal{r}$',
            'R',
            '$\\psi$',
            '$C_{d}$',
            '$Ar(\Delta)$',
            '$Ar(\Delta)*\\bar{O}$',
            '$Ar(\Delta)*\\bar{\psi}$',
            '$\\bar{\psi}*\\bar{O}$',
            '$nAr^{*}(\Delta)$']

    rocket = sns.color_palette("rocket")
    sns.jointplot(x="rpp",
                  y="Ar_star",
                  data=df,
                  kind="reg",
                  color=rocket[5])
    plt.ylabel('$nA_r^{*}(\Delta)$',fontsize=14)
    plt.xlabel('RPP',fontsize=14)
    plt.show()