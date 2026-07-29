import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.selector import *
from utils.retList import *

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


embeddingList, datasetList = retList(args.dataset)


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