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
parser.add_argument('--corrPlot1',
                    type=str,
                    default='r',
                    help="The first measure for correlation plots")
parser.add_argument('--corrPlot2',
                    type=str,
                    default='Ar_star',
                    help="The second measure for correlation plots")
parser.add_argument('--corrPlot_pallete',
                    type=str,
                    default='rocket',
                    help="The color pallete to be used for correlation plots")

args = parser.parse_args()

embeddingList, datasetList = retList(args.dataset)
labels = {'r':'$\\mathcal{r}$',
           'relevance':'R',
            'psi':'$\\psi$',
            'Cd':'$C_{d}$',
            'Ar_star':'$nAr^{*}(\Delta)$',
            'euclid':'Euclidean dist.',
            'Kendall':'$\\tau$',
            'DCG':'nDCG',
            'err':'err',
            'U':'U measure',
            'gre':'GRE',
            'infAp':'infAp',
            'rpp':'RPP'}


if(args.mode == 'corrPlots'):
    measureVal = get_params(embeddingList,
                            datasetList,
                            'full')
    df = make_df(np.array(measureVal))

    rocket = sns.color_palette(args.corrPlot_pallete)
    sns.jointplot(x=args.corrPlot1,
                  y=args.corrPlot2,
                  data=df,
                  kind="reg",
                  color=rocket[5])
    plt.xlabel(labels[args.corrPlot1],fontsize=14)
    plt.ylabel(labels[args.corrPlot2],fontsize=14)
    plt.show()