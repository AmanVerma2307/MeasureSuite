import argparse
import scipy
import pyCompare
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
parser.add_argument('--measure1',
                    type=str,
                    default='r',
                    help="The first measure for analysis")
parser.add_argument('--measure2',
                    type=str,
                    default='Ar_star',
                    help="The second measure for analysis")
parser.add_argument('--corrPlot_palette',
                    type=str,
                    default='rocket',
                    help="The color pallete to be used for correlation plots")
parser.add_argument('--initCorrFile',
                    type=int,
                    default=0,
                    help="If 1, then a new result file will be instantiated.")
parser.add_argument('--nameCorrFile',
                    type=str,
                    help="Name of the CorrFile for storing correlation results")
parser.add_argument('--baPath',
                    type=str,
                    help="Name of the experiment for plotting Baldman-Altman plot")

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
colors = {'soli':0,
          'handlogin':1,
          'tiny':2,
          'scut':3,
          'bdb':4,
          'ntu_60':5,
          'ntu_120':5}

measureVal = get_params(embeddingList,
                        datasetList,
                        'full')
df = make_df(np.array(measureVal))

if(args.mode == 'corrPlots'):
    rocket = sns.color_palette(args.corrPlot_palette)
    sns.jointplot(x=args.measure1,
                  y=args.measure2,
                  data=df,
                  kind="reg",
                  color=rocket[colors[args.dataset]])
    plt.xlabel(labels[args.measure1],fontsize=14)
    plt.ylabel(labels[args.measure2],fontsize=14)
    plt.show()

if(args.mode == 'corrQuants'):
        corrVal_spear, pVal_spear = scipy.stats.spearmanr(df[args.measure1].values[:],
                                                          df[args.measure2].values[:])
    
        corrVal_kend, pVal_kend = scipy.stats.kendalltau(df[args.measure1].values[:],
                                                         df[args.measure2].values[:])

        print('Spearman Corr: '+str(corrVal_spear))
        print('Spearman pVal: '+str(pVal_spear))
        print('Kendall Corr: '+str(corrVal_kend))
        print('Kendall pVal: '+str(pVal_kend))

        heads = ['dataset','measure1','measure2','CorrSpear','pValSpear','CorrKend','pValKend']
        entries = [args.dataset,
                   args.measure1,
                   args.measure2,
                   corrVal_spear,
                   pVal_spear,
                   corrVal_kend,
                   pVal_kend]

        if(args.initCorrFile == 1):
            corrFile = open('./_store/_corrFiles/'+args.nameCorrFile+'.txt','w')
            for idx, item in enumerate(labels):
                if(idx in [0,1,2,3,4,5]):
                    corrFile.write(str(item)+'      ')
                else:
                    corrFile.write(str(item)+'\n')

            for idx, item in enumerate(entries):
                if(idx in [0,1,2,3,4,5]):
                    corrFile.write(str(item)+'      ')
                else:
                    corrFile.write(str(item)+'\n')

        if(args.initCorrFile == 1):
            corrFile = open('./_store/_corrFiles/'+args.nameCorrFile+'.txt','a')
            for idx, item in enumerate(entries):
                if(idx in [0,1,2,3,4,5]):
                    corrFile.write(str(item)+'      ')
                else:
                    corrFile.write(str(item)+'\n')
        
if(args.mode == 'blandAltman'):
    pyCompare.blandAltman(df[args.measure1].values[:],
                          df[args.measure2].values[:],
                          savePath='./_store/graphs/_blandAltman'+args.baPath+'.png')
