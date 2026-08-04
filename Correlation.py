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
parser.add_argument("--bdbMode",
                    type=str,
                    default='Acc',
                    help="bdb sensor to be used.")
parser.add_argument('--mode',
                    type=str,
                    default='full',
                    help="Mode of correlation analysis")
parser.add_argument('--quantifier',
                    type=str,
                    default='dgbqa',
                    help="The quantifier to be used for score generation")
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
parser.add_argument('--corrPath',
                    type=str,
                    help="Name of the experiment for plotting Correlation plot")
parser.add_argument('--baPath',
                    type=str,
                    help="Name of the experiment for plotting Bland-Altman plot")

args = parser.parse_args()

embeddingList, datasetList = retList(args.dataset,args.bdbMode)
labels = {'r':'$\\mathcal{r}$',
           'relevance':'R',
            'psi':'$\\psi$',
            'Cd':'$C_{d}$',
            'Ar_star':'$nAr^{*}(\Delta)$',
            'euclid':'Euclidean dist.',
            'corr':'Correlation',
            'Kendall':'$\\tau$',
            'DCG':'DCG',
            'err':'err',
            'U':'U measure',
            'gre':'GRE',
            'infAp':'infAp',
            'neg_rel':'Negative Relevance',
            'rpp':'RPP'}
colors = {'soli':0,
          'handlogin':1,
          'tiny':2,
          'scut':3,
          'bdb':4,
          'ntu_60':5,
          'ntu_120':5}

if(args.dataset in ['bdb','ntu_60','ntu_120'] and args.quantifier == 'masterFace'):
    normalize = 0
else:
    normalize = 1

measureVal = get_params(embeddingList,
                        datasetList,
                        'full',
                        quantifier=args.quantifier,
                        normalize=normalize)
df = make_df(np.array(measureVal))

if(args.mode == 'corrPlots'):
    cp = sns.color_palette(args.corrPlot_palette)
    sns.jointplot(x=args.measure1,
                  y=args.measure2,
                  data=df,
                  kind="reg",
                  color=cp[colors[args.dataset]])
    plt.xlabel(labels[args.measure1],fontsize=14)
    plt.ylabel(labels[args.measure2],fontsize=14)
    plt.savefig('./_store/_graphs/_corrPlots/'+args.corrPath+'.png')
    plt.close()


if(args.mode == 'corrQuants'):
        corrVal_spear, pVal_spear = scipy.stats.spearmanr(df[args.measure1].values[:],
                                                          df[args.measure2].values[:])
    
        corrVal_kend, pVal_kend = scipy.stats.kendalltau(df[args.measure1].values[:],
                                                         df[args.measure2].values[:])

        print('Spearman Corr: '+str(corrVal_spear))
        print('Spearman pVal: '+str(pVal_spear))
        print('Kendall Corr: '+str(corrVal_kend))
        print('Kendall pVal: '+str(pVal_kend))

        heads = ['dataset','quants','measure1','measure2','CorrSpear','pValSpear','CorrKend','pValKend']
        entries = [args.dataset,
                   args.quantifier,
                   args.measure1,
                   args.measure2,
                   np.round(corrVal_spear,4),
                   np.round(pVal_spear,4),
                   np.round(corrVal_kend,4),
                   np.round(pVal_kend,4)]

        if(args.initCorrFile == 1):
            corrFile = open('./_store/_corrFiles/'+args.nameCorrFile+'.txt','w')
            for idx, item in enumerate(heads):
                if(idx in [0,1,2,3,4,5,6]):
                    corrFile.write(str(item)+'      ')
                else:
                    corrFile.write(str(item)+'\n')

            for idx, item in enumerate(entries):
                if(idx in [0,1,2,3,4,5,6]):
                    corrFile.write(str(item)+'      ')
                else:
                    corrFile.write(str(item)+'\n')

        if(args.initCorrFile == 0):
            corrFile = open('./_store/_corrFiles/'+args.nameCorrFile+'.txt','a')
            for idx, item in enumerate(entries):
                if(idx in [0,1,2,3,4,5,6]):
                    corrFile.write(str(item)+'      ')
                else:
                    corrFile.write(str(item)+'\n')

        
if(args.mode == 'blandAltman'):
    pyCompare.blandAltman(df[args.measure1].values[:],
                          df[args.measure2].values[:],
                         savePath='./_store/_graphs/_blandAltman/'+args.baPath+'.png')


if(args.mode == 'full'):
    # Correlation plots
    cp = sns.color_palette(args.corrPlot_palette)
    sns.jointplot(x=args.measure1,
                    y=args.measure2,
                    data=df,
                    kind="reg",
                    color=cp[colors[args.dataset]])
    plt.xlabel(labels[args.measure1],fontsize=14)
    plt.ylabel(labels[args.measure2],fontsize=14)
    plt.savefig('./_store/_graphs/_corrPlots/'+args.corrPath+'.png')
    plt.close()

    # Correlation values: statistical analysis
    corrVal_spear, pVal_spear = scipy.stats.spearmanr(df[args.measure1].values[:],
                                                      df[args.measure2].values[:])
    
    corrVal_kend, pVal_kend = scipy.stats.kendalltau(df[args.measure1].values[:],
                                                     df[args.measure2].values[:])
    print('Spearman Corr: '+str(corrVal_spear))
    print('Spearman pVal: '+str(pVal_spear))
    print('Kendall Corr: '+str(corrVal_kend))
    print('Kendall pVal: '+str(pVal_kend))

    heads = ['dataset','quants','measure1','measure2','CorrSpear','pValSpear','CorrKend','pValKend']
    entries = [args.dataset,
                args.quantifier,
                args.measure1,
                args.measure2,
                np.round(corrVal_spear,4),
                np.round(pVal_spear,4),
                np.round(corrVal_kend,4),
                np.round(pVal_kend,4)]

    if(args.initCorrFile == 1):
        corrFile = open('./_store/_corrFiles/'+args.nameCorrFile+'.txt','w')
        for idx, item in enumerate(heads):
            if(idx in [0,1,2,3,4,5,6]):
                corrFile.write(str(item)+'      ')
            else:
                corrFile.write(str(item)+'\n')

        for idx, item in enumerate(entries):
            if(idx in [0,1,2,3,4,5,6]):
                corrFile.write(str(item)+'      ')
            else:
                corrFile.write(str(item)+'\n')

    if(args.initCorrFile == 0):
        corrFile = open('./_store/_corrFiles/'+args.nameCorrFile+'.txt','a')
        for idx, item in enumerate(entries):
            if(idx in [0,1,2,3,4,5,6]):
                corrFile.write(str(item)+'      ')
            else:
                corrFile.write(str(item)+'\n')

    # Bland-Altman plots
    pyCompare.blandAltman(df[args.measure1].values[:],
                              df[args.measure2].values[:],
                             savePath='./_store/_graphs/_blandAltman/'+args.baPath+'.png')