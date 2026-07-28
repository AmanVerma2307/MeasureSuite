import torch
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from loss.icgd import *
from model import *
from epochs.eval import eval
from utils.parser import parse
from data import dataLoader, dataloader_nonShuffled
from sklearn.manifold import TSNE

args = parse()

if(args.reproducibility == 1):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

if(args.dataset == 'soli'):
    input_dim = 32
    patch_size = (5,5,5)
    T = 40
    H = 32
    W = 32
    C = 4
    d_model = args.d_model
    dff = args.dff
    num_heads = args.res3dvivit_heads
    num_encoders = args.num_encoders
    rate = 0.1
    G = 11
    I = 10

    cm_plot_labels = ['Pinch index','Palm tilt','Finger Slider','Pinch pinky','Slow Swipe','Fast Swipe','Push','Pull','Finger rub','Circle','Palm hold']
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf","yellow"]

    y_dev = np.load('./_data/soli/data/y_dev_DGBQA-Seen_SOLI.npz',allow_pickle=True)['arr_0']
    y_dev_id = np.load('./_data/soli/data/y_dev_DGBQA-Seen_SOLI.npz',allow_pickle=True)['arr_0']

if(args.dataset in ['ntu_60', 'ntu_120']):
    T = args.ntu_numFrames
    H = None
    W = None
    C = None

    if(args.dataset == 'ntu_60'):
        G = 6
        I = 40
        cm_plot_labels = ['Jump up','Throw','Nod head','Make a phone call','Check time','Use a fan']
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd","#8c564b"]
        
    if(args.dataset == 'ntu_120'):
        G = 4
        I = 69
        cm_plot_labels = ['Running on the spot','Arm swings','Side kick','Thumbs up']
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    y_dev = np.load('./_data/ntu_rgb/dataProcessed/y_dev_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']
    y_dev_id = np.load('./_data/ntu_rgb/dataProcessed/y_dev_id_non-idf_T'+str(args.ntu_numFrames)+'_'+str(args.dataset)+'.npz')['arr_0']

    
train_dataLoader, test_dataLoader = dataLoader(args)
test_dataLoader_ns = dataloader_nonShuffled(args)

model = quantModel(args,
                   T=T,
                   H=H,
                   W=W,
                   C=C,
                   G=G,
                   I=I)
model.load_state_dict(torch.load('./_store/weights/'+args.exp_name+'.pth', weights_only=True)) 
model.eval()

if(args.multi_gpu == False):
    device = torch.device(args.device)
    model = model.to(device)

g_hgr, g_id, f_theta = eval(test_dataLoader,
                            model,
                            device,
                            args,
                            writer=True)
f_theta = torch.nn.functional.normalize(torch.from_numpy(f_theta),dim=-1).numpy()

_, _, f_theta_ns = eval(test_dataLoader_ns,
                        model,
                        device,
                        args,
                        writer=False) # Non-shuffled 
f_theta_ns = torch.nn.functional.normalize(torch.from_numpy(f_theta_ns),dim=-1).numpy()

G_bar = np.matmul(f_theta,f_theta.T) # Gram-Matrix
G_bar_ns = np.matmul(f_theta_ns,f_theta_ns.T) # Gram-matrix non_shuffled

##### Plotting Heatmap
plt.rcParams["figure.figsize"] = [8,12]
def plot_heatmap(cm,filepath,classes,normalize=False,title='Avg. HGR Probabilities',cmap=plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    #plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

def plot_GramMatrix(cm,filepath,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

#### Heatmap Plotting
filepath_gram ='./_store/graphs/gramMatrix/'+args.exp_name+'.png'
filepath_gram_ns ='./_store/graphs/gramMatrix/'+args.exp_name+'_NonShuffled.png'
plot_GramMatrix(cm=G_bar,filepath=filepath_gram)
plot_GramMatrix(cm=G_bar_ns,filepath=filepath_gram_ns)

###### tSNE
col_mean = np.nanmean(f_theta, axis=0)
inds = np.where(np.isnan(f_theta))
f_theta[inds] = np.take(col_mean, inds[1])
np.savez_compressed('./_store/embeddings/'+args.exp_name+'.npz',f_theta)

tsne_X_dev = TSNE(n_components=2,
                  perplexity=30,
                  learning_rate=10,
                  n_iter=10000,
                  n_iter_without_progress=50).fit_transform(f_theta) # t-SNE Plots 

plt.rcParams["figure.figsize"] = [12,8]
for idx,color_index in zip(list(np.arange(G)),colors):
    plt.scatter(tsne_X_dev[y_dev == idx, 0],tsne_X_dev[y_dev == idx, 1],s=55,color=color_index,edgecolors='k',marker='h')
plt.legend(cm_plot_labels,loc='best',prop={'size': 12})
plt.savefig('./_store/graphs/tsne/'+args.exp_name+'.png')
