###### Importinng libraries
import argparse
import numpy as np
from ModelSelector import get_val, select_model

###### Selecting model embeddings

parser = argparse.ArgumentParser()
parser.add_argument("--metric",
                    type=str,
                    help="The metric to be used")
parser.add_argument('--quantifier',
                    type=str,
                    default='dgbqa',
                    help="The quantifier to be used for score generation")

args = parser.parse_args()

##### Defining essentials
embedding_list_soli = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-pt5_SOLI.npz',
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
dataset_list_soli = ['Soli']*39

embedding_list_hl = ['./Embeddings/MS_ViViT_pt5-pt5_HandLogin.npz',
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
dataset_list_hl = ['HandLogin']*55

embedding_list_tiny = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz',
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
dataset_list_tiny = ['Tiny']*12


dataset_curr = 'Soli'

if(dataset_curr == 'Soli'):
    y_dev = np.load('./Embeddings/y_dev_DeltaDistance_SOLI.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DeltaDistance_SOLI.npz')['arr_0']
    G_total = 11
    I_total = 10
    eer_values = [15.60,14.33,8.98,14.33,4.83,4.74,7.13,7.60,8.15,5.94,18.63]

if(dataset_curr == 'HandLogin'):
    y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_HandLogin.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_HandLogin.npz')['arr_0']
    G_total = 4
    I_total = 16
    eer_values = [0.44,1.29,4.89,1.05]

if(dataset_curr == 'Tiny'):
    y_dev = np.load('./Embeddings/y_dev_DGBQA_Seen_Tiny.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DGBQA_Seen_Tiny.npz')['arr_0']
    G_total = 11
    I_total = 26

    e1_val = 100 - 16.45
    e2_val = 100 - 23.36 
    e1 = np.array([16.38,22.19,21.60,11.61,9.24,8.95,14.58,14.45,17.30,9.25,35.47])
    e2 = np.array([21.12,26.42,32.30,20.34,18.18,17.33,19.81,24.45,25.70,11.52,39.81])
    eer_values = (e1_val*e1+e2_val*e2)/(e1_val+e2_val)
    eer_values = list(eer_values)

##### Model selection and prediction
model = select_model(embedding_list_soli,
                     dataset_list_soli,
                     args.metric,
                     quantifier=args.qauntifier)
print(model)

embedding = np.load(model)['arr_0']

val = get_val(embedding,
              y_dev,
              y_dev_id,
              eer_values,
              G_total,
              I_total,
              None,
              'full',
              quantifier=args.qauntifier)
print('nAr*: '+str(val[8]))
val = val[:4]

print('Rank deviation: '+str(val[0]))
print('Relevance: '+str(val[1]))
print('Trend deviation: '+str(val[2]))
print('Entanglement: '+str(val[3]))