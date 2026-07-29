import sys
sys.path.insert(1,'./')

from utils.hyperparameter import get_nar

embedding_list = ['./Embeddings/MS_ViViT_pt5-pt5_HandLogin.npz',
                  './Embeddings/Test/DGBQA_CGID_Res3D-MF_1-pt5_HandLogin.npz',
                  './Embeddings/MS_TPN_1pt5-1_HandLogin.npz',
                  './Embeddings/MS_TAM_1-2pt5_HandLogin.npz',
                  './Embeddings/MS_MViT_1-1pt5_HandLogin.npz']
        
y_dev_path = './Embeddings/y_dev_DGBQA_Seen_HandLogin.npz'
y_dev_id_path = './Embeddings/y_dev_id_DGBQA_Seen_HandLogin.npz'
eer_values = [0.44,1.29,4.89,1.05]

val = get_nar(embedding_list,y_dev_path,
                              y_dev_id_path,
                              eer_values,
                              4,
                              16,
                              alpha=2,
                              beta=0.75,
                              lambda_scale=2,
                              kappa=1,
                              nu=1)

print(val)