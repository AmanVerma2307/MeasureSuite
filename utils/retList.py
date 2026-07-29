def retList(dataset,
            bdbMode=None):

    """
    Function to return list of embedding file paths and dataset names

    INPUTS:-
    1) dataset: Name of the dataset
    2) bdbMode: Sensor mode for behavePassDB dataset

    OUTPUTS:-
    1) embedding_list: List containing embedding file paths
    2) dataset_list: List containing name of the datasets
    """

    if(dataset == 'soli'):
        embedding_list = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-pt5_SOLI.npz',
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
        dataset_list = ['Soli']*39

    if(dataset == 'handlogin'):
        embedding_list = ['./Embeddings/MS_ViViT_pt5-pt5_HandLogin.npz',
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
        dataset_list = ['HandLogin']*55

    if(dataset == 'tiny'):
        embedding_list = ['./Embeddings/DGBQA_CGID_Res3D-ViViT_pt5-1_Tiny.npz',
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
        dataset_list = ['Tiny']*12

    return embedding_list, dataset_list

