import math
import numpy as np
import scipy.special as sp
from scipy.spatial import distance

def dgbqa(embeddings,g_id,num_subjects,y_dev,y_dev_id):

    """
    DGBQA Score

    INPUTS:-
    1) embeddings: Feature Embeddings
    2) g_id: Index Value of the Gesture Class
    3) num_subjects: Number of Subjects
    4) y_dev: Ground-Truth Gesture Labelsw
    5) y_dev_id: Ground-Truth Subject Labels

    OUTPUTS:-
    1) gbqa_delta_distance: d_c_star/d_cs, GBQA Delta Distance Computed for a particular gesture

    """
    ###### Defining Essentials
    d_cs = [] # List to Maximal Intra-Subject Distance
    emb_avg_s = [] # List to Store Subject Specific Gesture Centroids

    ###### Iterating over Subjects
    for s_id in range(num_subjects): 

        ##### Defining Essentials
        embedding_store_s = [] # List to Store Embeddings from Gesture - 'g_id' and Subject - 's_id'
        dist_store_s = [] # List to Store Distance within subject 's_id' 

        ##### Intra-Subject Distance
        #### Curating Required Gesture List from g_id and s_id
        for idx in range(y_dev.shape[0]): # Iterating over Embeddings

            if(y_dev[idx] == g_id and y_dev_id[idx] == s_id):
                embedding_store_s.append(embeddings[idx]) # Storing the Required Embeddings

        #### Computing Intra-Gesture and Intra-Subject Distances
        for emb_query_idx, emb_query in enumerate(embedding_store_s):

            if(emb_query_idx != (len(embedding_store_s)-1)): # Checking if the Current Query is the Last Query

                for emb_key_idx in range(emb_query_idx+1,len(embedding_store_s),1): # Iterating over the Embeddings

                    emb_key_curr = embedding_store_s[emb_key_idx] # Extracting Current Embedding Key
                    dist_curr = distance.euclidean(emb_query,emb_key_curr) # Current Distance 
                    dist_store_s.append(dist_curr) # Appending the Computed Distance to dist_curr

        #### Computing Maximal Distance for the Current Gesture and Subject
        if(len(dist_store_s) >= 1):
            d_cs_curr = np.max(dist_store_s)
            d_cs.append(d_cs_curr) # Storing Values

        ##### Inter-Subject Distance
        #### Subject's Gesture Centroid
        if(len(embedding_store_s) >= 1):
            emb_avg_s_curr = np.average(embedding_store_s,axis=0) # Subject Specific Gesture Centroid
            emb_avg_s.append(emb_avg_s_curr)

    ###### Computing Avg. Maximal Intra-Subject Distance
    d_cs_avg = np.average(d_cs)
    
    ###### Computing Inter-Subject Distance
    ##### Defining Essentials
    dist_inter = [] # List to store Inter-Subject Distances

    ##### Computing Distances amongst the Subject Centroids
    for emb_query_idx, emb_query in enumerate(emb_avg_s):

            if(emb_query_idx != (len(emb_avg_s)-1)): # Checking if the Current Query is the Last Query

                for emb_key_idx in range(emb_query_idx+1,len(emb_avg_s),1): # Iterating over the Embeddings

                    emb_key_curr = emb_avg_s[emb_key_idx] # Extracting Current Embedding Key
                    dist_curr = distance.euclidean(emb_query,emb_key_curr) # Current Distance 
                    dist_inter.append(dist_curr) # Appending the Computed Distance to dist_curr  

    ##### Computing Average Inter-Subject Distance
    d_c_star = np.average(dist_inter)

    ###### Computing GBQA Distance Delta Score
    dgbqa_score = math.exp(d_c_star - d_cs_avg) - (1.0*(d_cs_avg/d_c_star)) # For Seen Identities
    dgbqa_score_wo = math.exp(d_c_star - d_cs_avg)

    return dgbqa_score, d_c_star, d_cs_avg, dgbqa_score_wo



def deltaDistance(embeddings,g_id,num_subjects,y_dev,y_dev_id):

    """
    Delta Distance

    INPUTS:-
    1) embeddings: Feature Embeddings
    2) g_id: Index Value of the Gesture Class
    3) num_subjects: Number of Subjects
    4) y_dev: Ground-Truth Gesture Labels
    5) y_dev_id: Ground-Truth Subject Labels

    OUTPUTS:-
    1) delta_dist: |d_cs - d_c|/d_c, Delta Distance Computed for a particular gesture

    """
    ###### Defining Essentials
    d_cs = [] # List to Maximal Intra-Subject Distance

    ###### Iterating over Subjects
    for s_id in range(num_subjects): 

        ##### Defining Essentials
        embedding_store_s = [] # List to Store Embeddings from Gesture - 'g_id' and Subject - 's_id'
        dist_store_s = [] # List to Store Distance within subject 's_id' 

        ##### Curating Required Gesture List from g_id and s_id
        for idx in range(y_dev.shape[0]): # Iterating over Embeddings

            if(y_dev[idx] == g_id and y_dev_id[idx] == s_id):
                embedding_store_s.append(embeddings[idx]) # Storing the Required Embeddings
        
        ##### Computing Intra-Gesture and Intra-Subject Distances
        for emb_query_idx, emb_query in enumerate(embedding_store_s):

            if(emb_query_idx != (len(embedding_store_s)-1)): # Checking if the Current Query is the Last Query

                for emb_key_idx in range(emb_query_idx+1,len(embedding_store_s),1): # Iterating over the Embeddings

                    emb_key_curr = embedding_store_s[emb_key_idx] # Extracting Current Embedding Key
                    dist_curr = distance.euclidean(emb_query,emb_key_curr) # Current Distance 
                    dist_store_s.append(dist_curr) # Appending the Computed Distance to dist_curr

        ##### Computing Maximal Distance for the Current Gesture and Subject
        if(len(dist_store_s) >= 1):
            d_cs_curr = np.max(dist_store_s)
            d_cs.append(d_cs_curr) # Storing Values

    ###### Computing Delta Distance
    d_c = np.max(d_cs) # Maximal Distance Amongst all the Subjects
    delta_dist = np.average(np.abs(d_cs - d_c)/d_c) # Averaging Computed Delta Distance Per Subject
    
    return delta_dist



def masterFace(d_c_star,d_size): 

    """
    MasterFace Capacity

    INPUTS:-
    1) d_c_star: Avg. Distance between different Identity Centroids within a Gesture
    2) d_size: Size of the Embeddings

    OUTPUTS:-
    1) MasterFace_Capacity: Estimated Biometric Capacity within Hand-Gestures  
    """

    MasterFace_Capacity = np.exp((d_size*(0.993 - 0.436*(1-d_c_star))+3.701-3.706*(1 - d_c_star)))
    return MasterFace_Capacity



def generativeCapacity(embeddings,
                       y_dev,
                       y_dev_id,
                       num_gestures,
                       total_ids,
                       d_size,
                       g_id):
    
    """
    Generative Capacity
    """
    
    def get_cosine_bounds(X, quantile=0.05):
        cosine_dist = np.dot(X, X.transpose())
        min_val = np.min(cosine_dist, axis=1)

        mask = np.ones(cosine_dist.shape, dtype=bool)
        np.fill_diagonal(mask, 0)
        cosine_dist = cosine_dist * mask

        max_val = np.max(cosine_dist, axis=1)
        range_val = max_val - min_val

        value = np.quantile(min_val, quantile)
        total_angle = np.arccos(value) * 180 / np.pi

        return total_angle, min_val, max_val
    
    def ratio_hyperspherical_caps(inter_class_angle, intra_class_angle, cos_delta, sin_delta, d):
        
        # compute cos(\theta) where \theta is the solid
        # angle corresponding to the inter-class hyper-spherical cap
        cos_theta = np.cos(inter_class_angle)
        sin_theta = np.sqrt(1 - cos_theta**2)
        
        cos_omega = cos_theta * cos_delta - sin_theta * sin_delta
        
        x = 1 - cos_omega * cos_omega
        a = (d - 1)/2
        b = 0.5 

        numerator = sp.betainc(a, b, x)
        index = np.where(cos_omega < 0)
        #numerator[index] = 0.5 + numerator[index]
        
        # compute cos(\phi) where \phi is the solid
        # angle corresponding to the intra-class hyper-spherical cap
        cos_phi = np.cos(intra_class_angle)
        sin_phi = np.sqrt(1 - cos_phi**2)
        cos_omega = cos_phi * cos_delta - sin_phi * sin_delta

        x = 1 - cos_omega * cos_omega
        a = (d - 1)/2
        b = 0.5

        denominator = sp.betainc(a, b, x)
        index = np.where(cos_omega < 0)
        #denominator[index] = 0.5 + denominator[index]

        capacity = numerator / denominator
        #capacity[capacity < 1] = 1
        
        return capacity
    
    g_angle = [] # List to store Intra-Gesture Angular Capacity
    g_id_angle = [] # List to store Intra-Gesture Id Angular Capacity
    Capacity_Value = [] # List to store Generative Biometric Capacity of each of the gesture

    ###### Iteration Loop
    for gesture_val in range(num_gestures): # Iterating over the Gestures

        ###### Gesture-level
        X_store = [] # List to store all the examples of that gesture
        #idx_store = [] # List to store all the indexes of the gestures being stored
        id_store = [] # List to store the identity-labels corresponding to the gesture
        g_id_angle_store = [] # List to store Angular Spreads of the 'N' identities involved in the dataset

        ##### Gesture-Store Curation
        for g_idx, X_ges in enumerate(embeddings): # Iterating over the features

            if(y_dev[g_idx] == gesture_val): # Checking for the Gesture Labels

                X_store.append(X_ges) # Storing the Feature
                id_store.append(y_dev_id[g_idx]) # Storing ID-label of the feature

        ##### Estimation of Gesture-Level Angular Spread
        g_angle_curr,_,_ = get_cosine_bounds(np.array(X_store))
        g_angle_curr = (g_angle_curr/2)*(np.pi/180)
        g_angle.append(g_angle_curr)

        ##### Estimation of Intra-Gesture Id-Level Angular Spread
        for id_idx in range(total_ids): # Searching for Particular Identities
            X_id_store = [] # List to store Intra-Gesture features of a particular identity

            for item_idx, item in enumerate(X_store): # Iterating over the Current Gesture-Store
                
                if(id_store[item_idx] == id_idx): # Identity Match-found
                    X_id_store.append(item) # Storing the Feature

            if(len(X_id_store) >= 2): # Adding condition to  check if there are enough samples per subject for that gesture
                #### Estimation of Intra-Gesture Intra-Id Angular Spread
                g_id_angle_curr,_,_ = get_cosine_bounds(np.array(X_id_store))
                g_id_angle_curr = (g_id_angle_curr/2)*(np.pi/180)    
                g_id_angle_store.append(g_id_angle_curr)

        g_id_angle_curr_overall = np.average(g_id_angle_store) # Avg. Angular Spread of all the Identities within the gesture under consideration
        g_id_angle.append(g_id_angle_curr_overall) # Storing in Global List

        ##### Estimation of Gesture's Biometric Capacity
        capacity_curr = ratio_hyperspherical_caps(g_angle_curr,g_id_angle_curr_overall,1,0,d_size) # Biometric Capacity of the Current Gesture
        Capacity_Value.append(capacity_curr) # Storing Values

    return Capacity_Value[g_id]



def swipeQuality(embeddings, 
                 y, 
                 num_gestures,
                 g_id):

    """
    Function to return quality measures per gesture   

    INPUTS:-
    1) embeddings: Input embeddings of dimensions (N,d)
    2) y: Corresponding gesture label list
    3) num_gestures: Total gestures in the dataset
    4) g_id: Index of the current gesture

    OUTPUTS:-
    1) quality_val: Quality values of the gestures of shape (num_gestures,)
    """

    ##### Global statistics
    mu_global = np.mean(embeddings,axis=0)
    sigma_global = np.var(embeddings,axis=0)

    ##### Local statisitics
    quality_val = []
    d = embeddings.shape[-1]

    for g_idx in range(num_gestures):

        curr_gest_embedds = [] # List to store embeddings of the current gesture

        for idx, emb_curr in enumerate(embeddings): # Collecting gesture of current 'g_id'
            if(y[idx] == g_idx): 
                curr_gest_embedds.append(emb_curr)

        #### Statistic estimation
        curr_gest_embedds = np.array(curr_gest_embedds)
        mu_local = np.mean(curr_gest_embedds,axis=0)
        sigma_local = np.var(curr_gest_embedds,axis=0)

        #### Quality estimation
        quality_val_curr = np.sum(np.abs(mu_global-mu_local)/(np.sqrt(sigma_global+sigma_local+1)))
        quality_val.append(quality_val_curr)
            
    return quality_val[g_id]



def distinctiveness(embeddings, y, y_id, num_gestures, num_id):

    """
    Function to return distinct measures per gesture   

    INPUTS:-
    1) embeddings: Input embeddings of dimensions (N,d)
    2) y: Corresponding gesture label list
    3) y_id: Corresponding id list
    4) num_gestures: Total gestures in the dataset
    5) num_id: Total subjects in the dataset

    OUTPUTS:-
    1) dist_val: Distinctiveness values of the gestures of shape (num_gestures,)
    """

    dist_val = []

    for g_id in range(num_gestures):

        dist_val_curr = 0 # Variable to track distinctiveness within the gesture
        g_embeddings = [] # List to store embeddings of the current gesture label
        id_list = [] # List to store ID labels        
        for idx, idx_label in enumerate(y): # Gesture embedding collection
            if(idx_label == g_id):
                g_embeddings.append(embeddings[idx])
                id_list.append(y_id[idx])

        #### Global statistics
        mu_global = np.mean(g_embeddings,axis=0)
        var_global = np.var(g_embeddings,axis=0)

        #### Local statistics
        for subject_id in range(num_id):

            sub_embeddings = [] # Current subject's and gesture's collection
            for idx, idx_label in enumerate(id_list):
                if(idx_label == subject_id):
                    sub_embeddings.append(g_embeddings[idx])

            mu_local_curr = np.mean(sub_embeddings,axis=0)
            var_local_curr = np.var(sub_embeddings,axis=0)

            dist_val_curr = dist_val_curr + np.sum(np.abs(mu_global-mu_local_curr)/(np.sqrt(var_global+var_local_curr+1)))
        dist_val_curr = dist_val_curr/(num_id) # Average computation
        dist_val.append(dist_val_curr)

    return dist_val



def getScores(embPath,
              quantifier,
              y_dev,
              y_dev_id,
              G_total,
              I_total,
              normalize=1,
              average=0):
    

    """
    Function to get scores for an embedding and a quantifier
    """

    dgbqa_score = []
    embedding = np.load(embPath,allow_pickle=True)['arr_0']

    for g_id in range(G_total):

        if(quantifier == 'dgbqa'):
            scoreCurr, _, _, _ = dgbqa(embedding,g_id,I_total,y_dev,y_dev_id)
            dgbqa_score.append(scoreCurr)

        if(quantifier == 'deltaDistance'):
            scoreCurr = deltaDistance(embedding,
                                      g_id,
                                      I_total,
                                      y_dev,
                                      y_dev_id)
            dgbqa_score.append(scoreCurr)

        if(quantifier == 'masterFace'):
            _, d_unq, _, _ = dgbqa(embedding,g_id,I_total,y_dev,y_dev_id)
            scoreCurr = masterFace(d_unq,embedding.shape[-1])
            dgbqa_score.append(scoreCurr)

        if(quantifier == 'genCapacity'):
            scoreCurr = generativeCapacity(embedding,
                                           y_dev,
                                           y_dev_id,
                                           G_total,
                                           I_total,
                                           embedding.shape[-1],
                                           g_id)
            dgbqa_score.append(scoreCurr)

        if(quantifier == 'swipeQuality'):
            scoreCurr = swipeQuality(embedding,
                                     y_dev,
                                     G_total,
                                     g_id)
            dgbqa_score.append(scoreCurr)

    dgbqa_score = np.array(dgbqa_score) # Array Formation
    dgbqa_score = (dgbqa_score - np.mean(dgbqa_score))/np.std(dgbqa_score) # Mean Normalization
    if(normalize == 1):
        dgbqa_score = dgbqa_score/np.linalg.norm(dgbqa_score) # L2-Normalization

    if(average == 1):
        dgbqa_score = np.mean(dgbqa_score,keepdims=False)

    return dgbqa_score


if __name__ == "__main__":
    x = np.load('./Embeddings/MS_MF_1-1pt5_Tiny.npz',allow_pickle=True)['arr_0']
    y_dev = np.load('./Embeddings/y_dev_DeltaDistance_SOLI.npz')['arr_0']
    y_dev_id = np.load('./Embeddings/y_dev_id_DeltaDistance_SOLI.npz')['arr_0']

    scores = getScores('./Embeddings/MS_MViT_1pt5-pt5_SOLI.npz',
                        quantifier='dgbqa',
                        y_dev=y_dev,
                        y_dev_id=y_dev_id,
                        G_total=11,
                        I_total=10,
                        normalize=1,
                        average=1)
    print(scores)
