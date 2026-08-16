######## Importing libraries      
import math
import numpy as np
from scipy.spatial import distance

##################################################################################################################
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
### DGBQA Score
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$#
##################################################################################################################

###### Score-Estimator
def gbqa_delta_dist_compute(embeddings,g_id,num_subjects,y_dev,y_dev_id):

    """
    Function to Compute Delta Distance for a Gesture

    INPUTS:-
    1) embeddings: Feature Embeddings
    2) g_id: Index Value of the Gesture Class
    3) num_subjects: Number of Subjects
    4) y_dev: Ground-Truth Gesture Labels
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

    dgbqa_score = math.exp(d_c_star - d_cs_avg) - (1.0*(d_cs_avg/d_c_star)) # For Seen Identities
    dgbqa_score_wo = math.exp(d_c_star - d_cs_avg)

    return dgbqa_score, d_c_star, d_cs_avg, dgbqa_score_wo
