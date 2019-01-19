
import re
from tqdm import tqdm
import numpy as np

def find_part_pii(text, model, sep = " "):
    tokenized_text = text.split(sep)
    
    part_pii = model.wv.doesnt_match(tokenized_text)
    
    return part_pii    



def get_word2vec_matrix(pii_list, model):
    # set the matrix dimensions
    column_num = model.trainables.layer1_size
    row_num = len(pii_list)
    # initialized the matrix
    pii2vec_mat = np.zeros((row_num, column_num))
    # iterate through the pii_list and assign the vectors to matrix.
    for index, ith_pii in enumerate(tqdm(pii_list)):
        pii2vec_mat[index,:] = model.wv[ith_pii]
    
    return pii2vec_mat


def get_doc2vec_matrix(texts, model):
    """
    A helper function to get the vector for each document
    and combine it into a matrix. 
    """
    num_columns = model.trainables.layer1_size
    num_rows = len(texts)
    
    doc2vec_pii_matrix = np.zeros((num_rows, num_columns))
    
    for index, text in enumerate(tqdm(texts)):
        
        doc2vec_pii_matrix[index,:] = model.infer_vector(text.split(" "))
        
    return doc2vec_pii_matrix


color_dict = {"Phone_number":"red","SSN":"blue","Address":"black","Name":"yellow",\
             "Plates":"orange","CreditCardNumber":"purple","None":'pink',"Email":"tan"}


def _match_re_and_extract_cv_scores(model,all_cv_keys, regex):
    match_keys = re.findall(regex," ".join(all_cv_keys))
    
    cv_scores = dict()
    
    for key in match_keys:
        cv_scores[key] = model.cv_results_[key]
    return cv_scores 


def extract_sk_cv(model, test = True):
    """
    A helper function to extract the training/testing cross validation result
    """
    all_cv_keys = model.cv_results_.keys()
    
    if test:
        sk_cv_result =  _match_re_and_extract_cv_scores(model = model,all_cv_keys = all_cv_keys\
                                                             ,regex = r"split[0-9]+_test_score")
        return pd.DataFrame(sk_cv_result)

    elif not test:
        
        sk_cv_result = _match_re_and_extract_cv_scores(model = model,all_cv_keys = all_cv_keys\
                                                             ,regex = r"split[0-9]+_train_score")
        return pd.DataFrame(sk_cv_result)
