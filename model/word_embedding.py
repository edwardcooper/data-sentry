import re
def clean_text(text):
    # replace  . and a space with only a space, then amke all words lower case.
    text = text.replace(". "," ").replace(",","").lower()
    # get rid of the . at the end of each line. 
    cleaned_text = re.sub("\.$","",text)
    
    return cleaned_text
 


class text_clean:
    """
    A class to help with cleaning text data. 
    """
    def fit(self, X, y):
        return self
    def transform(self, X):
        assert isinstance(X,pd.Series), "The input data should be pandas Series."
        X = X.apply(clean_text)
        
        return X



from tqdm import tqdm
import numpy as np
from gensim.models import Word2Vec
from gensim.models import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.externals import joblib



def _find_part_pii(text, model, sep = " "):
    tokenized_text = text.split(sep)
    
    part_pii = model.wv.doesnt_match(tokenized_text)
    
    return part_pii    



def _extracted_pii2matrix(pii_list, model):
    # set the matrix dimensions
    column_num = model.trainables.layer1_size
    row_num = len(pii_list)
    # initialized the matrix
    pii2vec_mat = np.zeros((row_num, column_num))
    # iterate through the pii_list and assign the vectors to matrix.
    for index, ith_pii in enumerate(tqdm(pii_list)):
        pii2vec_mat[index,:] = model.wv[ith_pii]
    
    return pii2vec_mat



class word_embedding:
    """
    A class to convert words/docs to vectors by applying any model supported by gensim.  
    
    This class will allow continued training on the pre-trained model by assigning
    the model to the pre_trained option in class initialization.  
    
    After training the model, it will dump the word2vec model to the path assigned to 
    dump_file option.  
    
    
    """
    def __init__(self, algo_name = "word2vec", size = 100, min_count = 1, window = 5, workers =1,\
                 epochs = 5, pre_train = None, dump_file = False,\
                 re_train_new_sentences = True):
        
        
        assert algo_name in ["word2vec", 'fasttext', 'doc2vec'], \
        "please enter a model name in ['word2vec', 'fasttext', 'doc2vec']"
        
        self.algo_name = algo_name
        self.epochs = epochs 
        self.pre_train = pre_train
        self.dump_file = dump_file 
        self.re_train_new_sentences = re_train_new_sentences
        
        # model options
        self.size = size
        self.min_count = min_count
        self.window = window
        self.workers = workers
        
        
    def _algo_init(self):
        if self.algo_name == "word2vec":
            model = Word2Vec(size = self.size, min_count = self.min_count,
                            window = self.window, workers = self.workers)
        elif self.algo_name == "fasttext":
            model = FastText(size = self.size, min_count = self.min_count,
                            window = self.window, workers = self.workers)
        elif self.algo_name == "doc2vec":
            model = Doc2Vec(vector_size = self.size, min_count = self.min_count,
                            window = self.window, workers = self.workers)
            
        self.model = model
        return self

    def _embedding_training(self, sentences, update = False):
        updated_model_with_vocab = self.model

        updated_model_with_vocab.build_vocab(sentences, update = update)
        
        updated_model_with_vocab.train(sentences, total_examples = len(sentences), epochs = self.epochs)
        
        # update the model with the trained one. 
        self.model = updated_model_with_vocab
        
    def _pd_to_gensim_format(self, text, tags):
        
        # special handling for doc2vec model. 
        if self.algo_name == "doc2vec":
            if tags is None:
                documents = [TaggedDocument(sentence.split(" "), [index])\
                      for index in range(len(text))] 
                print("Using index for the tags")
            else:
                assert len(text) == len(tags), "The y and X input must be of same length."
                documents = [TaggedDocument(paragraph.iloc[index].split(" "), [tags.iloc[index]])\
                      for index in range(len(text))]
                print("Using true tags")

        # for word2vec and fasttext     
        else:
            documents = [sentence.split(" ") for sentence in text]
            
            
        return documents
            
        
    def fit(self, X, y=None):
        """
        The fit method will get use the pre_trained model if the model is assigned to the pre_train option.
        
        If the pre_train is None, then the model will be trained. 
        """
        gensim_X = self._pd_to_gensim_format(text = X, tags = y)
        
        if self.pre_train is not None:
            self.model = self.pre_train
            return self
        else:
            # initialize the model, split the sentence into tokens and train it. 
            self._algo_init()
            self._embedding_training(sentences = gensim_X)
            
        return self
        
    
    def transform(self, X):
        """
        If re_train_new_sentences is True, which is the default setting, 
        the model will be re-trained on the new sentences. 
        This will create word embedding for words not in the original vocabulary.
        This will increase the model inference time since it invovles model training. 
        
        For using word2vec to predict PII data, it is recommended to update the model with new sentences. 
        For fastttext, it is not necessary since it will infer from the character n-grams. The fasttext training
        is much longer than word2vec. 
        """
        # update the embedding with new sentences or train the model. 
        if self.re_train_new_sentences:
            self._embedding_training(X, update = True)
            print("transforming while training {} model with new data.".format(self.algo_name))
        if self.algo_name == 'doc2vec':
            X = [" ".join(words[0]) for words in X ]
        # extract the PII 
        extracted_pii_list = [_find_part_pii(text, model = self.model)\
                    for text in tqdm(X) ]
        
        # convert the extracted pii text into vectors.
        piivec_matrix = _extracted_pii2matrix(pii_list = extracted_pii_list,\
                                          model = self.model)
        return piivec_matrix 
                                          