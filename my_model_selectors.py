import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose
        self.num_features = len(self.X[0])
        self.num_datapoints = len(self.X)
        if self.verbose:
            print("number of features = ", self.num_features)
            print("number of data points = ", self.num_datapoints)

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states, X = None, lengths = None):
        if X is None:
            X = self.X
        if lengths is None:
            lengths = self.lengths

        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(X, lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None

class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant
    """

    def select(self):
        """ select based on n_constant value
        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Baysian Information Criterion(BIC) score
    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components
        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_hmm_model = None
        best_hmm_model_score = float("inf")

        for states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(states)                
                model_score = hmm_model.score(self.X, self.lengths)
                
                parameters = (states * states) + (2 * states * self.num_features)
                score = -2 * model_score + parameters * np.log(self.num_datapoints)
                
                if score < best_hmm_model_score: 
                    best_hmm_model = hmm_model
                    best_hmm_model_score = score
                    
            except:
                if self.verbose:
                    print("Upps! word {} states {}".format(self.this_word, states))
                    
        return best_hmm_model


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion
    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        best_hmm_model = None
        best_hmm_model_score = float("-inf")

        for states in range(self.min_n_components, self.max_n_components + 1):
            try:
                hmm_model = self.base_model(states)
                model_score = hmm_model.score(self.X, self.lengths)
                
                sum_other_score = 0.0
                other_words = (word for word in self.hwords.keys() if word != self.this_word)
                other_word_count = 0
                
                for word in other_words:
                    other_X, other_lengths = self.hwords[word]
                    try:
                        other_word_score = hmm_model.score(other_X, other_lengths)                        
                        other_word_count += 1
                        sum_other_score += other_word_score
                        
                    except:
                        sum_other_score += 0.0                        
                                                        
                if other_word_count != 0:
                    score = model_score - sum_other_score / other_word_count                    
                    if score > best_hmm_model_score: 
                        best_hmm_model = hmm_model
                        best_hmm_model_score = score
                        
            except:
                if self.verbose:
                    print("Upps! word {} states {}".format(self.this_word, states))
                    
        return best_hmm_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_hmm_model = None
        best_hmm_model_socre = float("-inf")
        
        splits = len(self.sequences)
        if splits > 3:
            splits = 3
        elif splits < 2:
            best_hmm_model = self.base_model(self.n_constant)
            return best_hmm_model
        
        kf = KFold(splits)
        for states in range(self.min_n_components, self.max_n_components + 1):
            for train_index, test_index in kf.split(self.sequences):
                try:
                    X_train, lengths_train = combine_sequences(train_index, self.sequences)
                    X_test, lengths_test = combine_sequences(test_index, self.sequences)
                    
                    hmm_model = self.base_model(states, X_train, lengths_train)                    
                    score = hmm_model.score(X_test, lengths_test)
                    
                    if score > best_hmm_model_socre:
                        best_hmm_model = hmm_model
                        best_hmm_model_socre = score
                        
                except:
                    if self.verbose:
                        print("Upps! word {} states {}".format(self.this_word, states))
                        
        return best_hmm_model