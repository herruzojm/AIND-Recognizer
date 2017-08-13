import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    probabilities = []
    guesses = []
    
    
    for test_word_idx, test_word_features in test_set.get_all_Xlengths().items():
        scores = dict()
        best_word = None
        best_score = float("-inf")
        
        for word, model in models.items():
            try:
                score = model.score(test_word_features[0], test_word_features[1])
            except:
                score = float("-inf")
            
            scores[word] = score
            
            if score > best_score:
                best_word = word
                best_score = score
                
        probabilities.append(scores)
        guesses.append(best_word)
        
    return probabilities, guesses
    
