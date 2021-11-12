#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pickle
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Misc
#------------------------------------------------------------------------------
def break_links(x):
    """ Break link to x by copying it"""
    x = x.copy()
    return x

#------------------------------------------------------------------------------
# Save and load models
#------------------------------------------------------------------------------
# Save and load objects using pickle
def save_object_by_pickle(path,obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object_by_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)



class SingleSplit(object):
    """
    Split dataset in two fold (single split), similar to train-test split
    """
    # --------------------
    # Constructor function
    # --------------------
    def __init__(self,
                 test_size=0.25):
        self.test_size=test_size

    # --------------------
    # CLASS VARIABLES
    # --------------------        
    N_SPLITS = 1
        
    # --------------------
    # Public
    # --------------------        
    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where `n_samples` is the number of samples
            and `n_features` is the number of features.
        y : array-like of shape (n_samples,), default=None
            The target variable for supervised learning problems.
        groups : array-like of shape (n_samples,), default=None
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)    
        indices = np.arange(n_samples)

        n_test = int(self.test_size*n_samples)
        n_train = n_samples-n_test
        
        train = indices[0:n_train]
        test  = indices[n_train:n_samples]
        
        yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.N_SPLITS
