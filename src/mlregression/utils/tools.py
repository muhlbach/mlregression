#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import numpy as np
import pickle
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

# User
from .exceptions import WrongInputException
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

def isin(pattern, x, how="all", return_element_wise=True):
    """Check if any of the elements in 'pattern' is included in 'x'"""
    ALLOWED_HOW = ["all", "any"]
    ITERABLES = (list, tuple)
    
    if how not in ALLOWED_HOW:
        raise WrongInputException(input_name="how",
                                  provided_input=how,
                                  allowed_inputs=ALLOWED_HOW)

    if not isinstance(x, ITERABLES):
        if how=="all":
            is_in = all(p == x for p in pattern)
        elif how=="any":
            is_in = any(p == x for p in pattern)

    else:
        is_in = [isin(pattern, x=y, how=how) for y in x]
        
        if not return_element_wise:
            is_in = any(is_in)
                
    return is_in

def remove_conditionally_invalid_keys(d, invalid_values=["deprecated"]):
    """ Remove keys from dictionary for which values contain specified values"""
    d = {k:v for k,v in d.items() if not isin(pattern=invalid_values,x=v,how="any",return_element_wise=False)}
    
    return d

def remove_invalid_attr(obj, invalid_attr):
    ALLOWED_INVALID_ATTR = [dict, list]
    
    if isinstance(invalid_attr, dict):
        for k,v in invalid_attr.items():
            if hasattr(obj, k):
                delattr(obj, k)
    elif isinstance(invalid_attr, list):
        for k in invalid_attr:
            if hasattr(obj, k):
                delattr(obj, k)
    else:
        raise WrongInputException(input_name="invalid_attr",
                                  provided_input=invalid_attr,
                                  allowed_inputs=ALLOWED_INVALID_ATTR)        

    return obj

def unlist_dict_values(d):
    
    # Turn all elements into list
    d = {key:(value if isinstance(value, list) else [value]) for key,value in d.items()}
    
    # Unlist if possible
    d = {key:(value if len(value)>1 else value[0]) for key,value in d.items()}
    
    return d
    
def get_unique_elements_from_list(l, keep_order=True):
    
    if keep_order:    
        seen = set()
        l_unique = [x for x in l if not (x in seen or seen.add(x))]
    else:
        l_unique = list(set(l))
        
    return l_unique



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
