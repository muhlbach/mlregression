#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Standard
import pandas as pd
import numpy as np
import pickle
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples
import numbers

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

# NEEDS TO BE CHECKED !!!    
# def isin(pattern, x, how="all", return_element_wise=True):
#     """Check if any of the elements in 'pattern' is included in 'x'"""
#     ALLOWED_HOW = ["all", "any"]
    
#     # List iterable classes. Note that we cannot rely on __iter__ attribute, because then str will be iterable!
#     ITERABLES = (list, tuple, np.ndarray, pd.Series)
    
#     if how not in ALLOWED_HOW:
#         raise WrongInputException(input_name="how",
#                                   provided_input=how,
#                                   allowed_inputs=ALLOWED_HOW)
        
#     if not isinstance(x, ITERABLES):
#         if how=="all":
#             is_in = all(p == x for p in pattern)
#         elif how=="any":
#             is_in = any(p == x for p in pattern)
#     else:
#         is_in = [isin(pattern, x=y, how=how) for y in x]
        
#         if not return_element_wise:
#             is_in = any(is_in)
                
#     return is_in

def convert_to_list(x):
    """Convert object "x" to list"""
    
    # List non-iterables and iterables. Note that we cannot rely on __iter__ attribute, because then str will be iterable!
    NON_ITERABLES = (str,numbers.Number)
    ITERABLES = (list, tuple, np.ndarray, pd.Series)
    
    if isinstance(x,NON_ITERABLES):
        x = [x]
    elif isinstance(x,ITERABLES):
        x = [i for i in x]
    else:
        try:
            x = [i for i in x]
        except:
            x = [x]
            
    return x

def isin(a, b, how="all", return_element_wise=True):
    """Check if any/all of the elements in 'a' is included in 'b'
    Note: Argument 'how' has NO EFFECT when 'return_element_wise=True'
    
    """
    ALLOWED_HOW = ["all", "any"]
    
    if how not in ALLOWED_HOW:
        raise WrongInputException(input_name="how",
                                  provided_input=how,
                                  allowed_inputs=ALLOWED_HOW)

    # Convert "a" and "b" to lists
    a = convert_to_list(x=a)
    b = convert_to_list(x=b)

    # For each element (x) in a, check if it equals any element (y) in b
    is_in_temp = [any(x == y for y in b) for x in a]

    if return_element_wise:
        is_in = is_in_temp
    else:
        # Evaluate if "all" or "any" in found, when we only return one (!) answer
        if how=="all":
            is_in = all(is_in_temp)
        elif how=="any":
            is_in = any(is_in_temp)
                    
    if (len(a)==1) and isinstance(is_in, list):
        # Grab first and only argument if "a" is not iterable
        is_in = is_in[0]
            
    return is_in

def remove_conditionally_invalid_keys(d, invalid_values=["deprecated"]):
    """ Remove keys from dictionary for which values contain specified values"""    
    d = {k:v for k,v in d.items() if not isin(a=invalid_values,b=v,how="any",return_element_wise=False)}
        
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
