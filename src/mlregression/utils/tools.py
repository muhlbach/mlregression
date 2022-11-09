#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
# Import
import pandas as pd
import numpy as np

# User
from .exceptions import WrongInputException
#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# Misc
#------------------------------------------------------------------------------
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
