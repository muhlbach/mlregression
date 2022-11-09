#------------------------------------------------------------------------------
# Libraries
#------------------------------------------------------------------------------
import pandas as pd
import re
import numpy as np
import bodyguard as bg
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

#------------------------------------------------------------------------------
# Main
#------------------------------------------------------------------------------
class Preprocessor(object):
    
    # Constructor
    def __init__(self,
                 scale_vars,
                 with_mean=True,
                 with_std=True,
                 only_categorical_dtype_to_dummy=False,
                 drop_first_category=True,
                 handle_unknown="drop",
                 ):
        """
        Preprocess design matrix before ML pipeline
        """
        self.scale_vars = scale_vars
        self.with_mean = with_mean
        self.with_std = with_std
        self.only_categorical_dtype_to_dummy = only_categorical_dtype_to_dummy
        self.drop_first_category = drop_first_category
        self.handle_unknown = handle_unknown
        
        # Instantiate
        if scale_vars:
            self.scaler = StandardScaler(with_mean=self.with_mean,
                                         with_std=self.with_std)

        # Pre-allocate
        self.is_fitted_ = False

    # Private functions
    def _split_cols_into_cat_and_num(self, X, only_categorical_dtype_to_dummy=False):
        
        # Find num and non-num cols
        num_cols = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
        cat_cols = [col for col in X.columns if col not in num_cols]
        
        if only_categorical_dtype_to_dummy:
            # Subset to only categorical dtypes
            cat_cols = [col for col in cat_cols if pd.api.types.is_categorical_dtype(X[col])]
            
        return cat_cols,num_cols
    
    # Public functions
    def fit(self,X):
        
        # Convert to dataframe
        X = bg.convert.convert_to_df(x=X)
        
        # Separate cols
        self.cat_cols, self.num_cols = self._split_cols_into_cat_and_num(X=X, only_categorical_dtype_to_dummy=self.only_categorical_dtype_to_dummy)
        
        # One-hot encoding
        X = pd.get_dummies(data=X,
                           prefix=None,
                           prefix_sep='_',
                           dummy_na=False,
                           columns=self.cat_cols,
                           sparse=False,
                           drop_first=self.drop_first_category,
                           dtype=None)
        
        # Get final col names
        self.cols = X.columns
        
        if self.scale_vars:
            # Scale variables
            self.scalar.fit(X=X)
            
        # Update status
        self.is_fitted_ = True
        
        return self
        
    def transform(self,X):
        
        # Check status
        if not self.is_fitted_:
            raise Exception("Object if not fitted. Thus, call '.fit()' before proceeding")
            
        # Convert to dataframe
        X = bg.convert.convert_to_df(x=X)

        # Check that all columns are present
        if not all(col in X.columns for col in [self.cat_cols+self.num_cols]):
            raise Exception("Some cols are missing in X. Please make sure all below columns are present: \n{self.cols}")

        # One-hot encoding
        X = pd.get_dummies(data=X,
                           prefix=None,
                           prefix_sep='_',
                           dummy_na=False,
                           columns=self.cat_cols,
                           sparse=False,
                           drop_first=self.drop_first_category,
                           dtype=None)
        
        # Check categorical categories
        missing_categories = [col for col in self.cols if col not in X.columns]
        
        if bool(missing_categories):
            print(f"Categories {missing_categories} are not present in X, but do not worry, we all them as zeros.")
            
        for cat in missing_categories:
            X[cat] = 0
            
        # Subset to original cols
        X = X[self.cols]
        
        if self.scale_vars:
            X.iloc[:,:] = self.scaler.transform(X=X)
            
        return X
        
    def fit_transform(self,X):
        
        # Fit
        self.fit(X=X)
        
        # Transform
        X = self.transform(X=X)
        
        return X
    

class PolynomialExpander(object):
    # Constructor
    def __init__(self,
                 scale_vars,
                 with_mean=True,
                 with_std=True,
                 degree=2,
                 interaction_term_only=False,
                 self_term_only=False,
                 include_bias=False,
                 ):
        """
        Preprocess design matrix before ML pipeline
        """
        self.scale_vars = scale_vars
        self.with_mean = with_mean
        self.with_std = with_std
        self.degree = degree
        self.interaction_term_only = interaction_term_only
        self.self_term_only = self_term_only
        self.include_bias = include_bias

        # Instantiate
        if scale_vars:
            self.scaler = StandardScaler(with_mean=self.with_mean,
                                         with_std=self.with_std)

        self.expander = PolynomialFeatures(degree=self.degree,
                                           interaction_only=self.interaction_term_only,
                                           include_bias=self.include_bias)

        # Pre-allocate
        self.expander_is_fitted_ = False
        self.is_fitted_ = False

    # Private functions
    def _transform_without_scaling(self,X):
        # Check status
        if not self.expander_is_fitted_:
            raise Exception("Expander Object if not fitted. Thus, call '.fit()' before proceeding")
                        
        # Extract dummies
        X_dummy = X[self.cols_binary]
        
        # Transform polynomial (and convert to df)
        if self.self_term_only:
            X_poly = pd.DataFrame(data=self.expander.transform(X=X[self.cols_continuous]),
                                  columns=self.poly_cols_original,
                                  index=X.index)
        else:
            X_poly = pd.DataFrame(data=self.expander.transform(X=X),
                                  columns=self.poly_cols_original,
                                  index=X.index)
        
        # Subset to selected
        X_poly = X_poly[self.poly_cols_selected]
        
        # Construct transformed data
        X_poly = pd.DataFrame(data=X_poly.values,
                         columns=self.cols_updated,
                         index=X.index)
        
        # Concat
        X = pd.concat(objs=[X_poly,X_dummy], axis=1, ignore_index=False)
        
        # Drop duplicated cols (as some appear as dummies but are also included if interactions are allowed)
        X = X.iloc[:,~X.columns.duplicated()]
        
        # Obtatin new columns
        self.cols_new = [col for col in X.columns.tolist() if col not in self.cols_original]
        
        # Rearrange
        X = pd.concat(objs=[X[self.cols_original],X[self.cols_new]], axis=1, ignore_index=False)
        
        # We should not include dummies to higher terms as it makes no sense. Drop them next
        cols_to_be_dropped_because_dummy = []
        
        for col in X.columns:
            
            # Get power and interaction terms
            col_pow = col.split("^")[0]
            col_int = col.split("*")
            
            # Cases
            if (col_pow in self.cols_binary) and not any(c in self.cols_binary for c in col_int):
                # We infer that column must be dummy and of higher order. We should drop this
                cols_to_be_dropped_because_dummy.append(col)
                
            elif (col_pow not in self.cols_binary) and all(c in self.cols_binary for c in col_int) and (len(list(dict.fromkeys(col_int)))>len(col_int)):
                # We infer that column must be dummy and of interacted with itself. We should drop this
                cols_to_be_dropped_because_dummy.append(col)
                
        # Drop columns
        X.drop(columns=cols_to_be_dropped_because_dummy, inplace=True)
                
        return X
        
    # Public functions
    def fit(self,X):
        
        # Convert to dataframe
        X = bg.convert.convert_to_df(x=X)
        
        # Get dummy cols with only 2 distint values
        X_unique = X.nunique()
        self.cols_binary = X_unique.index[X_unique==2].tolist()
        self.cols_continuous = [col for col in X.columns if col not in self.cols_binary]

        # Obtain original columns
        self.cols_original = X.columns.tolist()

        # Fit polynomial
        if self.self_term_only:        
            # Since we only included terms like x^2, y^2 and NOT interactions (x*y), we focus on numerical columns and ignore dummy columns
            self.expander.fit(X=X[self.cols_continuous])
        else:
            self.expander.fit(X=X)
        
        # Extract cols assigned by expander (not what we want in the end)
        poly_cols_temp = self.expander.get_feature_names_out()
        
        # Update interactions terms from " " to "*"
        self.poly_cols_original = [col.replace(" ", "*") for col in poly_cols_temp]
        
        if self.self_term_only:
            # Get rid of interactions terms
            interaction_terms = [col for col in self.poly_cols_original if "*" in col]
            
            # Subset list
            self.poly_cols_selected = [col for col in self.poly_cols_original if col not in interaction_terms]
            
        else:
            self.poly_cols_selected = self.poly_cols_original.copy()
            
        # Extract poly cols corresponding to original cols
        poly_cols_main = self.poly_cols_selected[0:self.expander.n_features_in_]
        
        # Construct mapper between assigned poly cols and what we want
        self.cols_mapper = {poly_cols_main[i]:X.columns[i] for i in range(len(poly_cols_main))}
        
        # Apply mapper but take care of interactions and higher order terms
        self.cols_updated = []
        for col in self.poly_cols_selected:

            # Split col name into core variables
            col_core = re.sub(pattern="\\*|\\^", repl=" ", string=col).split(" ")
            
            for old_name, new_name in self.cols_mapper.items():
                if any(core==old_name for core in col_core):
                    col = col.replace(old_name,new_name)
                    
            self.cols_updated.append(col)
            
        # Update status
        self.expander_is_fitted_ = True
        
        if self.scale_vars:
            self.scaler.fit(X=self._transform_without_scaling(X=X))
            
        # Update final status
        self.is_fitted_ = True
            

        return self
        
    def transform(self,X):
        
        # Check status
        if not self.is_fitted_:
            raise Exception("Object if not fitted. Thus, call '.fit()' before proceeding")
            
        # Convert to dataframe
        X = bg.convert.convert_to_df(x=X)

        # Transform but not scale
        X = self._transform_without_scaling(X=X)
        
        if self.scale_vars:
            X.iloc[:,:] = self.scaler.transform(X=X)
            
        return X
                    
    def fit_transform(self,X):
        
        # Fit
        self.fit(X=X)
        
        # Transform
        X = self.transform(X=X)
        
        return X