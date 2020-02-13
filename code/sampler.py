from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NeighbourhoodCleaningRule
from imblearn.over_sampling import RandomOverSampler

# samplers to apply
samplers = [RandomOverSampler(0.3), TomekLinks(), NeighbourhoodCleaningRule()]


class IterativeSampler():
    """
    Applies a sequence of samplers. If the "return_ids" parameter
    is set to True, also returns the inices in the data frame or
    the matrix of the samples that have been selected. 
    """
    def __init__(self, samplers=samplers, return_ids=True):
        self.samplers = samplers
        self.return_ids = return_ids
        if return_ids:
            self.dict_ids = {}
        
    def fit_sample(self, X, y):

        for sampler in self.samplers:
            X, y = sampler.fit_sample(X, y)
            if self.return_ids:
                new_ids = list(range(len(y)))
                if self.dict_ids:
                    d0 = dict(zip(new_ids, sampler.sample_indices_))
                    d = {k: self.dict_ids[v] for k, v in d0.items()}
                else:
                    d = dict(zip(new_ids, sampler.sample_indices_))
                    
                self.dict_ids = d
                
        if self.return_ids:
            return X, y, self.dict_ids
        else:
            return X, y
