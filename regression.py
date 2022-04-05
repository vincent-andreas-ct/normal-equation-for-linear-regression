import numpy as np

class build_model:
    """Fitting Linear Regression to the given data."""
    
    def __init__(self, features, target):
        self.features = features
        self.target = target
    
    def check_features(self):
        """Check all the feature whether it does has feature zero or doesn't"""
        if type(self.features)==type(np.array([0])):
            try:
                if self.features.shape[1]>1:
                    first_column = self.features[:,0]
                    equal_one = [each==1 for each in first_column]
                    if (sum(equal_one)/len(equal_one))==1:
                        pass
                    else:
                        new_features = []
                        for each in self.features:
                            temp = [1]
                            for value in each:
                                temp.append(value)
                            new_features.append(temp)
                        
                        setattr(self, "features", new_features)
            except IndexError:
                new_features = []
                for value in self.features:
                    new_features.append([1, value])
                
                setattr(self, "features", new_features)
        
        elif type(self.features)==type([0]):
            new_features = []
            for each in self.features:
                try:
                    temp = [1]
                    temp.extend(each)
                    new_features.append(temp)
                except TypeError:
                    temp = [1]
                    temp.extend([each])
                    new_features.append(temp)
            
            setattr(self, "features", new_features)
        else:
            raise TypeError("Feature and target only accepts Python List or NumPy Array!")
    
    def set_matrix(self):
        """Set all feature and target to a matrix"""
        setattr(self, "features", np.matrix(self.features))
        
        new_target = []
        if type(self.target)==type([0]):
            for each in self.target:
                if type(each)==type([0]):
                    new_target.append(each)
                else:
                    new_target.append([each])
            
            setattr(self, "target", np.matrix(new_target))
        elif type(self.target)==type(np.array([0])):
            if self.shape[1]==1:
                new_target = np.matrix(self.target)
                
                setattr(self, "target", new_target)
            else:
                new_target = []
                for each in self.target:
                    new_target.append([each])
                
                setattr(self, "target", new_target)
        else:
            raise TypeError("Feature and target only accepts Python List or NumPy Array!")
            
        """
        new_target = []
        for each in self.target:
            new_target.append([each])
        
        setattr(self, "target", np.matrix(new_target))"""
    
    def fit_model(self):
        """Fit linear regression to the features"""
        self.check_features()
        self.set_matrix()
        
        features_t = self.features.transpose()
        product = features_t * self.features
        product_inv = product.getI()
        
        theta = product_inv * features_t * self.target
        setattr(self, "theta", np.asarray(theta).flatten())
    
    def show_params(self, num: int=2):
        """Show parameters from the result of fitting linear regression to
        the data"""
        for i in range(num):
            print("THETA {}: {}".format(i, self.theta[i]))
        return self.theta
    
    def calc_cost(self):
        """calculate the cost of the model with respect to the data"""
        params = np.matrix([[each] for each in self.theta])
        y_pred = self.features * params
        
        diff_y = y_pred - self.target
        temp = diff_y.transpose()*diff_y
        cost = temp.flatten()/(2*len(self.features))
        print("COST: {}".format(cost))
        return cost