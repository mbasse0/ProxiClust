# import gpytorch
# from gpytorch.models import ExactGP
# from gpytorch.likelihoods import GaussianLikelihood
# from gpytorch.mlls import ExactMarginalLogLikelihood
import torch
# from tqdm import tqdm

# import torch.nn as nn
# import torch.optim as optim
#gp with sklearn
from sklearn.gaussian_process.kernels import ConstantKernel as C
# from sklearn.gaussian_process.kernels import WhiteKernel
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
# from sklearn.gaussian_process.kernels import RBF
#dot product kernel
# from sklearn.gaussian_process.kernels import DotProduct
#rq kernel
from sklearn.gaussian_process.kernels import RationalQuadratic
#matern kernel
# from sklearn.gaussian_process.kernels import Matern
#exponential kernel

import numpy as np
# from sklearn.gaussian_process.kernels import Kernel, ConstantKernel

class gp_predictor_sklearn():
    def __init__(self, train_x, train_y):
        # #tensor to array
        # train_x=train_x.detach().numpy()
        # train_y=train_y.detach().numpy()
        #remove average
        y_average=np.mean(train_y)
        train_y_centered=train_y-np.mean(train_y)

        # Initialize the Gaussian Process Regressor with the RBF kernel
            #  kernel = RBF()*C()++ WhiteKernel(noise_level_bounds=(1e-5, 1))
        #best:         kernel = RBF()*C()
        # kernel=RationalQuadratic(length_scale=0.001,length_scale_bounds=(1e-10, 1e5))
        # kernel=RationalQuadratic(alpha=1,length_scale=0.001,length_scale_bounds=(1e-10, 100))+DotProduct(sigma_0=1, sigma_0_bounds=(1e-10, 1e5))*5

        kernel=RationalQuadratic(alpha=1,length_scale=0.001,length_scale_bounds=(1e-10, 1))+1*RationalQuadratic(length_scale=10,length_scale_bounds=(1e-6, 1e3), alpha_bounds=(1e-10, 1e8))
        
        
        # kernel=WhiteKernel(noise_level_bounds=(1e-50, 1))+RationalQuadratic(length_scale=0.001,length_scale_bounds=(1e-10, 1e5))
        # kernel=Matern(length_scale=0.001,length_scale_bounds=(1e-10, 1e5))
        #best:         kernel =++ WhiteKernel(noise_level_bounds=(1e-50, 1))+RationalQuadratic(length_scale=0.001,length_scale_bounds=(1e-10, 1e5))
        #list all kernels in sklearn#


        self.model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=0, random_state=0)

        self.train_x = train_x
        self.train_y = train_y
        self.train_y_centered=train_y_centered
        self.y_average=y_average


        #random seed
        
    def train_pred(self, num_epochs=100, learning_rate=0.1):
        # Fit the GP model to the training data
        self.model.fit(self.train_x, self.train_y)
        print("Learned kernel: %s" % self.model.kernel_)
    
    def predict_pred(self, test_x):
        #tenspr to array
        test_x=test_x
        # Make predictions on the test data
        mean_predictions, std_predictions = self.model.predict(test_x, return_std=True)
        var_predictions=std_predictions**2
        #to tensor
        #add average
        mean_predictions=mean_predictions#+self.y_average
        mean_predictions=torch.from_numpy(mean_predictions)
        var_predictions=torch.from_numpy(var_predictions)
        return mean_predictions, var_predictions
    