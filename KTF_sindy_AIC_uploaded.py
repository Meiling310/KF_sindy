import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import Lasso
from pysindy.optimizers import FROLS
from pysindy.differentiation import FiniteDifference
from sklearn.model_selection import cross_validate, RepeatedKFold, cross_val_score, GridSearchCV, TimeSeriesSplit, ShuffleSplit

import pysindy as ps

from pysindy import SINDy
from pysindy.differentiation import FiniteDifference
from pysindy.feature_library import PolynomialLibrary
import os
import os.path as osp
from scipy.integrate import solve_ivp, RK45
from pysindy.utils import lorenz
from sklearn.metrics import r2_score
from pysindy.feature_library import CustomLibrary
from scipy.stats import t
import math
from itertools import combinations


from scipy.interpolate import interp1d

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams.update({'font.size': 14})

class SINDyCV(ps.SINDy):
    def __init__(self, optimizer=None, feature_library=None, differentiation_method=None, feature_names=None, t_default=0.1, discrete_time=False):
        super(SINDyCV, self).__init__(
            optimizer=optimizer,
            feature_library=feature_library,
            differentiation_method=differentiation_method,
            feature_names=feature_names,
            t_default=t_default,
            discrete_time=discrete_time)
        
    def set_coefficients(self, coefficients):
        """Set new coefficients for the model."""
        self.coefficients = coefficients
           
    def fit(self, x, y, **kwargs):  # library_ensemble=True, multiple_trajectories=True,
        return super(SINDyCV, self).fit(x, x_dot=y, **kwargs)
          
    def score(self, x, y, t=None, u=None, multiple_trajectories=False, metric=r2_score, **metric_kws):
        return super(SINDyCV, self).score(
            x,
            x_dot=y,
            t=t,
            u=u,
            multiple_trajectories=multiple_trajectories,
            metric=r2_score,
            **metric_kws)

### AIC calculation function
def calculate_aic(rss, n, k):
    log_likelihood = np.log(rss / n)
    aic = 2 * k + n* log_likelihood + 2*(k+1)*(k+2)/ (n-k-2)

    bic = n*log_likelihood + k* np.log(n)
    return aic

def get_effective_terms(coefficients, num_terms):
    """Retrieve the most significant coefficients up to num_terms."""
    abs_coeffs = np.abs(coefficients).flatten()
    #print(abs_coeffs,'abs_coeffs')
    # Ensure num_terms does not exceed the number of coefficients
    num_terms = min(num_terms, len(abs_coeffs))
    
    indices = np.argsort(abs_coeffs)[-num_terms:]
    #print(indices,'indices')
    limited_coefficients = np.zeros_like(coefficients)
    flat_limited_coefficients =limited_coefficients.flatten()
    flat_coefficients = coefficients.flatten()

    flat_limited_coefficients[indices] = flat_coefficients[indices]
 
    
    # Reshape back to original shape if needed
    limited_coefficients = flat_limited_coefficients.reshape(coefficients.shape)

    #print("Updated limited_coefficients:", limited_coefficients)
    
    return limited_coefficients

def get_combinations_of_terms(coefficients, num_terms):
    """Generate all combinations of num_terms coefficients."""
    # Flatten the coefficients to make it easier to index
    flat_coefficients = coefficients.flatten()
    
    # Get the indices of all coefficients
    all_indices = np.arange(len(flat_coefficients))
    
    # Generate all combinations of indices
    combinations_indices = list(combinations(all_indices, num_terms))
    
    # Store all combinations of coefficients
    combinations_list = []
    for indices in combinations_indices:
        limited_coefficients = np.zeros_like(flat_coefficients)
        limited_coefficients[list(indices)] = flat_coefficients[list(indices)]
        combinations_list.append(limited_coefficients.reshape(coefficients.shape))
    
    return combinations_list

def fH(H, H_t, effective_coeffs, feature_library):
    """
    Calculate the derivative dH/dt based on the state variables H and H_t.
    
    Parameters:
    H -- The current state variable.
    H_t -- The delayed state variable.
    effective_coeffs -- Coefficients for the model.
    feature_library -- Feature library used to transform the state.
    
    Returns:
    dHdt -- The derivative of H with respect to time.
    """
    feature_names = feature_library.get_feature_names()
    
    # Initialize the equation for dHdt
    dHdt = 0
    
    # Combine coefficients with terms to create the equation
    for coeff, term in zip(effective_coeffs[0], feature_names):  # Assuming single equation
        if coeff != 0:
            # Replace variables in the term with actual values H and H_t
            term = term.replace('x0', f'({H})').replace('x1', f'({H_t})')
            # Replace '^' with '**' for exponentiation
            term = term.replace('^', '**')
            # Replace spaces between variables with multiplication
            term = term.replace(' ', ' * ')
            # Evaluate the term and add to the equation
            dHdt += coeff * eval(term)
            
    
    return dHdt


def runge_kutta_integration(H0, delay_time, t, effective_coeffs, feature_library):
    """
    Integrate the ODE model over time using the Runge-Kutta method.
    
    Parameters:
    H0 -- Initial state vector [H0, H_t0].
    delay_time -- The delay time to be considered.
    t -- Time array.
    effective_coeffs -- Coefficients for the model.
    feature_library -- Feature library used to transform the state.
    
    Returns:
    H -- Integrated states over time.
    """
    num = len(t)
    H = np.zeros(num+delay_time)
    H[:delay_time+1] = H0[:delay_time+1]  # Initialize with initial conditions
    
    h = 0.1  # Time step size
    
    for i in range(delay_time, num+delay_time - 1):
        xa = H[i]
        xb = H[i - delay_time]
        
        # Runge-Kutta 4th order method
        k1 = fH(xa, xb, effective_coeffs, feature_library)
        k2 = fH(xa + 0.5 * h * k1, xb, effective_coeffs, feature_library)
        k3 = fH(xa + 0.5 * h * k2, xb, effective_coeffs, feature_library)
        k4 = fH(xa + h * k3, xb, effective_coeffs, feature_library)
        
        H[i + 1] = xa + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    return H


def integrate_model(H0, t, optimal_delay, effective_coeffs, feature_library):
    """
    Integrate the ODE model using the Runge-Kutta method.
    
    Parameters:
    H0 -- Initial state vector [H0, H_t0].
    t -- Time array.
    effective_coeffs -- Coefficients for the model.
    feature_library -- Feature library used to transform the state.
    
    Returns:
    H_pred -- Integrated states over time.
    """
    delay_time = optimal_delay  # Define delay time here or pass it as an argument
    H_pred = runge_kutta_integration(H0, delay_time, t, effective_coeffs, feature_library)
    return H_pred

### add delay
def delay_model(m, data_add, predict=False, AIC=False):
    print(f"Processing delay: {m}")

    x0_train = data_add[:, m:]
    x1_train = data_add[:, :data_add.shape[1] - m]
    x_train = np.dstack((x0_train, x1_train))
    print(x_train.shape,'x_train.shape')  #(104, 2901, 2) x_train.shape
    
    x_tri = []
    for i in range(2):
        aa = x_train[:, :, i]
        aa = aa.reshape(Ntri * (data_add.shape[1] - m),)
        x_tri.append(aa)
    x_tri = np.stack(x_tri, axis=1)

    fd = FiniteDifference()
    dt = 0.1
    t_train = np.arange(0, (data_add.shape[1] - m) / 10, dt)
    
    x_dot_tri = []
    for i in range(1):
        b = []
        for j in range(Ntri):
            a = fd(x_train[j, :, i], t=t_train)       
            b.append(a)
        b = np.stack(b, axis=0)
        b = b.reshape(Ntri * (data_add.shape[1] - m),)
        x_dot_tri.append(b)
    x_dot_tri = np.stack(x_dot_tri, axis=1)

    best_score = -np.inf
    best_aic = np.inf
    best_model = None
    best_num_terms = None
    aic_list =[]

    model = SINDyCV(
            feature_library=PolynomialLibrary(degree=2),
            differentiation_method=fd,
            optimizer=ps.STLSQ(max_iter=1000)   #normalize_columns=True
        )

    if (AIC == False):
        param_grid = {'optimizer__threshold': [10**-8,10**-4, 10**-2, 0.01],
                'optimizer__alpha': [0,1,0.5]}
               # 'feature_library__degree': [1, 2, 3]}

        search = GridSearchCV(model, param_grid, cv=ShuffleSplit(n_splits=4, test_size=0.20, random_state=0))
        search.fit(x_tri, y = x_dot_tri)
        print(search.get_params().keys())
        print('Best parameters:', search.best_params_)
        search.best_estimator_.print(precision=9)

        if (predict is False):
             return (search.best_score_ )
        else:
             return (search.best_estimator_)

    else: 
    
        # Lambda values for regularization
        optimizer__threshold = np.linspace(10**-4, 4*10**-4, 8)  #np.logspace(2, 3, 50)
        for lambda_val in optimizer__threshold:
             # Perform sparse regression

             model.optimizer.set_params(threshold=lambda_val)
        
             model.fit(x_tri, x_dot_tri)
             coefficients = model.coefficients()
             print(lambda_val,coefficients.shape,'lambda_val,coefficients.shape')

             rss = 0
             for pp in range(Ntri): 
                x_initial = data_add[pp, :]
                
                x_pred = integrate_model(x_initial, t_train, m, coefficients, model.feature_library)
        
                #print(x_pred.shape,'x_pred')
            
                residuals = np.abs(x_initial - x_pred)
                plt.plot(x_initial ,'r--')
                plt.plot(x_pred,'k')
                plt.xlabel('time')
                plt.legend(('true_values','model_prediction'))
                #plt.show()
                plt.close()
            
                #print(residuals.shape,'residuals.shape')

            
                rss = rss + np.sum(residuals**2)
            
             k = np.count_nonzero(coefficients)
             aic = calculate_aic(rss/Ntri, len(x_pred), k)
             print(aic, rss, len(x_pred), k, 'AIC, rss, len(x_dot_tri), k')
             print(coefficients,'coefficients')

             aic_list.append((k, lambda_val, aic))

             if (aic < best_aic):     
                best_aic = aic
                best_model = model
                best_num_terms = k
                best_coeffs = coefficients
                best_features = model.feature_library

                print(best_aic,best_num_terms,best_coeffs,'best_aic,best_num_terms,best_coeffs')

        return best_model, best_num_terms, best_aic, best_coeffs,aic_list,best_features


def get_max_score(scores):
    max_value = max(scores)
    max_index = scores.index(max_value)
    return max_value, max_index

### load data #####################
data_path = os.getcwd() + '/'
np.random.seed(100) 
H = np.loadtxt(osp.join(data_path, 'DATA/H_data'))
print(H.shape, 'H.shape')
H = H.reshape(130, 3250)

H1 = H[0:50, 100:]
H2 = H[50:76, 100:]
H3 = H[76:, 100:]
data_add = np.concatenate((H1, H3), axis=0)
Ntes = 26
Ntri = H.shape[0] - Ntes

#### optimize delay time ##########

# Compute AIC for different delay times and numbers of terms


tao = np.arange(0, 400, 1)

# Step 1: Get best scores for different time delays
scores = [delay_model(m, data_add, predict=False, AIC=False) for m in tao]
np.savetxt('best_scores.txt', scores)
print(scores)
plt.plot(tao, scores, 'o--')
plt.xlabel('Time Delay')
plt.ylabel('Best Score')
plt.savefig(osp.join(data_path, 'best_scores.png'))
plt.show()
plt.close()

# Step 2: Find the optimal delay
best_delay_score, optimal_delay_index = get_max_score(scores)

optimal_delay = tao[optimal_delay_index]
print(f"Optimal Delay Time: {optimal_delay}")

# Step 3: Get the model with the lowest AIC for the optimal delay
model_optimized, best_num_terms,best_aic, best_coeffs,aic_list,best_features = delay_model(optimal_delay, data_add, predict=True,AIC=True)

#print(f"Optimized Model: {model_optimized.print}")
print(f"Best Number of Terms: {best_num_terms}")

# Extract x and z values
num_terms = [entry[0] for entry in aic_list]
aic_values = [entry[2] for entry in aic_list]

# Find the minimum AIC value
min_aic = min(aic_values)
min_index = aic_values.index(min_aic)
print(aic_list[min_index],'aic_LISTS[min_index]')
print(min_aic,'min_aic')

# Calculate relative AIC values
relative_aic_values = [(aic - min_aic) for aic in aic_values]

# Create a scatter plot of x vs z
plt.figure(figsize=(6, 6))
plt.scatter(num_terms, aic_values, color='blue', marker='o')
plt.scatter(num_terms[min_index], aic_values[min_index], color='orange', marker='o', s=100)

# Add labels and title
plt.xlabel('Number of terms')
plt.ylabel('AIC score')
# Show the plot grid
plt.grid(True)
plt.savefig('AIC Values vs Number of Terms_1')
# Show the plot
plt.show()
plt.close()

# Create a scatter plot of num_terms vs relative_aic_values
plt.figure(figsize=(8, 6))
plt.scatter(num_terms, relative_aic_values, color='blue', marker='o')
plt.scatter(num_terms[min_index], relative_aic_values[min_index], color='orange', marker='o', s=100)
# Add labels and title
plt.xlabel('Number of terms')
plt.ylabel('Relative AIC score')
#plt.ylim(0, 10)
#plt.title('Relative AIC Values vs Number of Terms')

# Show the plot grid
plt.grid(True)
plt.savefig('Relative AIC Values vs Number of Terms_2')

# Show the plot
plt.show()
plt.close()



#### get optimize model ##############
#model_optimized = delay_model(delay_time, data_add, True)
delay_time = optimal_delay
print(model_optimized, 'model_optimized')
#print(model_optimized.get_params().keys())

### model evaluation with test datasets
x0_test = H2[:, delay_time:]
x1_test = H2[:, :data_add.shape[1] - delay_time]
diff_test = np.zeros([H2.shape[0], H2.shape[1] - delay_time])
for i in range(0, H2.shape[0]):
    for j in range(0, H2.shape[1] - delay_time):
        diff_test[i, j] = x0_test[i, j] - x1_test[i, j]
x_test = np.dstack((x0_test, x1_test))

### get derivates of test data
fd = FiniteDifference()
dt = 0.1
t_tes = np.arange(0, (data_add.shape[1] - delay_time) / 10, dt)
x_tes = []
for i in range(0, Ntes):
    aa = x_test[i, :, :]
    x_tes.append(aa)
x_tes = np.stack(x_tes, axis=0)
x_tes = x_tes.reshape(Ntes * (data_add.shape[1] - delay_time), 2)

x_dot_tes = []
for i in range(0, Ntes):
    bb = []
    for j in range(0, 2):
        aa = fd(x_test[i, :, j], t=t_tes)
        bb.append(aa)
    bb = np.stack(bb, axis=1)
    x_dot_tes.append(bb)
x_dot_tes = np.stack(x_dot_tes, axis=0)
x_dot_tes = x_dot_tes.reshape(Ntes * (data_add.shape[1] - delay_time), 2)
x_dot_tes = x_dot_tes[:, 0:1]

#evaluation_score = model_optimized.score(x_tes, y=x_dot_tes)
#print(evaluation_score, 'evaluation_score_test')

##### Predict derivatives with learned model
H_pre = []
for PP in range(0, Ntes):
    num = 3050 - 100
    x_tes1 = x_test[PP, :, :]  # 101 trajectory

    x_dot_test_predicted = np.dot(model_optimized.feature_library.transform(x_tes1),best_coeffs.T)
    print(x_dot_test_predicted.shape, 'x_dot_test_predicted.shape')

    x_dot_test_computed = fd(x_tes1, t=dt)
    print(x_dot_test_computed.shape, 'x_dot_test_computed.shape')

    #### plot
    plt.plot(t_tes, x_dot_test_predicted[:, 0], 'k')
    plt.plot(t_tes, x_dot_test_computed[:, 0], 'r--')
    plt.xlabel('time')
    plt.ylabel('$\dot H_{%d}$' % PP)
    plt.legend(('model_prediction', 'numberical_derivative'))
    plt.savefig(osp.join(data_path, 'results/test_H_%d' % PP))
    # plt.show()
    plt.close()

    tr =np.arange(0,(num-delay_time), 1)

    H_initial = np.zeros([num-delay_time,]) 
    H_initial[0:delay_time+1] = H2[PP,0:delay_time+1]

  
    H_pred = integrate_model(H_initial,t_tes, delay_time, best_coeffs, best_features)


    H_values = np.stack(H_pred, axis =0)
  
    plt.plot(H_values[:],'r--')
    plt.plot(x_tes1[:,0],'k')
    plt.xlabel('time')
    plt.ylabel('$H_{%d}$' %PP)
    plt.legend(('model_prediction','true_values'))
    plt.savefig(osp.join(data_path,'results/H_predicition_%d' %PP))
    #plt.show()
    plt.close()    
  
    H_pre.append(H_values)

H_pre = np.stack(H_pre, axis=0)
print(H_pre.shape,'H_pre.shape')



