##% This file is part of scikit-from-matlab.
##% 
##% scikit-from-matlab is free software: you can redistribute it and/or modify
##% it under the terms of the GNU General Public License as published by
##% the Free Software Foundation, either version 3 of the License, or
##% (at your option) any later version.
##% 
##% scikit-from-matlab is distributed in the hope that it will be useful,
##% but WITHOUT ANY WARRANTY; without even the implied warranty of
##% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##% GNU General Public License for more details.
##% 
##% You should have received a copy of the GNU General Public License
##% along with scikit-from-matlab.  If not, see <https://www.gnu.org/licenses/>.
##
##% Author: Abhishek Jaiantilal (abhirana@gmail.com)
## scikit-from-matlab 0.0.1

##if you run this script in python, it will construct numpy arrays  (of type that matlab also passes to the script)
##and runs all the algorithms below to test them.
##note that if your favorite algorithm is missing (either missing in scikit or not mentioned below), it is very easy to add it below
##missing in scikit: you need to have (i think) a fit, score, predict function defined http://danielhnyk.cz/creating-your-own-estimator-scikit-learn/
##missing below: what i have done is make a dict of algo->import-library
##let's say you wanted to add RandomForestRegressor & RandomForestClassifier (already added but just as an example), then you will be defining
##rf = ['RandomForestRegressor', 'RandomForestClassifier']
##rf_lib = ['sklearn.ensemble'] #<- the library from where both the algorithm can be imported
## modify __return_external_libs__() function and add: external_libs.update( __construct_mapping_algo_to_lib__(rf, rf_lib) )
##if you want to add a new CV algorithm the same idea goes and you just modify this dict CV_search_algorithms

try:
    import numpy as np
except ImportError:
    print('Install Numpy/Scipy (https://scipy.org/install.html) it can be as easy as pip install numpy scipy --user on the command line')

try:
    import sklearn
    from sklearn.model_selection import cross_validate, GridSearchCV   #Additional scklearn functions
    from sklearn import datasets
except ImportError:
    print('Install scikit-learn (https://scikit-learn.org/stable/install.html) it can be as easy as pip install scikit-learn --user on the command line')

import importlib, warnings, sys, traceback,math

#disable deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning) 


#Mapping of Algorithms to the library they come from. Modify here if some algorithm is missing
#e.g. GLM comes from sklearn.linear_model
#what we do is at runtime import the algorithm from the library
GLM = [
    'ARDRegression', 'BayesianRidge', 'ElasticNet','ElasticNetCV', 'HuberRegressor','Lars',
    'LarsCV','Lasso','LassoCV','LassoLars','LassoLarsCV','LassoLarsIC','LinearRegression','LogisticRegression',
    'LogisticRegressionCV',
    'OrthogonalMatchingPursuit','OrthogonalMatchingPursuitCV','PassiveAggressiveClassifier',
    'PassiveAggressiveRegressor','Perceptron','Ridge','RidgeCV','RidgeClassifier','RidgeClassifierCV',
    'SGDClassifier','SGDRegressor','TheilSenRegressor',
    #,'RANSACRegressor' - was seeming to crap out
    #,'MultiTaskElasticNet','MultiTaskElasticNetCV''MultiTaskLassoCV''MultiTaskLasso', - seems to not work with the twonorm dataset
    ]
GLM_lib = ['sklearn.linear_model'] 

#discriminant analysis family
Discriminant = ['LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis']
Discriminant_lib = ['sklearn.discriminant_analysis']

#ensemble family
Ensemble = ['AdaBoostClassifier', 'AdaBoostRegressor','BaggingClassifier','BaggingRegressor', 'ExtraTreesClassifier',
            'ExtraTreesRegressor', 'GradientBoostingClassifier','GradientBoostingRegressor','IsolationForest',
            'RandomForestClassifier','RandomForestRegressor', 
            #,'VotingClassifier''VotingRegressor',,'RandomTreesEmbedding'
            #not found 'HistGradientBoostingRegressor','HistGradientBoostingClassifier'
            ]
Ensemble_lib = ['sklearn.ensemble']

#Xgboost is separate from the scikit base so
XGboost = ['XGBRegressor', 'XGBClassifier']
XGboost_lib = ['xgboost.sklearn']

#gaussian process family
Gaussian_processes = ['GaussianProcessClassifier', 'GaussianProcessRegressor']
Gaussian_processes_lib = ['sklearn.gaussian_process']

#kernel ridge family
Kernel_ridge = ['KernelRidge']
Kernel_ridge_lib = ['sklearn.kernel_ridge']

# svm family
SVM = ['LinearSVC', 'LinearSVR', 'NuSVC', 'NuSVR', 'SVC', 'SVR']
SVM_lib = ['sklearn.svm']

#decision tree family
DecisionTrees = ['DecisionTreeClassifier', 'DecisionTreeRegressor', 'ExtraTreeClassifier','ExtraTreeRegressor']
DecisionTrees_lib = ['sklearn.tree']


#CV search types and mapping to the libraries
CV_search_algorithms =  {'GridSearchCV':'sklearn.model_selection', 'RandomizedSearchCV':'sklearn.model_selection'}

def __construct_mapping_algo_to_lib__(algo_list, lib_family):
    #constructs a mapping from algos to the library. note that as multiple algos may come from a single family
    #this will just construct a mapping dict
    return dict(zip(algo_list, lib_family * len(algo_list)))

def __return_external_libs__():
    #constructs the mapping of different algorithm to their respective libraries
    external_libs = {}
    external_libs.update( __construct_mapping_algo_to_lib__(GLM, GLM_lib) )
    external_libs.update( __construct_mapping_algo_to_lib__(Discriminant, Discriminant_lib) )
    external_libs.update( __construct_mapping_algo_to_lib__(Ensemble, Ensemble_lib) )
    external_libs.update( __construct_mapping_algo_to_lib__(XGboost, XGboost_lib) )
    external_libs.update( __construct_mapping_algo_to_lib__(Gaussian_processes, Gaussian_processes_lib) )
    external_libs.update( __construct_mapping_algo_to_lib__(Kernel_ridge, Kernel_ridge_lib) )
    external_libs.update( __construct_mapping_algo_to_lib__(SVM, SVM_lib) )
    external_libs.update( __construct_mapping_algo_to_lib__(DecisionTrees, DecisionTrees_lib) )
    #if you want to add an existing algorithm from a library add it here
    return (external_libs)

external_libs = __return_external_libs__()

def list_of_algorithms():
    #send a list of all algorithms known
    return list(external_libs.keys())

def create_algo_object_with_params(algo_name, params):
    '''
    Based on algorithm to run, we try to load the package/module required to run the package
    Then we load the sub-module within that package and pass the params that were given by the
    user and then return the object
    '''
    try:
        #print(algo_name)
        if algo_name in external_libs:
            #print(external_libs[algo_name])
            algo_module = importlib.import_module(external_libs[algo_name])
        algo_object = getattr(algo_module, algo_name)(**params)
        #print(algo_object)
    except Exception as e:
        sys.stdout.write(__file__ + traceback.format_exc())
        raise
    return (algo_object)

def create_CV_object_with_params(CV_name, algo_object, CV_params_for_algo, CV_params):
    '''
    Based on CV search to run, we try to load the package/module required to run the package
    Then we load the sub-module within that package and pass the params that were given by the
    user and then return the object
    '''
    try:
        if CV_name in CV_search_algorithms:
            #print(external_libs[CV_name])
            CV_module = importlib.import_module(CV_search_algorithms[CV_name])
        try:
            algo_object = getattr(CV_module, CV_name)(algo_object, param_grid = CV_params_for_algo, **CV_params)
        except ValueError as e:
            warnings.warn('Ensure that the parameter passed as CV parameters are correct')
            raise
        #print(algo_object)
    except Exception as e:
        sys.stdout.write(__file__ + traceback.format_exc())
        raise
    return (algo_object)

def __reshape_np_array(x):
    #matlab sends in a list (numpy array flattened, dim_1 size, dim_2 size)
    #what we do is reshape the numpy array from 1D back to 2D
    #no need to do that for label/targets/y
    if x[2]==1:
        return np.array(x[0][:])
    else:
        return np.array(x[0][:]).reshape(x[1],x[2])
    
def train(xtrn, ytrn, algo_name, algo_params):
    #we reshape the input X array to 2D
    #then create an algorithm object depending on the name of algo and params passed
    #then use the data with the algorithm using the fit function
    try:
        reshaped_Xtrn = __reshape_np_array(xtrn)
        reshaped_Ytrn = __reshape_np_array(ytrn)

        algo_object = create_algo_object_with_params(algo_name, algo_params)
        algo_object.fit(reshaped_Xtrn, reshaped_Ytrn)
    except Exception as e:
        sys.stdout.write(__file__ + traceback.format_exc())
        raise
    return(algo_object)


def trainCV(xtrn, ytrn, algo_name, algo_params, CV_strategy, CV_params_for_algo, CV_params):
    #we reshape the input X array to 2D
    #then create an algorithm object depending on the name of algo and params passed
    #ALSO, create a CV object with params in conjuction with the algorithm object
    #then use the data with the algorithm using the fit function    
    try:
        reshaped_Xtrn = __reshape_np_array(xtrn)
        reshaped_Ytrn = __reshape_np_array(ytrn)

        for key in CV_params_for_algo:
            CV_params_for_algo[key] = CV_params_for_algo[key].tolist()
        algo_object = create_algo_object_with_params(algo_name, algo_params)
        clf = create_CV_object_with_params(CV_strategy, algo_object, CV_params_for_algo, CV_params)
        clf.fit(reshaped_Xtrn, reshaped_Ytrn)
    except Exception as e:
        sys.stdout.write(__file__ + traceback.format_exc())
        raise
    return(clf)


def predict(xtst, clf):
    #we reshape the input Xtst array to 2D and get predictions on the input array Xtst    
    try:
        reshaped_Xtst = __reshape_np_array(xtst)
        ypred = clf.predict(reshaped_Xtst)
    except Exception as e:
        sys.stdout.write(__file__ + traceback.format_exc())
        raise
    return (ypred)



def TestMe():
    def reshape_to_mimic_matlab_inputs(data):
        data_list = list()
        data_shape = [x for x in data.shape]
        if len(data_shape)==1:
            data_shape.append(1)
            data_list.append(data)
        else:
            data_list.append(data.reshape(data_shape[0], data_shape[1]))
        data_list.append(data_shape[0])
        data_list.append(data_shape[1])
        return(data_list)
    
    
    with open("data/X_twonorm.txt") as f:
        data = np.loadtxt(f)
    with open("data/Y_twonorm.txt") as f:
        label = np.loadtxt(f)
    
    #reshape to make it the same format as the matlab call
    data_list = reshape_to_mimic_matlab_inputs(data)
    label_list= reshape_to_mimic_matlab_inputs(label)

    list_algorithms = list_of_algorithms()



    #Without CV, just the default parameters
    res = []
    for algo in list_algorithms:
        clf = train(data_list, label_list, algo, dict())
        ypred = predict(data_list, clf)        
        res.append(np.linalg.norm((ypred - label)/math.sqrt(len(ypred))))

    sort_indx = np.argsort(res)

    print('Testing algorithms with default Parameters')
    print('%30s  %s' %('Algorithm','norm diff'))
    for i in range(len(res)):
        print('%30s  %0.3f' %(list_algorithms[sort_indx[i]], res[sort_indx[i]]))



    #With CV
    res = []
    cv_type = []

    print('\n\nTesting algorithms with CV and default Parameters, technically just testing if CV is working correctly')
    print('RandomizedSearchCV requires a param grid so omitting in testing below')
    for CV_type in CV_search_algorithms.keys():
        if CV_type=='RandomizedSearchCV':
            continue
        
        for algo in list_algorithms:
            if algo=='IsolationForest':
                print('IsolationForest was requiring a score so omitting in testing below')
                continue
            clf = trainCV(data_list, label_list, algo, dict(), CV_type, dict(), dict())
            ypred = predict(data_list, clf)        
            res.append(np.linalg.norm((ypred - label)/math.sqrt(len(ypred))))
            cv_type.append(CV_type)

    sort_indx = np.argsort(res)

    print('%30s  %s  %s' %('Algorithm','norm diff', 'CV strategy'))
    for i in range(len(res)):
        print('%30s  %0.3f  %15s' %(list_algorithms[sort_indx[i]], res[sort_indx[i]], cv_type[sort_indx[i]]))


if __name__ == "__main__":
    TestMe()
