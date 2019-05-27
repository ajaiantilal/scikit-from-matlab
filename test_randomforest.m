% This file is part of scikit-from-matlab.
% 
% scikit-from-matlab is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% scikit-from-matlab is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with scikit-from-matlab.  If not, see <https://www.gnu.org/licenses/>.

% Author: Abhishek Jaiantilal (abhirana@gmail.com)
% scikit-from-matlab 0.0.1

% This uses RandomForest as an example to show how to call the RandomForest algorithm
% as-is (default parameters), custom values for parameters,
% use a CV strategy and pass custom values for CV parameters 


%This is a small test of the scikit_from_matlab package using RandomForest it shows how 
%for windows i put in the dll files in a subfolder. seems like that is the
%only way i could run RandomForest. maybe expanding the matlab python path to
%include that will help

clear classes %clears the py classes. seems like 'clear all' doesn't clear these python classes

path_to_package = '.';
scikit_setup(path_to_package); %%%%%%%%%%%%%call this script to set things up  <- need to set this up

addpath('data');

num_threads = 8; %change this to how many threads are available
%it seems that both GridSearchCV and RandomForest uses only 1 thread if this variable is not specified
%so make sure to set n_jobs to num_threads (just search for n_jobs in this
%code), i have it complain on an old version of matlab and i had to comment it out

%%%% Classification example
%load twonorm, fudge the Ytrn to add a new class. then send and see if it
%works
load twonorm

Xtrn = inputs';
Ytrn = outputs;

Ytrn(end-10:end) = 3;

Xtst = Xtrn;
Ytst = Ytrn;


%Let's try RandomForest https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

% Classification with default parameters
algo_name = 'RandomForestClassifier';

fprintf('Tutorial 1: running %s with default parameters\n', algo_name);
tic
model = scikit_train_supervised(Xtrn, Ytrn, algo_name);
toc
tic
ypred = scikit_predict_supervised(Xtst, model);
toc
accuracy = length(find((ypred ~= Ytst))) / length(ypred) %accuracy

%now we send in some algo_params, lets try with n_estimators=50
fprintf('Tutorial 2: running %s with additional parameter of num estimators = 50\n', algo_name);
%create a dict(ionary) object and pass in the argument. Make sure if its
%int or float or whatever type required in scikit algorithm
algo_params = py.dict;
update(algo_params, py.dict(pyargs('n_estimators',int64(50))))
update(algo_params, py.dict(pyargs('n_jobs',int64(num_threads))))
tic
model = scikit_train_supervised(Xtrn, Ytrn, algo_name, algo_params);
toc
fprintf('model now has this many n_esimators %d\n', int64(model.n_estimators))
tic
ypred = scikit_predict_supervised(Xtst, model);
toc
accuracy = length(find((ypred ~= Ytst))) / length(ypred) %accuracy


%now we use some form of CV, lets try GridSearchCV, with search over
%lets try max_depth, learning_rate and gamma for RandomForest
fprintf('Tutorial 3: running %s with additional parameter of GridSearchCV and trying (max_features, n_estimators) search\n', algo_name);
algo_params = py.dict;
CV_strategy = 'GridSearchCV';
CV_params_for_algo = py.dict;
update(CV_params_for_algo, py.dict(pyargs('n_estimators',	   int32(100:200:500) )))
update(CV_params_for_algo, py.dict(pyargs('max_features',	   int32(1:2:10) )))
tic
model = scikit_train_supervised(Xtrn, Ytrn, algo_name, algo_params, CV_strategy, CV_params_for_algo);
toc
fprintf('model parameter searched over\n')
model.param_grid
tic
ypred = scikit_predict_supervised(Xtst, model);
toc
accuracy = length(find((ypred ~= Ytst))) / length(ypred) %accuracy



%now we use some form of CV, lets try GridSearchCV, with search over
%lets try max_depth, learning_rate and gamma for RandomForest 
%and pass parameters to GridSearchCV
fprintf('Tutorial 4: running %s with additional parameter of GridSearchCV and trying (max_features, n_estimators) search\n', algo_name);
fprintf('Furthermore we now select a CV time of 4 and do refit=true\n')
algo_params = py.dict;
CV_strategy = 'GridSearchCV';
CV_params_for_algo = py.dict;
update(CV_params_for_algo, py.dict(pyargs('n_estimators',	   int32(100:200:500) )))
update(CV_params_for_algo, py.dict(pyargs('max_features',	   int32(1:2:10) )))

%https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
CV_params = py.dict;
update(CV_params, py.dict(pyargs('cv',    int64(4)     )))
update(CV_params, py.dict(pyargs('refit', int64(true)  )))
update(CV_params, py.dict(pyargs('n_jobs', int64(-1)  ))) %-1 means all processors

tic
model = scikit_train_supervised(Xtrn, Ytrn, algo_name, algo_params, CV_strategy, CV_params_for_algo, CV_params);
toc
fprintf('CV parameters are now\n')
model.cv
model.refit
tic
ypred = scikit_predict_supervised(Xtst, model);
toc
accuracy = length(find((ypred ~= Ytst))) / length(ypred) %accuracy




%regression example
load diabetes
algo_name = 'RandomForestRegressor';


Xtrn = diabetes.x;
Ytrn = diabetes.y;
Xtst = Xtrn;
Ytst = Ytrn;

fprintf('Tutorial 5: running %s with default parameters\n', algo_name);
tic
model = scikit_train_supervised(Xtrn, Ytrn, algo_name);
toc
tic
ypred = scikit_predict_supervised(Xtst, model);
toc
norm_diff = norm(ypred - Ytst)/sqrt(length(ypred))  %norm difference


%now we send in some algo_params, lets try with n_estimators=50
fprintf('Tutorial 6: running %s with additional parameter of num estimators = 50\n', algo_name);
%create a dict(ionary) object and pass in the argument. Make sure if its
%int or float or whatever type required in scikit algorithm
algo_params = py.dict;
update(algo_params, py.dict(pyargs('n_estimators',int64(50))))
update(algo_params, py.dict(pyargs('n_jobs',int64(num_threads))))
tic
model = scikit_train_supervised(Xtrn, Ytrn, algo_name, algo_params);
toc
tic
ypred = scikit_predict_supervised(Xtst, model);
toc
norm_diff = norm(ypred - Ytst)/sqrt(length(ypred))  %norm difference


%now we use some form of CV, lets try GridSearchCV, with search over
%lets try max_depth, learning_rate and gamma for RandomForest
fprintf('Tutorial 7: running %s with additional parameter of GridSearchCV and trying (max_features, n_estimators) search\n', algo_name);
algo_params = py.dict;
CV_strategy = 'GridSearchCV';
CV_params_for_algo = py.dict;
update(CV_params_for_algo, py.dict(pyargs('n_estimators',	   int32(100:200:500) )))
update(CV_params_for_algo, py.dict(pyargs('max_features',	   int32(1:2:10) )))
tic
model = scikit_train_supervised(Xtrn, Ytrn, algo_name, algo_params, CV_strategy, CV_params_for_algo);
toc
fprintf('model parameter searched over\n')
model.param_grid
tic
ypred = scikit_predict_supervised(Xtst, model);
toc
norm_diff = norm(ypred - Ytst)/sqrt(length(ypred))  %norm difference


%now we use some form of CV, lets try GridSearchCV, with search over
%lets try max_depth, learning_rate and gamma for RandomForest 
%and pass parameters to GridSearchCV
fprintf('Tutorial 8: running %s  with additional parameter of GridSearchCV and trying (max_features, n_estimators) search\n', algo_name);
fprintf('Furthermore we now select a CV time of 4 and do refit=true\n')
algo_params = py.dict;
CV_strategy = 'GridSearchCV';
CV_params_for_algo = py.dict;
update(CV_params_for_algo, py.dict(pyargs('n_estimators',	   int32(100:200:500) )))
update(CV_params_for_algo, py.dict(pyargs('max_features',	   int32(1:2:10) )))

%https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
CV_params = py.dict;
update(CV_params, py.dict(pyargs('cv',    int64(4)     )))
update(CV_params, py.dict(pyargs('refit', int64(true)  )))
update(CV_params, py.dict(pyargs('n_jobs', int64(-1)  ))) %-1 means all processors

tic
model = scikit_train_supervised(Xtrn, Ytrn, algo_name, algo_params, CV_strategy, CV_params_for_algo, CV_params);
toc
fprintf('CV parameters are now: CV=%s, model.refit=%s\n', char(model.cv),char(model.refit))
tic
ypred = scikit_predict_supervised(Xtst, model);
toc
norm_diff = norm(ypred - Ytst)/sqrt(length(ypred))  %norm difference
