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

function model = scikit_train_supervised(Xtrn, Ytrn, algo_name, algo_params, CV_strategy, CV_params_for_algo, CV_params)
    %will send the data to python. run CV or default parameters to find best CV parameters
    if nargin < 3
        error('Atleast supply an algo name')
    end
    if ~exist('algo_params','var')
        algo_params = py.dict;
    end
    
    %if both CV_params_for_algo and CV_params are not available
    %just don't do CV
    if ~exist('CV_strategy','var')
        DoCV = false;
    else
        DoCV = true;
    end
    
    if ~exist('CV_params_for_algo','var')
        CV_params_for_algo = py.dict;
    end
    if ~exist('CV_params','var')
        CV_params = py.dict;
    end
    
    %note that array data is represented differently in C (xgboost) / matlab
    %colum major, row major. so what we do is flip it before sending
    Xtrn = Xtrn';
    
    %call_for_xgboost_python
    if count(py.sys.path,'') == 0
        insert(py.sys.path,int32(0),'');
    end

    mod = py.importlib.import_module('scikit_train_predict_supervised');
    if str2num(pyversion) >= 3
        py.importlib.reload(mod); %python >= version 3
    else
        py.reload(mod);  %python == version 2.xx
    end


    
    %note how size(,2), size(,1) is shown because we have flipped the data
    %so that the order is preserved, but when reshaped via numpy on python
    %side what we do is reshape to the original N,D rather than the
    %transposed D,N size
    pyXtrn = py.list(cell({Xtrn(:).', int32(size(Xtrn,2)), int32(size(Xtrn,1))}));
    pyYtrn = py.list(cell({Ytrn(:).', int32(size(Ytrn,1)), int32(size(Ytrn,2))}));
    
    if DoCV
        model = py.scikit_train_predict_supervised.trainCV(pyXtrn, pyYtrn, algo_name, algo_params, CV_strategy, CV_params_for_algo, CV_params);
    else
        model = py.scikit_train_predict_supervised.train(pyXtrn, pyYtrn, algo_name, algo_params);    
    end
end
