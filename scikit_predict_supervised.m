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

function ypred = scikit_predict_supervised(Xtst, model)
    %will send the data to python. run CV or default parameters to find best CV parameters
    if nargin < 2
        error('Need to have test data AND model to evaluate')
    end
    
    %note that array data is represented differently in C (xgboost) / matlab
    %colum major, row major. so what we do is flip it before sending
    Xtst = Xtst';
        
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
    pyXtst = py.list(cell({Xtst(:).', int32(size(Xtst,2)), int32(size(Xtst,1))}));    
    
    ypred = py.scikit_train_predict_supervised.predict(pyXtst, model);
    ypred = from_numpy_array_to_matlab(ypred)';    
end

function y = from_numpy_array_to_matlab(y_)
    y = double(py.array.array('d',py.numpy.nditer(y_)));
end