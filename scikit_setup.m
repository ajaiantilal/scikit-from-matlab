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

function  scikit_setup(package_path)
%%%%%%%%%%%%%%%%%This is all init stuff that python-matlab requires. you
%%%%%%%%%%%%%%%%%can call it at init

%call to add python files to the path
if count(py.sys.path,package_path) == 0
    insert(py.sys.path,int32(0),package_path);
end

mod = py.importlib.import_module('scikit_train_predict_supervised');
if str2num(pyversion) >= 3
    py.importlib.reload(mod); %python >= version 3
else
    py.reload(mod);  %python == version 2.xx
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
