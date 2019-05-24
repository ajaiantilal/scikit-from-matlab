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

function list_of_algorithm = scikit_list_algorithms
    list_of_algorithm = py.scikit_train.list_of_algorithms();