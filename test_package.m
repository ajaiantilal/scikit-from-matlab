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

% This file uses all available (supervised) algorithms in scikit and
% runs thems on some data

scikit_setup; %%%%%%%%%%%%%call this script to set things up  <- important to use this script first

addpath('data'); %added data dir to the path

load twonorm % we just use a classification dataset for both regression/classification (toy example)

Xtrn = inputs';
Ytrn = outputs;

Xtst = Xtrn;
Ytst = Ytrn;

list_of_algorithms = scikit_list_algorithms;

for i = 1:length(list_of_algorithms)
    algo_name = char(list_of_algorithms{i});

    fprintf('Running %s with default parameters ', algo_name);
    tic
    model = scikit_train_supervised(Xtrn, Ytrn, algo_name);
    train_time = toc;
    tic
    ypred = scikit_predict_supervised(Xtst, model);
    test_time = toc;
    fprintf(' train time %0.3fs, test time %0.3fs\n', train_time, test_time);
    norm_diff(i) = norm(ypred - Ytst)/sqrt(length(ypred));  %norm difference
end

fprintf('%30s\t\tNormDiff\n','Algorithm');

[~, indx] = sort(norm_diff);
for i = 1:length(list_of_algorithms)
    algo_name = char(list_of_algorithms{indx(i)});
    fprintf('%30s\t\t%0.3f\n', algo_name, norm_diff(indx(i)));
end
