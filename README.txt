Matlab and Python (sklearn) both have a large domain of algorithms. Unfortunately Matlab doesn't contain all the algorithms required or may require additional (pricey) toolboxes. And then there is the case of reproducibility of results. Not all researchers use both languages fluently or it may be hard to diversify their current matlab code to use algorithms present in sklearn

So, this is a try to use sklearn within matlab and in the first version use supervised algorithms in sklearn from within matlab. Hopefully, if it works for you, and if you have any updates, let me know and i can integrate them in. My hope is that one day most features of sklearn/numpy/scipy could be used within matlab, including pipelines, unsupervised learning, CV, clustering among others

Use test_package.m to run ALL supervised learning algorithms available in scikit on a test dataset

Use test_randomforest.m to run randomforest algorithm on a test dataset in a way where you can either use the default parameters, supply parameters, use CV, use CV with parameters

Use test_xgboost.m to run XGBoost on a test dataset just like the randomforest algorithm test script above


To ensure things are working you'll need Matlab & Python installed on your computer. Also, ensure that the pyversion command works in the matlab terminal.

Run the below command in a command terminal to install packages required for python (scipy, scikit, and numpy)
pip install scipy scikit-learn numpy  --user

To make sure xgboost works i had to install xgboost in python via pip on a command line (0.80 was the latest version that worked for me on windows).
pip install xgboost==0.80 --user

If you find it useful or have some changes that will help out, please let me know!

Let me know if you run into any issues.
Abhi (abhirana@gmail.com)
scikit-from-matlab 0.0.1
