# Convex-Hull
ConvexHull.py reads in training and test data, (in this case the input dataset is TCO Eg/Ef with training data designated trainEgEf.txt and test designated testEgEf.txt) train for the ML algorithm, set to SVR with 5 fold cross validation.  After the ML algorithm is completed, ML prediction errors are obtained.  The errors are randomly split; 80 % are used to construct the convex hull algorithm as described in the main text and 20 % of them are used to test the ability of the algorithm to separate well predicted from poorly predicted points.
