import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from scipy.spatial.distance import cdist
from sklearn.preprocessing import scale
from sklearn import datasets
import matplotlib.cm as cm
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.svm import SVR
import seaborn as sns;

sns.set()
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
import re
import gzip
import gensim
import logging
from gensim.models import Word2Vec
from gensim.test.utils import common_texts, get_tmpfile
from csv import reader
from math import sqrt
from sklearn.preprocessing import StandardScaler

# setup word2vec routines for extracting numerical data from text
# for classification of medical descriptions
path = get_tmpfile("word2vec.model")
# import nltk
# nltk.download('punkt')
# from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action='ignore')
import gensim
from gensim.models import Word2Vec
from sklearn.metrics import mean_absolute_error
from numpy.random import seed
from numpy.random import randn
from numpy import percentile
from sklearn.tree import export_graphviz
import pydot
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn import tree
from scipy.stats import uniform
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree=3, include_bias=False)
import scipy
import matplotlib.pyplot as pyplot
from sklearn import datasets, linear_model

from sklearn.linear_model import LinearRegression

modellinear = LinearRegression()
from sklearn.ensemble import ExtraTreesRegressor

import numpy as np
import pandas as pd
from time import time
import scipy.stats as stats
from sklearn.utils.fixes import loguniform
import random
import math

from hpsklearn import HyperoptEstimator, any_classifier
from hyperopt import tpe
import numpy as np
from hpsklearn import random_forest, svc, knn
from hyperopt import hp
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from scipy import spatial

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import xgboost
from xgboost import XGBRegressor

xboostr = XGBRegressor()
from sklearn.model_selection import RandomizedSearchCV

param_distxg = {'n_estimators': stats.randint(50, 150),
                'learning_rate': stats.uniform(0.01, 0.6),
                'subsample': stats.uniform(0.3, 0.9),
                'max_depth': [3, 4, 5, 6, 7, 8, 9],
                'colsample_bytree': stats.uniform(0.5, 0.9),
                'min_child_weight': [1, 2, 3, 4],
                'seed': [2]
                }

dftest0 = pd.read_csv(r'testEgEf.txt')
dftrain0 = pd.read_csv(r'trainEgEf.txt')
dftest = dftest0.drop("Eg", 1)
dftrain = dftrain0.drop("Eg", 1)

# Model 3 - Support Vector Regression (SVR)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

tsne = TSNE(n_components=4, learning_rate='auto', init='random')
ldac = LinearDiscriminantAnalysis(n_components=5)
pca = PCA(n_components=5)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


X_train1 = dftrain.drop("Ef", 1)
X_test1 = dftest.drop("Ef", 1)
X_train1=sc.fit_transform(X_train1)
X_test1=sc.fit_transform(X_test1)
y_train1 = np.asarray(dftrain['Ef']).astype('float')
y_test1 = np.asarray(dftest['Ef']).astype('float')

steps = [('scaler', StandardScaler()), ('SVM', SVR())]
pipeline = Pipeline(steps)
grid = GridSearchCV(pipeline, param_grid={'SVM__C': [100], 'SVM__gamma': ['auto'], 'SVM__kernel': ['rbf'],
                                          'SVM__epsilon': [0.001]}, cv=5)

grid.fit(X_train1, y_train1)
svr_score = grid.score(X_train1, y_train1)
svr_score1 = grid.score(X_test1, y_test1)
y_predicted1 = grid.predict(X_test1)
prederror = np.zeros(len(y_predicted1))
for i in range(len(y_predicted1)):
    prederror[i] = abs(y_predicted1[i] - y_test1[i])

stack1 = np.array(np.vstack((np.array(X_test1).T, prederror, y_test1, y_predicted1)))

trainconvexhull10 = stack1.T

trainint = np.array(trainconvexhull10.T[0:int(np.array(X_train1).shape[1])].T)

trainout = np.array(trainconvexhull10).T[int(np.array(X_train1).shape[1])]

from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

tsne = TSNE(n_components=4, learning_rate='auto', init='random')
ldac = LinearDiscriminantAnalysis(n_components=5)
pca = PCA(n_components=5)

trainconvexhull1T = np.array(trainconvexhull10.T[0:(np.array(X_train1).shape[1])])

from sklearn.utils import shuffle

pcafit = pca.fit(trainconvexhull1T)
latshape00tpc = pcafit.components_
trainconvexhullFint = np.hstack((np.array(trainconvexhull10), np.array(latshape00tpc).T))
trainconvexhullF = shuffle(np.array(trainconvexhullFint), random_state=0)

trainconvexhull = trainconvexhullF[0:int(0.8 * np.array(trainconvexhullF).shape[0])]
trainconvexhulltest = trainconvexhullF[
                      int(0.8 * np.array(trainconvexhullF).shape[0]):np.array(trainconvexhullF).shape[0]]

import matplotlib.path as mplPath
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from scipy.optimize import linprog


def lin_in_hull(x, points):
    n_points = len(points)
    n_dim = len(x)
    c = np.zeros(n_points)
    A = np.r_[points.T, np.ones((1, n_points))]
    b = np.r_[x, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b)
    return lp.success


def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


y_train1int = y_train1.astype('int')
trainconvexhull1 = trainconvexhull
trainconvexhull2 = trainconvexhull
Costlim = []
Errlim = []
Costprog = []
Errprog = []
print(np.array(trainconvexhull).shape)

for j in range(2000):
    if (j == 0):
        print(j)
        costsum = 0
        errsum = 0
        Ng = 0

        hull00 = ConvexHull(trainconvexhull[:,
                            [int(np.array(trainconvexhull).shape[1] - 5), int(np.array(trainconvexhull).shape[1] - 4),
                             int(np.array(trainconvexhull).shape[1] - 3), int(np.array(trainconvexhull).shape[1] - 2),
                             int(np.array(trainconvexhull).shape[1] - 1)]])

        coststep = []
        for i in range(np.array(trainconvexhull).shape[0]):

            if (point_in_hull(trainconvexhull[:,
                              [int(np.array(trainconvexhull).shape[1] - 5), int(np.array(trainconvexhull).shape[1] - 4),
                               int(np.array(trainconvexhull).shape[1] - 3), int(np.array(trainconvexhull).shape[1] - 2),
                               int(np.array(trainconvexhull).shape[1] - 1)]][i], hull00) == True):

                errsum = errsum + trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 8)]
                if (trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 8)] <= np.percentile(
                        np.array(trainconvexhull).T[np.array(trainconvexhull).shape[1] - 8], 5)):
                    coststep = np.append(coststep, 0.0)
                    Ng = Ng + 1

                elif (trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 8)] > np.percentile(
                        np.array(trainconvexhull).T[np.array(trainconvexhull).shape[1] - 8], 5)):
                    Ng = Ng + 1
                    coststep = np.append(coststep, (
                        math.exp(3.0 * trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 8)])))

        costsum = -Ng + np.sum(np.array(coststep))
        Costlim = np.append(Costlim, costsum)
        Errlim = np.append(Errlim, errsum / Ng)
    else:
        print(Ng)
        costsum = 0
        errsum = 0
        Ng = 0
        if (np.array(trainconvexhull1).shape[0] > 25):
            if (((1 / 3) * Costlim[0] < Costlim[j - 1] < 0.5 * Costlim[0])):

                trainconvexhullint, testconvexhullint = train_test_split(np.array(trainconvexhull1), test_size=10,
                                                                         random_state=0)
            elif ((Costlim[j - 1] < (1 / 3) * Costlim[0])):
                trainconvexhullint, testconvexhullint = train_test_split(np.array(trainconvexhull1), test_size=5,
                                                                         random_state=0)
            else:
                trainconvexhullint, testconvexhullint = train_test_split(np.array(trainconvexhull1), test_size=20,
                                                                         random_state=0)
        else:
            trainconvexhullint = trainconvexhull1
        print(np.array(trainconvexhullint).shape[0])

        hull00 = ConvexHull(trainconvexhullint[:,
                            [int(np.array(trainconvexhull).shape[1] - 5), int(np.array(trainconvexhull).shape[1] - 4),
                             int(np.array(trainconvexhull).shape[1] - 3), int(np.array(trainconvexhull).shape[1] - 2),
                             int(np.array(trainconvexhull).shape[1] - 1)]])
        #    hull00 = ConvexHull(SelectKBest(chi2, k=5).fit_transform(trainconvexhullint, testconvexhullint))
        coststep = []
        ptconv = np.empty((0, 3), float)
        ptconverr = np.empty((0, 5), float)

        for i in range(np.array(trainconvexhull).shape[0]):

            if (point_in_hull(trainconvexhull[:,
                              [int(np.array(trainconvexhull).shape[1] - 5), int(np.array(trainconvexhull).shape[1] - 4),
                               int(np.array(trainconvexhull).shape[1] - 3), int(np.array(trainconvexhull).shape[1] - 2),
                               int(np.array(trainconvexhull).shape[1] - 1)]][i], hull00) == True):

                errsum = errsum + trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 8)]
                ptconv = np.append(ptconv, np.array([[trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 8)],
                                                      trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 7)],
                                                      trainconvexhull[i][
                                                          int(np.array(trainconvexhull).shape[1] - 6)]]]), axis=0)

                ptconverr = np.append(ptconverr,
                                      np.array([[trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 5)],
                                                 trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 4)],
                                                 trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 3)],
                                                 trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 2)],
                                                 trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 1)]]]),
                                      axis=0)

                if (trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 8)] <= np.percentile(
                        np.array(trainconvexhull).T[np.array(trainconvexhull).shape[1] - 8], 5)):
                    coststep = np.append(coststep, 0.0)
                    Ng = Ng + 1

                elif (trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 8)] > np.percentile(
                        np.array(trainconvexhull).T[np.array(trainconvexhull).shape[1] - 8], 5)):
                    Ng = Ng + 1

                    coststep = np.append(coststep, (
                        math.exp(3.0 * trainconvexhull[i][int(np.array(trainconvexhull).shape[1] - 8)])))

        costsum = -Ng + np.sum(np.array(coststep))
        Costlim = np.append(Costlim, costsum)

        if (Ng != 0):
            Errlim = np.append(Errlim, errsum / Ng)
        else:
            Errlim = np.append(Errlim, 1.0)

        print(Costlim[j], 'Costlim')
        print(Errlim[j], 'Errlim')
        if ((Costlim[j] < Costlim[j - 1])):
            print('Cost Drop')
        else:
            print('no Cost Drop')
        if ((Costlim[j] < Costlim[j - 1]) and (Errlim[j] < Errlim[j - 1])):
            trainconvexhull1 = trainconvexhullint
            Costprog = np.append(Costprog, Costlim[j])
            Errprog = np.append(Errprog, Errlim[j])

        else:
            if (((1 / 3) * Costlim[0] < Costlim[j - 1] < 0.5 * Costlim[0])):
                trainconvexhull1 = np.vstack((testconvexhullint[0:6, :], trainconvexhullint))
            elif ((Costlim[j - 1] < (1 / 3) * Costlim[0])):
                trainconvexhull1 = np.vstack((testconvexhullint[0:3, :], trainconvexhullint))
            else:
                trainconvexhull1 = np.vstack((testconvexhullint[0:12, :], trainconvexhullint))

        if (Errlim[j] < 0.022):
            trainconvexhull2 = trainconvexhullint
            print(Errlim[j], 'low err')

        if (0.55 * Errlim[0] < Errlim[j] < 0.85 * Errlim[0] and Ng > 50):

            if (np.array(trainconvexhullint).shape[0] < 40):
                break

            trainhull = ptconverr
            testinput = trainconvexhullF[:,
                        [int(np.array(trainconvexhullF).shape[1] - 5), int(np.array(trainconvexhullF).shape[1] - 4),
                         int(np.array(trainconvexhullF).shape[1] - 3), int(np.array(trainconvexhullF).shape[1] - 2),
                         int(np.array(trainconvexhullF).shape[1] - 1)]]
            testhull = testinput[int(0.8 * np.array(trainconvexhullF).shape[0]):np.array(trainconvexhullF).shape[0]]


            hullt = ConvexHull(trainhull)


            def minnorm(arr, point):
                n = np.zeros((len(arr)))
                for i in range(len(n)):
                    n[i] = np.linalg.norm(arr[i] - point)
                return min(n)


            ic = 0
            hullvec = np.empty((0, 4), float)
            for i in range(len(testhull)):
                hullvec = np.append(hullvec, np.array(
                    [[trainconvexhulltest[i][int(np.array(trainconvexhulltest).shape[1] - 8)],
                      np.max(np.dot(hullt.equations[:, :-1], testhull[i].T).T + hullt.equations[:, -1], axis=-1),
                      trainconvexhulltest[i][int(np.array(trainconvexhulltest).shape[1] - 7)],
                      trainconvexhulltest[i][int(np.array(trainconvexhulltest).shape[1] - 6)]]]),
                                    axis=0)

                if (point_in_hull(testhull[i], hullt) == True):
                    ic = ic + 1
            print(ic, 'ConvexHull')
            print(np.min(hullvec.T[1]), np.max(hullvec.T[1]), 'Hull width')

            fig = plt.figure()
            ax2 = fig.add_subplot(1, 1, 1)
            plt.plot(hullvec.T[1], hullvec.T[0], 'bo', label=r'')

            ax2.grid()
            ax2.tick_params(which='major', bottom=True, top=False, left=True,
                            right=False, width=2, direction='in')
            ax2.tick_params(which='major', labelbottom=True, labeltop=False,
                            labelleft=True, labelright=False, width=2, direction='in')
            ax2.tick_params(axis=u'both', which='major', length=4)
            ax2.xaxis.set_major_locator(MultipleLocator(0.01))
            ax2.yaxis.set_major_locator(MultipleLocator(0.1))
            ax2.tick_params(which='minor', bottom=True, top=False, left=True,
                            right=False, width=2, direction='in')
            ax2.tick_params(which='minor', labelbottom=True, labeltop=False,
                            labelleft=True, labelright=False, width=2, direction='in')
            ax2.tick_params(axis=u'both', which='minor', length=4)
            ax2.xaxis.set_minor_locator(MultipleLocator(0.01))
            ax2.yaxis.set_minor_locator(MultipleLocator(0.1))
            ax2.set_facecolor('w')
            plt.xlim(-0.017, 0.05)
            ax2.spines['bottom'].set_color('black')
            ax2.spines['top'].set_color('black')
            ax2.spines['right'].set_color('black')
            ax2.spines['left'].set_color('black')
            plt.xlabel('Convex Hull Distance', fontsize=18)
            plt.ylabel('Error (eV/atom)', fontsize=18)
            plt.savefig(r'PlotTCOEfDIstances.svg')


            xrline = np.percentile(np.array(trainconvexhull).T[np.array(trainconvexhull).shape[1] - 8], 5)
            err2i = []
            for i in range(len(hullvec.T[1])):
                if (np.min(hullvec.T[1]) <= hullvec.T[1][i] < 0.2 * np.min(hullvec.T[1])):
                    err2i = np.append(err2i, hullvec.T[0][i])
                    print(i, '2')

            err4i = []
            for i in range(len(hullvec.T[1])):
                if (0.2 * np.min(hullvec.T[1]) <= hullvec.T[1][i] <= 0.2 * np.max(hullvec.T[1])):
                    err4i = np.append(err4i, hullvec.T[0][i])
                    print(i, '4')
            err6i = []
            for i in range(len(hullvec.T[1])):
                if (0.2 * np.max(hullvec.T[1]) < hullvec.T[1][i] <= np.max(hullvec.T[1])):
                    err6i = np.append(err6i, hullvec.T[0][i])
                    print(hullvec.T[1][i], np.max(hullvec.T[1]), len(hullvec.T[1]), '6')

            import seaborn as sb

            fig = plt.figure()
            ax2 = fig.add_subplot(1, 1, 1)
            sns.kdeplot(err2i, color='red', common_norm="true",
                        label='Inside Hull', lw=3)
            sns.kdeplot(err4i, color='blue', common_norm="true",
                        label='Edge of Hull', lw=3)
            sns.kdeplot(err6i, color='green', common_norm="true",
                        label='Outside Hull', lw=3)
            plt.title('Error Distribution', fontsize=18)
            plt.xlim(0.0, 0.3)
            plt.axvline(x=xrline, color='b', label='error cutoff')
            ax2.set_facecolor('w')
            ax2.spines['bottom'].set_color('black')
            ax2.spines['top'].set_color('black')
            ax2.spines['right'].set_color('black')
            ax2.spines['left'].set_color('black')
            legend18 = plt.legend(loc='upper right', shadow=True, fontsize='medium')
            legend18.get_frame().set_facecolor('w')
            plt.xlabel('Error (eV/atom)', fontsize=18)
            plt.ylabel('Frequency', fontsize=18)
            plt.savefig(r'DistributionConvexHullTCOEf.svg')
