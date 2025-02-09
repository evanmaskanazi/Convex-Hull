import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
import math


# Load and prepare data
def prepare_data():
    # Load the data
    dftest0 = pd.read_csv('testEgEf.txt')
    dftrain0 = pd.read_csv('trainEgEf.txt')

    dftest = dftest0.drop("Ef", axis=1)
    dftrain = dftrain0.drop("Ef", axis=1)

    sc = StandardScaler()

    X_train1 = dftrain.drop("Eg", axis=1)
    X_test1 = dftest.drop("Eg", axis=1)
    X_train1 = sc.fit_transform(X_train1)
    X_test1 = sc.fit_transform(X_test1)
    y_train1 = np.asarray(dftrain['Eg']).astype('float')
    y_test1 = np.asarray(dftest['Eg']).astype('float')

    # SVR Pipeline setup
    steps = [('scaler', StandardScaler()), ('SVM', SVR())]
    pipeline = Pipeline(steps)
    grid = GridSearchCV(pipeline,
                        param_grid={'SVM__C': [100],
                                    'SVM__gamma': ['auto'],
                                    'SVM__kernel': ['rbf'],
                                    'SVM__epsilon': [0.001]},
                        cv=5)

    grid.fit(X_train1, y_train1)
    y_predicted1 = grid.predict(X_test1)

    # Calculate prediction errors
    prederror = np.abs(y_predicted1 - y_test1)

    # Stack the data
    stack1 = np.vstack((np.array(X_test1).T, prederror, y_test1, y_predicted1))
    trainconvexhull10 = stack1.T

    # PCA transformation
    pca = PCA(n_components=5)
    trainconvexhull1T = trainconvexhull10.T[0:X_train1.shape[1]]
    pcafit = pca.fit(trainconvexhull1T)
    latshape00tpc = pcafit.components_

    # Final data preparation
    trainconvexhullFint = np.hstack((trainconvexhull10, np.array(latshape00tpc).T))
    trainconvexhullF = np.random.RandomState(0).permutation(trainconvexhullFint)

    return trainconvexhullF


def point_in_hull(point, hull, tolerance=1e-12):
    return all(
        (np.dot(eq[:-1], point) + eq[-1] <= tolerance)
        for eq in hull.equations)


def optimize_hull(data, n_iterations=2000):
    train_size = int(0.8 * len(data))
    trainconvexhullF = data
    trainconvexhull = trainconvexhullF[:train_size]
    trainconvexhulltest = trainconvexhullF[train_size:]

    Costlim = []
    Errlim = []
    Costprog = []
    Errprog = []
    trainconvexhull1 = trainconvexhull.copy()
    trainconvexhull2 = trainconvexhull.copy()

    for j in range(n_iterations):
        costsum = 0
        errsum = 0
        Ng = 0

        if j == 0:
            hull00 = ConvexHull(trainconvexhull[:, -5:])
            coststep = []

            for i in range(len(trainconvexhull)):
                if point_in_hull(trainconvexhull[i, -5:], hull00):
                    errsum += trainconvexhull[i, -8]
                    if trainconvexhull[i, -8] <= np.percentile(trainconvexhull[:, -8], 5):
                        coststep.append(0.0)
                    else:
                        coststep.append(math.exp(3.0 * trainconvexhull[i, -8]))
                    Ng += 1

            costsum = -Ng + np.sum(coststep)
            Costlim.append(costsum)
            Errlim.append(errsum / Ng if Ng > 0 else 1.0)

        else:
            if len(trainconvexhull1) > 25:
                if (1 / 3) * Costlim[0] < Costlim[j - 1] < 0.5 * Costlim[0]:
                    trainconvexhullint, testconvexhullint = train_test_split(trainconvexhull1, test_size=10,
                                                                             random_state=0)
                elif Costlim[j - 1] < (1 / 3) * Costlim[0]:
                    trainconvexhullint, testconvexhullint = train_test_split(trainconvexhull1, test_size=5,
                                                                             random_state=0)
                else:
                    trainconvexhullint, testconvexhullint = train_test_split(trainconvexhull1, test_size=20,
                                                                             random_state=0)
            else:
                trainconvexhullint = trainconvexhull1

            hull00 = ConvexHull(trainconvexhullint[:, -5:])
            coststep = []
            ptconv = []
            ptconverr = []

            for i in range(len(trainconvexhull)):
                if point_in_hull(trainconvexhull[i, -5:], hull00):
                    errsum += trainconvexhull[i, -8]
                    ptconv.append([trainconvexhull[i, -8], trainconvexhull[i, -7], trainconvexhull[i, -6]])
                    ptconverr.append(trainconvexhull[i, -5:])

                    if trainconvexhull[i, -8] <= np.percentile(trainconvexhull[:, -8], 5):
                        coststep.append(0.0)
                    else:
                        coststep.append(math.exp(3.0 * trainconvexhull[i, -8]))
                    Ng += 1

            costsum = -Ng + np.sum(coststep)
            Costlim.append(costsum)
            Errlim.append(errsum / Ng if Ng > 0 else 1.0)

            if Costlim[j] < Costlim[j - 1] and Errlim[j] < Errlim[j - 1]:
                trainconvexhull1 = trainconvexhullint
                Costprog.append(Costlim[j])
                Errprog.append(Errlim[j])
            else:
                if (1 / 3) * Costlim[0] < Costlim[j - 1] < 0.5 * Costlim[0]:
                    trainconvexhull1 = np.vstack((testconvexhullint[:6], trainconvexhullint))
                elif Costlim[j - 1] < (1 / 3) * Costlim[0]:
                    trainconvexhull1 = np.vstack((testconvexhullint[:3], trainconvexhullint))
                else:
                    trainconvexhull1 = np.vstack((testconvexhullint[:12], trainconvexhullint))

            if Errlim[j] < 0.022:
                trainconvexhull2 = trainconvexhullint

            if 0.65 * Errlim[0] < Errlim[j] < Errlim[0] and Ng > 50:
                if len(trainconvexhullint) < 60:
                    final_hull = ConvexHull(np.array(ptconverr))
                    return trainconvexhullint, trainconvexhulltest, final_hull, Costlim, Errlim

    return trainconvexhull2, trainconvexhulltest, hull00, Costlim, Errlim


def plot_results(train_data, test_data, hull):
    # Calculate hull distances for test points
    distances = []
    for point in test_data[:, -5:]:
        max_dist = np.max(np.dot(hull.equations[:, :-1], point) + hull.equations[:, -1])
        distances.append(max_dist)

    # First plot: Hull distances vs errors
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(distances, test_data[:, -8], 'bo')

    ax.grid(True)
    ax.tick_params(which='major', direction='in', length=4, width=2)
    ax.tick_params(which='minor', direction='in', length=4, width=2)

    ax.xaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    plt.xlim(-0.017, 0.05)
    ax.set_facecolor('w')
    for spine in ax.spines.values():
        spine.set_color('black')

    plt.xlabel('Convex Hull Distance', fontsize=18)
    plt.ylabel('Error (eV/atom)', fontsize=18)
    plt.savefig('iiPlotTCOEgDistances.svg')
    plt.close()

    # Second plot: Error distributions
    err2i = [test_data[i, -8] for i in range(len(distances))
             if distances[i] < 0.2 * min(distances)]
    err4i = [test_data[i, -8] for i in range(len(distances))
             if 0.2 * min(distances) <= distances[i] <= 0.2 * max(distances)]
    err6i = [test_data[i, -8] for i in range(len(distances))
             if distances[i] > 0.2 * max(distances)]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(data=err2i, color='red', label='Inside Hull', lw=3)
    sns.kdeplot(data=err4i, color='blue', label='Edge of Hull', lw=3)
    sns.kdeplot(data=err6i, color='green', label='Outside Hull', lw=3)

    xrline = np.percentile(train_data[:, -8], 5)
    plt.axvline(x=xrline, color='b', label='error cutoff')

    plt.title('Error Distribution', fontsize=18)
    plt.xlim(0.0, 1.0)
    ax.set_facecolor('w')
    for spine in ax.spines.values():
        spine.set_color('black')

    plt.legend(loc='upper right', shadow=True, fontsize='medium')
    plt.xlabel('Error (eV/atom)', fontsize=18)
    plt.ylabel('Frequency', fontsize=18)
    plt.savefig('iiDistributionConvexHullTCOEg.svg')
    plt.close()


# Main execution
def main():
    # Prepare data
    data = prepare_data()

    # Run optimization
    train_data, test_data, final_hull, cost_history, error_history = optimize_hull(data)

    # Generate plots
    plot_results(train_data, test_data, final_hull)


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
