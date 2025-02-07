import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import math


class MLErrorAnalyzer:
    def __init__(self, train_file, test_file):
        """Initialize with training and test data files."""
        self.train_data = pd.read_csv(train_file)
        self.test_data = pd.read_csv(test_file)
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=5)

    def prepare_data(self):
        """Prepare and split data for training."""
        # Prepare features and target
        X_train = self.train_data.drop("Eg", axis=1)
        X_test = self.test_data.drop("Eg", axis=1)
        y_train = self.train_data['Eg'].astype('float')
        y_test = self.test_data['Eg'].astype('float')

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, X_train, y_train):
        """Train SVR model with pipeline."""
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ])

        param_grid = {
            'svr__C': [100],
            'svr__gamma': ['auto'],
            'svr__kernel': ['rbf'],
            'svr__epsilon': [0.001]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        return grid_search

    def calculate_error_metrics(self, model, X_test, y_test):
        """Calculate prediction errors and related metrics."""
        y_pred = model.predict(X_test)
        errors = np.abs(y_pred - y_test)

        # Stack features, errors, and predictions
        feature_matrix = np.column_stack((
            X_test,
            errors,
            y_test,
            y_pred,
            self.pca.fit_transform(X_test)
        ))

        return feature_matrix

    def convex_hull_analysis(self, data, test_size=0.2):
        """Perform convex hull analysis on the data."""
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=0)

        # Extract PCA components for hull calculation
        pca_cols = data.shape[1] - 5  # Last 5 columns are PCA components
        hull = ConvexHull(train_data[:, -5:])

        hull_distances = []
        errors = []

        for point in test_data:
            # Calculate hull distance
            hull_eq = np.dot(hull.equations[:, :-1], point[-5:]) + hull.equations[:, -1]
            max_dist = np.max(hull_eq)

            hull_distances.append(max_dist)
            errors.append(point[pca_cols - 3])  # Error column

        return np.array(hull_distances), np.array(errors)

    def plot_error_distribution(self, hull_distances, errors, output_file):
        """Plot error distribution based on hull distances."""
        # Define distance thresholds
        min_dist = np.min(hull_distances)
        max_dist = np.max(hull_distances)
        mid_threshold = 0.2

        # Categorize errors
        inner_errors = errors[hull_distances <= min_dist * 0.2]
        edge_errors = errors[(hull_distances > min_dist * 0.2) & (hull_distances <= max_dist * 0.2)]
        outer_errors = errors[hull_distances > max_dist * 0.2]

        # Create plot
        plt.figure(figsize=(10, 6))
        sns.kdeplot(inner_errors, color='red', label='Inside Hull', lw=3)
        sns.kdeplot(edge_errors, color='blue', label='Edge of Hull', lw=3)
        sns.kdeplot(outer_errors, color='green', label='Outside Hull', lw=3)

        plt.title('Error Distribution', fontsize=18)
        plt.xlabel('Error (eV/atom)', fontsize=18)
        plt.ylabel('Frequency', fontsize=18)
        plt.xlim(0.0, 1.0)

        # Add error cutoff line
        error_cutoff = np.percentile(errors, 5)
        plt.axvline(x=error_cutoff, color='b', linestyle='--', label='Error Cutoff')

        plt.legend(loc='upper right', shadow=True)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()


def main():
    # Initialize analyzer
    analyzer = MLErrorAnalyzer('trainEgEf.txt', 'testEgEf.txt')

    # Prepare data
    X_train, X_test, y_train, y_test = analyzer.prepare_data()

    # Train model
    model = analyzer.train_model(X_train, y_train)

    # Calculate errors
    feature_matrix = analyzer.calculate_error_metrics(model, X_test, y_test)

    # Perform convex hull analysis
    hull_distances, errors = analyzer.convex_hull_analysis(feature_matrix)

    # Plot results
    analyzer.plot_error_distribution(
        hull_distances,
        errors,
        'error_distribution.svg'
    )


if __name__ == "__main__":
    main()
