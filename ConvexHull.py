import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVR
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    def __init__(self, test_path, train_path):
        self.sc = StandardScaler()
        self.pca = PCA(n_components=5)
        self.test_path = test_path
        self.train_path = train_path
        
    def load_and_preprocess(self):
        dftest0 = pd.read_csv(self.test_path)
        dftrain0 = pd.read_csv(self.train_path)
        
        # Remove Ef column and split features/target
        dftest = dftest0.drop("Ef", axis=1)
        dftrain = dftrain0.drop("Ef", axis=1)
        
        X_train = dftrain.drop("Eg", axis=1)
        X_test = dftest.drop("Eg", axis=1)
        
        # Scale features
        X_train = self.sc.fit_transform(X_train)
        X_test = self.sc.transform(X_test)
        
        y_train = dftrain['Eg'].astype('float')
        y_test = dftest['Eg'].astype('float')
        
        return X_train, X_test, y_train, y_test

class ModelTrainer:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=100, gamma='auto', epsilon=0.001))
        ])
        
    def train_and_predict(self, X_train, X_test, y_train):
        self.pipeline.fit(X_train, y_train)
        return self.pipeline.predict(X_test)

class ConvexHullAnalyzer:
    def __init__(self):
        self.error_threshold = 0.022
        
    def point_in_hull(self, point, hull):
        return all((np.dot(eq[:-1], point) + eq[-1] <= 1e-12) for eq in hull.equations)
    
    def analyze_errors(self, data_matrix, n_iterations=2000):
        costs = []
        errors = []
        current_data = data_matrix.copy()
        best_result = None
        
        for i in range(n_iterations):
            cost, error, n_points, hull_data = self._evaluate_iteration(current_data)
            costs.append(cost)
            errors.append(error)
            
            if error < self.error_threshold:
                best_result = hull_data
                break
                
            if i > 0 and cost < costs[-2] and error < errors[-2]:
                current_data = hull_data
            else:
                current_data = self._update_data(current_data, hull_data)
                
            if 0.65 * errors[0] < error < errors[0] and n_points > 50:
                if len(hull_data) < 60:
                    break
        
        return best_result, costs, errors
    
    def _evaluate_iteration(self, data):
        hull = ConvexHull(data[:, -5:])
        points_in_hull = []
        total_error = 0
        n_points = 0
        cost = 0
        
        for point in data:
            if self.point_in_hull(point[-5:], hull):
                error = point[-8]
                total_error += error
                n_points += 1
                cost += 0.0 if error <= np.percentile(data[:, -8], 5) else np.exp(3.0 * error)
                points_in_hull.append(point)
        
        cost = -n_points + cost
        error = total_error / n_points if n_points > 0 else 1.0
        
        return cost, error, n_points, np.array(points_in_hull)
    
    def _update_data(self, current_data, hull_data):
        _, test_data = train_test_split(current_data, test_size=0.1, random_state=0)
        return np.vstack((test_data[:6], hull_data))

class Visualizer:
    @staticmethod
    def plot_hull_distances(hull_data, save_path='hull_distances.svg'):
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.plot(hull_data[:, -5], hull_data[:, -8], 'bo')
        
        ax.grid(True)
        ax.set_xlabel('Convex Hull Distance', fontsize=18)
        ax.set_ylabel('Error (eV/atom)', fontsize=18)
        ax.set_xlim(-0.017, 0.05)
        
        for spine in ax.spines.values():
            spine.set_color('black')
            
        plt.savefig(save_path)
        plt.close()
    
    @staticmethod
    def plot_error_distribution(error_groups, cutoff, save_path='error_distribution.svg'):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['red', 'blue', 'green']
        labels = ['Inside Hull', 'Edge of Hull', 'Outside Hull']
        
        for errors, color, label in zip(error_groups, colors, labels):
            sns.kdeplot(errors, color=color, label=label, lw=3)
            
        plt.axvline(x=cutoff, color='b', label='error cutoff')
        ax.set_xlabel('Error (eV/atom)', fontsize=18)
        ax.set_ylabel('Frequency', fontsize=18)
        ax.set_xlim(0.0, 1.0)
        
        plt.legend(loc='upper right', fontsize='medium')
        plt.savefig(save_path)
        plt.close()

def main():
    # Initialize components
    processor = DataProcessor('testEgEf.txt', 'trainEgEf.txt')
    trainer = ModelTrainer()
    analyzer = ConvexHullAnalyzer()
    visualizer = Visualizer()
    
    # Load and process data
    X_train, X_test, y_train, y_test = processor.load_and_preprocess()
    
    # Train model and get predictions
    y_pred = trainer.train_and_predict(X_train, X_test, y_train)
    
    # Prepare data matrix for convex hull analysis
    prediction_errors = np.abs(y_pred - y_test)
    initial_data = np.hstack((
        X_test,
        prediction_errors.reshape(-1, 1),
        y_test.reshape(-1, 1),
        y_pred.reshape(-1, 1),
        processor.pca.fit_transform(X_test)
    ))
    
    # Run convex hull analysis
    best_hull_data, costs, errors = analyzer.analyze_errors(initial_data)
    
    # Visualize results
    visualizer.plot_hull_distances(best_hull_data)
    error_groups = analyzer._group_errors_by_hull_position(best_hull_data)
    visualizer.plot_error_distribution(error_groups, np.percentile(initial_data[:, -8], 5))

if __name__ == "__main__":
    main()
