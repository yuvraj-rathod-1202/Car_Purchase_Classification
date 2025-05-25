from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

class ModelTrain(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass


class RandomForestModelTraining(ModelTrain):
    def train(self, X_train, y_train):
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 8, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True, False],
        }

        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        print("Best parameters:", grid_search.best_params_)

        return grid_search.best_estimator_
    
class ModelTraining():
    def __init__(self, model_type):
        self.model_type = model_type

    def set_model_type(self, type):
        self.model_type = type

    def execute_model_training(self, X_train, y_train):
        return self.model_type.train(X_train, y_train)


if __name__ == "__main__":
    pass