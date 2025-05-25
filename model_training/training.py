from abc import ABC, abstractmethod

from sklearn.ensemble import RandomForestClassifier

class ModelTrain(ABC):
    @abstractmethod
    def train(self, X_train, y_train):
        pass


class RandomForestModelTraining(ModelTrain):
    def __init__(self, n_estimators=10, criterion="entropy", boorstrap=True, random_state=42):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.bootstrap = boorstrap
        self.random_state = random_state
        
    def train(self, X_train, y_train):
        
        model = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, bootstrap=self.bootstrap, random_state=self.criterion)
        model.fit(X_train, y_train)
        return model
    
class ModelTraining():
    def __init__(self, model_type):
        self.model_type = model_type

    def set_model_type(self, type):
        self.model_type = type

    def execute_model_training(self, X_train, y_train):
        self.model_type.train(X_train, y_train)


if __name__ == "__main__":
    pass