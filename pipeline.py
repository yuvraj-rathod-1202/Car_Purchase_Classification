from extract_ingest.ingest import Ingestor
from data_preproccesing.feature_engineering import FeatureEngineer
from data_preproccesing.outlier_detection import OutlierProcessing
from model_training.data_splitter import DataSplitter
from model_training.training import ModelTraining, RandomForestModelTraining

class TrainingPipeline():
    def training_pipeline(self):

        data = Ingestor().ingest("C:\projects\ml_learn\EToE\data_extracted\car_data.csv")
        data.drop(columns="User ID", inplace=True)

        tranformed_data = FeatureEngineer().execute_transformation(data, "Gender", "OneHotEncodding")

        clean_data = OutlierProcessing().process_outliers(tranformed_data, tranformed_data.columns, "IQROutlierDetection")

        X_train, X_test, y_train, y_test = DataSplitter().split(clean_data, "Purchased", "simple")
    
        model = ModelTraining(RandomForestModelTraining()).execute_model_training(X_train, y_train)

        return model, X_test, y_test


if __name__ == "__main__":
    # TrainingPipeline().training_pipeline()
    pass