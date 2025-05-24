from extract_ingest.ingest import Ingestor
from data_preproccesing.feature_engineering import FeatureEngineer
from data_preproccesing.outlier_detection import OutlierProcessing

def pipeline():

    data = Ingestor().ingest("C:\projects\ml_learn\EToE\data_extracted\car_data.csv")
    data.drop(columns="User ID", inplace=True)

    tranformed_data = FeatureEngineer().execute_transformation(data, "Gender", "OneHotEncodding")

    clean_data = OutlierProcessing().process_outliers(tranformed_data, tranformed_data.columns, "IQROutlierDetection")

    print(clean_data)


if __name__ == "__main__":
    pipeline()