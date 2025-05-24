from extract_ingest.ingest import Ingestor
from data_preproccesing.feature_engineering import FeatureEngineer

def pipeline():

    data = Ingestor().ingest("C:\projects\ml_learn\EToE\data_extracted\car_data.csv")

    tranformed_data = FeatureEngineer().execute_transformation(data, "Gender", "OneHotEncodding")


if __name__ == "__main__":
    pipeline()