from extract_ingest.ingest import Ingestor

def pipeline():

        data = Ingestor().ingest("C:\projects\ml_learn\EToE\data_extracted\car_data.csv")

        