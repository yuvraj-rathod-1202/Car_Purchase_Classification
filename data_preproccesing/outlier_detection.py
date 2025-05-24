from abc import ABC, abstractmethod

import pandas as pd

class OutlierDetectionStrategy(ABC):
    @abstractmethod
    def detect_outliers(self, df):
        pass

class IQROutlierDetection(OutlierDetectionStrategy):
    def detect_outliers(self, series):
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = (series < lower_bound) | (series > upper_bound)
        return outliers
    

class OutlierProcessing():
    def process_outliers(self, df, feature ,strategy):
        if strategy == "IQROutlierDetection":
            outliers = IQROutlierDetection().detect_outliers(df[feature])
            return df[~outliers]