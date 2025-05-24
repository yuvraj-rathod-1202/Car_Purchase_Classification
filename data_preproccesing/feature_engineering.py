from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler

class FeatureEngineeringStrategy(ABC):
    @abstractmethod
    def apply_transformation(self, df):
        pass


class StandardScalling(FeatureEngineeringStrategy):
    def __init__(self, feature):
        self._feature = feature
        self.scaler = StandardScaler()

    def apply_transformation(self, df):
        df_transform = df.copy()
        df_transform[self._feature] = self.scaler.fit_transform(df[self._feature])
        return df_transform
    
class MinMaxScalling(FeatureEngineeringStrategy):
    def __init__(self, feature):
        self._feature = feature
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def apply_transformation(self, df):
        df_transform = df.copy()
        df_transform[self._feature] = self.scaler.fit_transform(df[self._feature])
        df_transform

class OneHotEncodding(FeatureEngineeringStrategy):
    def __init__(self, feature):
        self._feature = feature
        self.encoder = OneHotEncoder(drop="first", sparse=False)

    def apply_transformation(self, df):
        df_transform = df.copy()
        encoded_df = pd.DataFrame(
            self.encoder.fit_transform(df[self._feature]),
            columns=self.encoder.get_feature_names_out(df[self._feature])
        )
        df_transformed = df_transformed.drop(columns=self.features).reset_index(drop=True)
        df_transformed = pd.concat([df_transformed, encoded_df], axis=1)
        return df_transformed
    

class FeatureEngineer():
    def execute_transformation(self, df, feature, type):
        if type=="StandardScalling":
            StandardScalling(feature).apply_transformation(df)
        elif type=="MinMaxScalling":
            MinMaxScalling(feature).apply_transformation(df)
        elif type=="OneHotEncodding":
            OneHotEncodding(feature).apply_transformation(df)