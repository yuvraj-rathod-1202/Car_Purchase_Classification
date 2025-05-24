from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class bivariateDataAnalysis(ABC):
    @abstractmethod
    def analyze(self, df, feature1, feature2):
        pass

class NumericNumericAnalysis(bivariateDataAnalysis):
    def analyze(self, df, feature1, feature2):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} Vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()

class CategoricalNumericAnalysis(bivariateDataAnalysis):
    def analyze(self, df, feature1, feature2):
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()


class BivariateAnalyzer():
    def execute_strategy(self, df, feature1, feature2):
        if feature1 not in df.columns:
            print(f"'{feature1}' is not a valid column in the DataFrame.")
            return
        
        if feature2 not in df.columns:
            print(f"'{feature2}' is not a valid column in the DataFrame.")
            return
        
        dtype1 = df[feature1].dtype
        dtype2 = df[feature2].dtype

        if pd.api.types.is_numeric_dtype(dtype1) and pd.api.types.is_numeric_dtype(dtype2):
            print(f"'{feature1}' and '{feature2}' are a numerical feature.")
            NumericNumericAnalysis().analyze(df, feature1, feature2)

        if pd.api.types.is_numeric_dtype(dtype1) and (pd.api.types.is_categorical_dtype(dtype2) or pd.api.types.is_object_dtype(dtype2)):
            print(f"'{feature1} is a numerical feature and '{feature2} is a categorical feature")
            CategoricalNumericAnalysis().analyze(df, feature2, feature1)

        if pd.api.types.is_numeric_dtype(dtype2) and (pd.api.types.is_categorical_dtype(dtype1) or pd.api.types.is_object_dtype(dtype1)):
            print(f"'{feature1} is a numerical feature and '{feature2} is a categorical feature")
            CategoricalNumericAnalysis().analyze(df, feature1, feature2)

        if (pd.api.types.is_categorical_dtype(dtype1) or pd.api.types.is_object_dtype(dtype1)) and (pd.api.types.is_categorical_dtype(dtype2) or pd.api.types.is_object_dtype(dtype2)):
            print(f"'{feature1}' and '{feature2}' are a categorical feature.")

if __name__ == "__main__":
    pass