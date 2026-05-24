import os
import sys
from dataclasses import dataclass
from src.components.model_trainer import ModelTrainer
from src.components.data_transformation import DataTransformation
from sklearn.model_selection import train_test_split
import pandas as pd

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "raw.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):

        df = pd.read_csv('notebook/data/water_potability.csv')

        os.makedirs(os.path.dirname(
            self.ingestion_config.train_data_path), exist_ok=True)

        df.to_csv(self.ingestion_config.raw_data_path,
                  index=False, header=True)

        train_set, test_set = train_test_split(
            df,
            test_size=0.2,
            random_state=42
        )

        train_set.to_csv(
            self.ingestion_config.train_data_path,
            index=False,
            header=True
        )

        test_set.to_csv(
            self.ingestion_config.test_data_path,
            index=False,
            header=True
        )

        print("Data ingestion completed")

        return (
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
        )


if __name__ == "__main__":

    ingestion_obj = DataIngestion()

    train_data, test_data = ingestion_obj.initiate_data_ingestion()

    transformation_obj = DataTransformation()

    train_arr, test_arr, _ = transformation_obj.initiate_data_transformation(
        train_data,
        test_data
    )

    model_trainer_obj = ModelTrainer()

    model_trainer_obj.initiate_model_trainer(
        train_arr,
        test_arr
    )