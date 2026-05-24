import os
import pickle
import pandas as pd


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):

        model_path = os.path.join("artifacts", "model.pkl")
        preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)

        with open(preprocessor_path, "rb") as preprocessor_file:
            preprocessor = pickle.load(preprocessor_file)

        data_scaled = preprocessor.transform(features)

        prediction = model.predict(data_scaled)

        return prediction


class CustomData:
    def __init__(
        self,
        ph: float,
        Hardness: float,
        Solids: float,
        Chloramines: float,
        Sulfate: float,
        Conductivity: float,
        Organic_carbon: float,
        Trihalomethanes: float,
        Turbidity: float
    ):

        self.ph = ph
        self.Hardness = Hardness
        self.Solids = Solids
        self.Chloramines = Chloramines
        self.Sulfate = Sulfate
        self.Conductivity = Conductivity
        self.Organic_carbon = Organic_carbon
        self.Trihalomethanes = Trihalomethanes
        self.Turbidity = Turbidity

    def get_data_as_dataframe(self):

        custom_data_input_dict = {
            "ph": [self.ph],
            "Hardness": [self.Hardness],
            "Solids": [self.Solids],
            "Chloramines": [self.Chloramines],
            "Sulfate": [self.Sulfate],
            "Conductivity": [self.Conductivity],
            "Organic_carbon": [self.Organic_carbon],
            "Trihalomethanes": [self.Trihalomethanes],
            "Turbidity": [self.Turbidity]
        }

        return pd.DataFrame(custom_data_input_dict)