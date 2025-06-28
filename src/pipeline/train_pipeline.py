import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
import joblib  
from src.exception import CustomException

class TrainPipeline:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            return df
        except Exception as e:
            raise CustomException(e, sys)

    def preprocess_and_train(self):
        try:
            df = self.load_data()

           
            X = df.drop('target', axis=1)  
            y = df['target']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

           
            categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
            numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), numerical_cols),
                    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
                ]
            )

            # Create pipeline with preprocessor and model
            model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', LogisticRegression())
            ])

         
            model.fit(X_train, y_train)

     
            model_path = os.path.join('artifacts', 'model.pkl')
            os.makedirs('artifacts', exist_ok=True)
            joblib.dump(model, model_path)

            print(f"Model saved at: {model_path}")

            return model

        except Exception as e:
            raise CustomException(e, sys)
