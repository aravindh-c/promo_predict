# importing libraires
from fastapi import FastAPI
from pydantic import BaseModel, Field

import pandas as pd
import joblib

app = FastAPI()

class Input(BaseModel):
    department: object
    region: int
    education: object
    gender: object
    recruitment_channel: object
    no_of_trainings: int
    age: int
    previous_year_rating: int
    KPIs_met : int = Field(alias="KPIs_met >80%", default=None)
    awards_won: int = Field(alias="awards_won?", default=None)
    avg_training_score: int
    length_of_service: int



class Output(BaseModel):
    is_promoted: int

@app.post("/predict")
def predict(data: Input) -> Output:
    X_input = pd.DataFrame([[data.department,data.region,data.education,data.gender,
              data.recruitment_channel,data.no_of_trainings,data.age,data.previous_year_rating,
              data.KPIs_met,data.awards_won,data.avg_training_score,data.length_of_service]])

    X_input.columns = ['department','region','education','gender','recruitment_channel',
                       'no_of_trainings','age','previous_year_rating','KPIs_met >80%','awards_won?',
                       'avg_training_score','length_of_service']

    #load model
    model = joblib.load('promotion_pipeline_model.pkl')

    #predict
    prediction = model.predict(X_input)

    #output
    return Output(is_promoted = prediction)
