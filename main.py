from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    sequence_to_classify: str
    candidate_labels: list


app = FastAPI()
classifier = pipeline("zero-shot-classification",
                      model="facebook/bart-large-mnli")


@app.get("/")
def root():
    return {"message": "Основная страница"}


@app.post("/predict_bestLable/")
def predict_bestLable(item: Item):
    """
    Возвращает одну метку с лучшим счетом

    Параметры:
                    item.sequence_to_classify (str): предложение - контекст
                    item.candidate_labels (list): набор меток

    Возвращаемое значение:
                    (dict): метка(str): счет(float)
    """
    result = classifier(item.sequence_to_classify, item.candidate_labels)
    return ({result["labels"][0]: result["scores"][0]})


@app.post("/predict_multiLable/")
def predict_multiLable(item: Item):
    """
    Возвращает все метки и их счет

    Параметры:
                    item.sequence_to_classify (str): предложение - контекст
                    item.candidate_labels (list): набор меток

    Возвращаемое значение:
                    (dict): метка(str): счет(float)
    """
    result = classifier(item.sequence_to_classify,
                        item.candidate_labels, multi_label=True)
    return ({result["labels"][i]: result["scores"][i] for i in range(len(result["labels"]))})
