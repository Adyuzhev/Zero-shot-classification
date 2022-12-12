from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    sequence_to_classify: str
    candidate_labels: list

app = FastAPI ()
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


@app.get("/")
def root():
    return {"message": "Основная страница"}

@app.post("/predict/")
def predict(item: Item):
    result = classifier(item.sequence_to_classify, item.candidate_labels)
    return result['labels'], result['scores']



# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# sequence_to_classify = "Я хочу поехать в Австралию"

# candidate_labels = ["спорт", "путешествия", "музыка", "кино", "книги", "наука"]

# result = classifier(sequence_to_classify, candidate_labels)

# print(result['labels'], result['scores'])
