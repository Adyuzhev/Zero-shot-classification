from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

sequence_to_classify = "В современном мире за космос принимают бескрайнее пространство, начинающееся сразу после атмосферы Земли. В нем находятся планеты, звезды, галактики и другие небесные объекты. Для большего удобства космос разделяют на ближний, который можно исследовать с помощью современных спутников и аппаратов, и дальний, добраться до которого пока невозможно."

candidate_labels = ['космос', 'полет', 'Земля', 'готовить', 'спать']

print(classifier(sequence_to_classify, candidate_labels))
