from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Основная страница"}


def test_predict_bestLable_eu():
    response = client.post("/predict_bestLable/",
                           json={"sequence_to_classify": "A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world. Konstantin Batygin did not set out to solve one of the solar system’s most puzzling mysteries when he went for a run up a hill in Nice, France. Dr. Batygin, a Caltech researcher, best known for his contributions to the search for the solar system’s missing “Planet Nine,” spotted a beer bottle. At a steep, 20 degree grade, he wondered why it wasn’t rolling down the hill. He realized there was a breeze at his back holding the bottle in place. Then he had a thought that would only pop into the mind of a theoretical astrophysicist: “Oh! This is how Europa formed.” Europa is one of Jupiter’s four large Galilean moons. And in a paper published Monday in the Astrophysical Journal, Dr. Batygin and a co-author, Alessandro Morbidelli, a planetary scientist at the Côte d’Azur Observatory in France, present a theory explaining how some moons form around gas giants like Jupiter and Saturn, suggesting that millimeter-sized grains of hail produced during the solar system’s formation became trapped around these massive worlds, taking shape one at a time into the potentially habitable moons we know today.",
                                 "candidate_labels": ["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"]}
                           )
    json_data = response.json()

    assert response.status_code == 200
    assert list(key for key in json_data.keys())[0] == "scientific discovery"
    assert list(key for key in json_data.values())[0] >= 0.5


def test_predict_multiLable_eu():
    response = client.post("/predict_multiLable/",
                           json={"sequence_to_classify": "A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world. Konstantin Batygin did not set out to solve one of the solar system’s most puzzling mysteries when he went for a run up a hill in Nice, France. Dr. Batygin, a Caltech researcher, best known for his contributions to the search for the solar system’s missing “Planet Nine,” spotted a beer bottle. At a steep, 20 degree grade, he wondered why it wasn’t rolling down the hill. He realized there was a breeze at his back holding the bottle in place. Then he had a thought that would only pop into the mind of a theoretical astrophysicist: “Oh! This is how Europa formed.” Europa is one of Jupiter’s four large Galilean moons. And in a paper published Monday in the Astrophysical Journal, Dr. Batygin and a co-author, Alessandro Morbidelli, a planetary scientist at the Côte d’Azur Observatory in France, present a theory explaining how some moons form around gas giants like Jupiter and Saturn, suggesting that millimeter-sized grains of hail produced during the solar system’s formation became trapped around these massive worlds, taking shape one at a time into the potentially habitable moons we know today.",
                                 "candidate_labels": ["space & cosmos", "scientific discovery", "microbiology", "robots", "archeology"]}
                           )
    json_data = response.json()

    assert response.status_code == 200
    assert len(json_data.keys()) == 5


def test_predict_bestLable_ru():
    response = client.post("/predict_bestLable/",
                           json={"sequence_to_classify": "Во многих странах Евразии лето жаркое, температуры изнуряющие, а солнце жалит и обжигает. Не всем приходится по душе такая погода, многие чувствуют себя плохо, получают солнечные удары и ожоги, боятся – и вполне оправдано – выходить из дома после полудня, ведь это не просто неприятно, но и опасно для здоровья. Осень же радует комфортной температурой, приближающейся прохладой, щадящим солнцем. Оно светит ярко, но не изнуряет – одно удовольствие прогуливаться по парку ясным сентябрьским утром.",
                                 "candidate_labels": ["время года", "насекомые", "погода", "автомобили", "путешествие", "стихи"]}
                           )
    json_data = response.json()

    assert response.status_code == 200
    assert list(key for key in json_data.keys())[0] == "погода"
    assert list(key for key in json_data.values())[0] >= 0.5


def test_predict_multiLable_ru():
    response = client.post("/predict_multiLable/",
                           json={"sequence_to_classify": "Во многих странах Евразии лето жаркое, температуры изнуряющие, а солнце жалит и обжигает. Не всем приходится по душе такая погода, многие чувствуют себя плохо, получают солнечные удары и ожоги, боятся – и вполне оправдано – выходить из дома после полудня, ведь это не просто неприятно, но и опасно для здоровья. Осень же радует комфортной температурой, приближающейся прохладой, щадящим солнцем. Оно светит ярко, но не изнуряет – одно удовольствие прогуливаться по парку ясным сентябрьским утром.",
                                 "candidate_labels": ["время года", "насекомые", "погода", "автомобили", "путешествие", "стихи"]}
                           )
    json_data = response.json()

    assert response.status_code == 200
    assert len(json_data.keys()) == 6
