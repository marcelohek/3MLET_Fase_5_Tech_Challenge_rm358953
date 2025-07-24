import json
import pytest
import numpy as np
from fastapi.testclient import TestClient
import importlib
import sys

# Fixture para criar artefatos dummy antes de importar a app
@pytest.fixture(autouse=True)
def dummy_artifacts(monkeypatch, tmp_path):
    # Dummy vectorizer: transforms list of texts to array-like
    class DummyVectorizer:
        def transform(self, texts):
            # Return a sparse-like object with multiply and sum attributes
            arr = np.ones((len(texts), 3))  # simple array
            class SparseDummy:
                def __init__(self, a): self._a = a
                def multiply(self, other): return self
                def sum(self, axis): return np.array([1])
            return SparseDummy(arr)
    
    # Dummy OneHotEncoder
    class DummyOHE:
        def get_feature_names_out(self, input_features):
            return [f"{feat}_{val}" for feat, val in zip(input_features, ['a','b','c'])]
        def transform(self, df):
            return np.array([[1, 0, 1]])

    # Dummy LabelEncoder
    class DummyLE:
        def __init__(self):
            self.classes_ = np.array(['ClienteX'])
        def transform(self, arr):
            return [1]

    # Dummy model
    class DummyModel:
        def predict_proba(self, X):
            # Return probability 0.75 for class 1
            return np.array([[0.25, 0.75]])

    # Dummy feature columns
    dummy_cols = [
        'exp_anos', 'sim_text', 'cliente_encoded',
        'senioridade_a', 'nivel_ingles_b', 'area_atuacao_c'
    ]

    seq = [DummyVectorizer(), DummyOHE(), DummyLE(), dummy_cols, DummyModel()]
    def fake_load(path):
        # Pop first dummy artifact
        return seq.pop(0)
    monkeypatch.setattr('joblib.load', fake_load)
    # Ensure module reload
    if 'api_predict' in sys.modules:
        del sys.modules['api_predict']
    return

@pytest.fixture
def client():
    import api_predict
    return TestClient(api_predict.app)


def make_payload():
    return {
        'applicant': {
            'infos_basicas': {},
            'informacoes_pessoais': {},
            'formacao_e_idiomas': {'nivel_ingles': 'Ingles'},
            'cv_pt': '5 anos de experiência'
        },
        'vaga': {
            'informacoes_basicas': {'titulo_vaga': 'Developer', 'cliente': 'ClienteX'},
            'perfil_vaga': {'nivel_profissional': 'Senior', 'areas_atuacao': 'TI'}
        },
        'prospect': {}
    }


def test_predict_success(client):
    payload = make_payload()
    response = client.post('/predict', json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 'probability' in data
    assert isinstance(data['probability'], float)
    assert abs(data['probability'] - 0.75) < 1e-6


def test_predict_invalid_payload(client):
    # Missing applicant key
    response = client.post('/predict', json={'vaga': {}, 'prospect': {}})
    assert response.status_code == 422
