from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import pandas as pd
import re
import joblib
import numpy as np

# Carregar artefatos de pré-processamento e modelo
vectorizer = joblib.load('vectorizer.joblib')
ohe = joblib.load('ohe.joblib')
le = joblib.load('le.joblib')
feature_cols = joblib.load('feature_cols.joblib')
model = joblib.load('xgb_model.joblib')

app = FastAPI(
    title="API de Previsão de Candidatos",
    description="Recebe dados brutos de applicant, vaga e prospect e retorna probabilidade de sucesso",
    version="2.0"
)

class RawPredictRequest(BaseModel):
    applicant: Dict[str, Any]
    vaga: Dict[str, Any]
    prospect: Dict[str, Any]

class PredictResponse(BaseModel):
    probability: float

# Funções de pré-processamento
def normalize_text(text: str) -> str:
    txt = str(text).lower()
    txt = re.sub(r"[\u0300-\u036f]", "", txt)
    txt = re.sub(r"[^a-z0-9 ]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()

def extract_exp(text: str) -> float:
    m = re.search(r"(\d+) anos?", str(text))
    return float(m.group(1)) if m else 0.0

@app.post("/predict", response_model=PredictResponse)
async def predict(request: RawPredictRequest):
    """
    Recebe a estrutura bruta de applicant, vaga e prospect, faz o pré-processamento internamente
    e retorna a probabilidade de sucesso do candidato para aquela vaga.
    """
    try:
        # Flatten applicant
        appl = request.applicant
        base = appl.get("infos_basicas", {})
        perso = appl.get("informacoes_pessoais", {})
        form = appl.get("formacao_e_idiomas", {})
        cv_text = appl.get("cv_pt", "")
        nivel_ingles = form.get("nivel_ingles", "Desconhecido")

        # Flatten vaga
        vag = request.vaga
        info = vag.get("informacoes_basicas", {})
        perfil = vag.get("perfil_vaga", {})
        titulo_vaga = info.get("titulo_vaga", "")
        cliente = info.get("cliente", "")
        senioridade = perfil.get("nivel profissional") or perfil.get("nivel_profissional") or "Desconhecido"
        area_vaga = perfil.get("areas_atuacao") or perfil.get("area_atuacao") or "Desconhecido"

        # Features numéricas
        exp_anos = extract_exp(cv_text)

        # Similaridade textual
        cv_clean = normalize_text(cv_text)
        titulo_clean = normalize_text(titulo_vaga)
        tf_cv = vectorizer.transform([cv_clean])
        tf_ti = vectorizer.transform([titulo_clean])
        sim_text = float(np.array(tf_cv.multiply(tf_ti).sum(axis=1)).ravel()[0])

        # Categóricas via OneHotEncoder
        cat_df = pd.DataFrame([{
            "senioridade": senioridade,
            "nivel_ingles": nivel_ingles,
            "area_atuacao": area_vaga
        }])
        ohe_feats = ohe.transform(cat_df)
        ohe_dict = dict(zip(
            ohe.get_feature_names_out(["senioridade", "nivel_ingles", "area_atuacao"]),
            ohe_feats[0]
        ))

        # Cliente encoding
        cliente_encoded = int(le.transform([cliente])[0]) if cliente in le.classes_ else 0

        # Montar vetor final na ordem esperada
        row = {
            "exp_anos": exp_anos,
            "sim_text": sim_text,
            "cliente_encoded": cliente_encoded,
            **ohe_dict
        }
        X = pd.DataFrame([[row.get(col, 0) for col in feature_cols]], columns=feature_cols)

        # Predição
        proba = float(model.predict_proba(X)[0][1])
        return PredictResponse(probability=proba)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na predição: {e}")
