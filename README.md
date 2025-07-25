# 3MLET_Fase_5_Tech_Challenge_rm358953

## API Previsão de Contratação

## Como Executar o Projeto

### 1. Clone o Repositório

```bash
git clone https://github.com/marcelohek/3MLET_Fase_5_Tech_Challenge_rm358953.git
```

### 2. Copie os arquivos applicants.json, prospects.json e vagas.json para o diretório /data

### 3. Gere os artefatos do modelo
Na pasta /model:
Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```
Instale as dependências:
```bash
pip install -r requirements.txt
```
Execute os scripts:
```bash
python Pre_processamento.py
python Modelo.py
```
Serão gerados os arquivos:
```bash
feature_cols.joblib
features_preprocessed.csv
le.joblib
ohe.joblib
target.csv
vectorizer.joblib
xgb_model.joblib
```
Para realizar testes:
```bash
pytest --maxfail=1 --disable-warnings -q
```
### 4. API
Na pasta /api:
Crie um ambiente virtual:
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```
Instale as dependências:
```bash
pip install -r requirements.txt
```
Copie os arquivos da pasta /model para a pasta /api:
```bash
feature_cols.joblib
features_preprocessed.csv
le.joblib
ohe.joblib
target.csv
vectorizer.joblib
xgb_model.joblib
```
Execute a aplicação:
```bash
uvicorn api_predict:app --host 0.0.0.0 --port 8000
```
Para realizar testes:
```bash
pytest --maxfail=1 --disable-warnings -q
```
### 5. Docker
Para criar uma imagem:
```bash
docker build -t techchallengemodelo .
```
Para executar uma instância:
```bash
docker run -d --name techchallenge -p 8000:8000  techchallengemodelo
```
### 5. API em produção
Faça uma request do tipo POST para 
```bash
http://54.208.178.156:8000/predict
```
passando no Body, um JSON no formato (exemplo):
```bash
{
  "applicant": {
    "infos_basicas": {
      "nome": "Marcelo da Silva",
      "idade": 40,
      "cargo_atual": "Analista Dados",
      "data_experiencia": "2010-03-01"
    },
    "informacoes_pessoais": {
      "estado_civil": "Solteiro",
      "cidade": "São Paulo"
    },
    "formacao_e_idiomas": {
      "nivel_academico": "Mestrado",
      "nivel_ingles": "",
      "nivel_espanhol": "Intermediário"
    },
    "cv_pt": "Tenho 10 anos de experiência em ciência de dados, machine learning e análise estatística. Atuei em projetos de previsão de demanda e classificação de imagens, usando Python e frameworks como scikit-learn e TensorFlow."
  },
  "vaga": {
    "informacoes_basicas": {
      "titulo_vaga": "Cientista de Dados",
      "cliente": "Global Tech"
    },
    "perfil_vaga": {
      "nivel profissional": "Pleno",
      "areas_atuacao": "TI - Data Science",
      "nivel_ingles": "Fluente",
      "nivel_espanhol": "Básico"
    }
  },
  "prospect": {
    "codigo": "12345",
    "situacao_candidado": "Prospect",
    "data_candidatura": "2025-07-20",
    "ultima_atualizacao": "2025-07-21",
    "comentario": "Candidato interno",
    "recrutador": "Carlos Souza"
  }
}

```
