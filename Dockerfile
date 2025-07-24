FROM python:3.10-slim
WORKDIR /app

# Instala dependências da API
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copia a pasta da API
COPY api ./api

# Copia os artefatos de dados do model
COPY model/feature_cols.joblib ./model/feature_cols.joblib
COPY model/le.joblib         ./model/le.joblib
COPY model/ohe.joblib       ./model/ohe.joblib
COPY model/vectorizer.joblib ./model/vectorizer.joblib
COPY model/xgb_model.joblib ./model/xgb_model.joblib
COPY model/features_preprocessed.csv ./model/features_preprocessed.csv
COPY model/target.csv ./model/target.csv

# Define variável de ambiente para Python encontrar pacotes
ENV PYTHONPATH="/app/api:/app/model"

# Expõe porta da API
EXPOSE 8000

# Comando padrão para iniciar a API
CMD ["uvicorn", "api.api_predict:app", "--host", "0.0.0.0", "--port", "8000"]