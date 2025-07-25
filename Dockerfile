FROM python:3.10-slim
WORKDIR /app/api

# Instala dependências da API
COPY api/requirements.txt ./api/requirements.txt
RUN pip install --no-cache-dir -r api/requirements.txt

# Copia a pasta da API
COPY api ./api

# Copia os artefatos de dados do model
COPY model/feature_cols.joblib /app/model/feature_cols.joblib
COPY model/le.joblib         /app/model/le.joblib
COPY model/ohe.joblib       /app/model/ohe.joblib
COPY model/vectorizer.joblib /app/model/vectorizer.joblib
COPY model/xgb_model.joblib /app/model/xgb_model.joblib
COPY model/features_preprocessed.csv /app/model/features_preprocessed.csv
COPY model/target.csv /app/model/target.csv

# Define variável de ambiente para Python encontrar pacotes
ENV PYTHONPATH="/app"

# Expõe porta da API
EXPOSE 8000

# Comando padrão para iniciar a API
CMD ["uvicorn", "api.api_predict:app", "--host", "0.0.0.0", "--port", "8000"]