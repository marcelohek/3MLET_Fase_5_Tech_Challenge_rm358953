# %% [markdown]
# # 1. Importações

# %%
import pandas as pd
import numpy as np
import json
import re
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from datetime import datetime

# %% [markdown]
# # 2. Função de carregamento flexível de JSON

# %%
def load_json_flexible(path):
    """Tenta carregar JSON padrão; em caso de erro, aplica correções e usa json5."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON padrão falhou: {e}\nAplicando correções e tentando json5...")
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        text = re.sub(r",\s*([\]}])", r"\1", text)
        try:
            import json5
        except ImportError as err:
            raise ImportError(
                "Para usar load_json_flexible com JSON5, instale primeiro o pacote:\n"
                "    pip install json5"
            ) from err
        return json5.loads(text)

# %% [markdown]
# # 3. Carregamento e flattening de JSONs

# %%
app_json_raw      = load_json_flexible('../data/applicants.json')
vagas_json_raw    = load_json_flexible('../data/vagas.json')
prospects_json_raw = load_json_flexible('../data/prospects.json')

# %% [markdown]
# # 3.1 Flatten Applicants

# %%
applicants_records = []
for code_str, data in app_json_raw.items():
    base   = data.get('infos_basicas', {})
    perso  = data.get('informacoes_pessoais', {})
    form   = data.get('formacao_e_idiomas', {})
    applicants_records.append({
        'codigo_profissional': base.get('codigo_profissional', code_str),
        'nome': base.get('nome'),
        'email': base.get('email'),
        'nivel_academico': form.get('nivel_academico'),
        'nivel_ingles': form.get('nivel_ingles'),
        'area_atuacao': perso.get('area_atuacao'),
        'cv_pt': data.get('cv_pt', '')
    })
applicants = pd.DataFrame(applicants_records)
print(f"Applicants shape: {applicants.shape}")

# %% [markdown]
# # 3.2 Flatten Vagas

# %%
vagas_records = []
for id_str, data in vagas_json_raw.items():
    info   = data.get('informacoes_basicas', {})
    perfil = data.get('perfil_vaga', {})
    vagas_records.append({
        'id_vaga': int(id_str),
        'titulo_vaga': info.get('titulo_vaga'),
        'cliente': info.get('cliente'),
        'prazo_contratacao': info.get('prazo_contratacao'),
        'tipo_contratacao': info.get('tipo_contratacao'),
        'senioridade': perfil.get('nivel profissional') or perfil.get('nivel_profissional'),
        'area_atuacao': perfil.get('areas_atuacao') or perfil.get('area_atuacao')
    })
vagas = pd.DataFrame(vagas_records)
print(f"Vagas shape: {vagas.shape}")

# %% [markdown]
# # 3.3 Flatten Prospects

# %%
prospects_records = []
for id_str, data in prospects_json_raw.items():
    for p in data.get('prospects', []):
        prospects_records.append({
            'id_vaga': int(id_str),
            'codigo_profissional': p.get('codigo'),
            'nome_candidato': p.get('nome'),
            'situacao_candidato': p.get('situacao_candidado'),
            'data_candidatura': p.get('data_candidatura'),
            'data_atualizacao': p.get('ultima_atualizacao'),
            'comentario': p.get('comentario'),
            'recrutador': p.get('recrutador')
        })
prospects = pd.DataFrame(prospects_records)
print(f"Prospects shape: {prospects.shape}")

# %% [markdown]
# # 4. Integração dos dados

# %%
# Unir prospects com applicants e vagas
df = prospects.merge(applicants, on='codigo_profissional', how='left')
df = df.merge(vagas, on='id_vaga', how='left')
print(f"Dataset integrado: {df.shape}")

# %% [markdown]
# # 5. Pré-processamento

# %% [markdown]
# ## 5.1 Limpeza de textos e datas
# 

# %%
# Normalize texto removendo acentos e caracteres especiais
def normalize_text(text):
    if pd.isna(text): return ''
    txt = str(text).lower()
    txt = re.sub(r"[\u0300-\u036f]", '', txt)
    txt = re.sub(r"[^a-z0-9 ]", ' ', txt)
    return re.sub(r"\s+", ' ', txt).strip()

text_cols = ['cv_pt', 'titulo_vaga']
for col in text_cols:
    df[col + '_clean'] = df[col].apply(normalize_text)

# Converter datas
date_cols = ['data_candidatura', 'data_atualizacao', 'prazo_contratacao']
for col in date_cols:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

# Criar target binário (1=Contratado)
sucesso = df['situacao_candidato'].str.contains('Contratado', case=False, na=False)
df['target'] = sucesso.astype(int)

# %% [markdown]
# ## 5.2 Tratamento de valores ausentes

# %%
# Categorias e datas
df['nivel_ingles']    = df['nivel_ingles'].fillna('Desconhecido')
df['nivel_academico'] = df['nivel_academico'].fillna('Desconhecido')
for col in ['data_candidatura', 'data_atualizacao']:
    df[col] = df[col].fillna(df[col].median())

# %% [markdown]
# # 6. Análise Exploratória de Dados (EDA)

# %%
print("Distribuição target:\n", df['target'].value_counts(normalize=True))
print("Senioridade:\n", df['senioridade'].value_counts())
print("Inglês vs Sucesso:\n", pd.crosstab(df['nivel_ingles'], df['target'], normalize='index'))
numeric_cols = df.select_dtypes(include=['int64','float64']).columns
print("Correlação numérica:\n", df[numeric_cols].corr()['target'].sort_values(ascending=False))

# %% [markdown]
# # 7. Feature Engineering

# %% [markdown]
# # 7.1 Extrair experiência em anos do CV

# %%
def extract_exp(text):
    m = re.search(r"(\d+) anos?", str(text))
    return int(m.group(1)) if m else np.nan

df['exp_anos'] = df['cv_pt_clean'].apply(extract_exp)
df['exp_anos'] = df['exp_anos'].fillna(df['exp_anos'].median())

# %% [markdown]
# # 7.2 Similaridade textual (memória linear)

# %%
vectorizer = TfidfVectorizer(max_features=5000)
corpus = pd.concat([df['cv_pt_clean'], df['titulo_vaga_clean']]).unique()
vectorizer.fit(corpus)

tfidf_cv    = vectorizer.transform(df['cv_pt_clean'])
tfidf_title = vectorizer.transform(df['titulo_vaga_clean'])
sim = tfidf_cv.multiply(tfidf_title).sum(axis=1)
df['sim_text'] = np.array(sim).ravel()

# %% [markdown]
# # 7.3 One-hot encoding em variáveis categóricas

# %%
for col in ['senioridade', 'nivel_ingles', 'area_atuacao']:
    if col not in df.columns:
        df[col] = 'Desconhecido'
cat_cols = ['senioridade','nivel_ingles','area_atuacao']
ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
ohe_feats = ohe.fit_transform(df[cat_cols])
f_names = ohe.get_feature_names_out(cat_cols)
df_ohe = pd.DataFrame(ohe_feats, columns=f_names, index=df.index)
df = pd.concat([df, df_ohe], axis=1)

# %% [markdown]
# # 7.4 Label Encoding do cliente

# %%
le = LabelEncoder()
df['cliente_encoded'] = le.fit_transform(df['cliente'])

# %% [markdown]
# # 8. Preparação final do dataset

# %%
feature_cols = ['exp_anos','sim_text','cliente_encoded'] + list(f_names)
X = df[feature_cols]
y = df['target']
print(f"Shape X: {X.shape}, y: {y.shape}")

# Salvar datasets
X.to_csv('features_preprocessed.csv', index=False)
y.to_csv('target.csv', index=False)

# Exportar artefatos para API
joblib.dump(vectorizer, 'vectorizer.joblib')
joblib.dump(ohe, 'ohe.joblib')
joblib.dump(le, 'le.joblib')
joblib.dump(feature_cols, 'feature_cols.joblib')

print("Pré-processamento, EDA e engenharia de features concluídos e artefatos salvos.")


