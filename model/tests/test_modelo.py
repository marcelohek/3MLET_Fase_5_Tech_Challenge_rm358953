import pytest
import sys
import subprocess
import pandas as pd
import numpy as np
import shutil
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "Modelo.py"

@pytest.fixture
def dummy_env(tmp_path):
    # Agora geramos 8 amostras: 4 de cada classe
    X = pd.DataFrame({
        'f1': np.arange(8),
        'f2': np.arange(8, 16)
    })
    # [0,1] repetido 4 vezes → 4 zeros e 4 uns
    y = pd.Series([0, 1] * 4, name='target')

    X.to_csv(tmp_path / "features_preprocessed.csv", index=False)
    y.to_csv(tmp_path / "target.csv", index=False)
    shutil.copy(SCRIPT, tmp_path / "Modelo.py")
    return tmp_path

def test_modelo_script_runs_and_saves_model(dummy_env):
    # Executa o script dentro do tmp_path
    result = subprocess.run(
        [sys.executable, "Modelo.py"],
        cwd=dummy_env,
        capture_output=True,
        text=True
    )
    # Deve finalizar com código zero
    assert result.returncode == 0, f"Erro ao rodar Modelo.py:\n{result.stderr}"

    # Verifica se o modelo foi salvo
    model_file = dummy_env / "xgb_model.joblib"
    assert model_file.exists() and model_file.stat().st_size > 0
