import ast
import pytest
import tempfile
import json
import re
import numpy as np
import pandas as pd

# Extrai apenas as funções sem executar código top-level

def load_functions(path):
    with open(path, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source)
    # Ambiente mínimo para as funções rodarem
    env = {
        'json': json,
        're': re,
        'np': np
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            'load_json_flexible', 'normalize_text', 'extract_exp'
        ):
            code = ast.get_source_segment(source, node)
            exec(code, env)
    return env['load_json_flexible'], env['normalize_text'], env['extract_exp']




def test_extract_exp():
    _, _, extract_exp = load_functions('Pre_processamento.py')
    assert extract_exp('5 anos de experiência') == 5
    nan_val = extract_exp('sem informação')
    assert isinstance(nan_val, float) and np.isnan(nan_val)


def test_load_json_flexible(tmp_path):
    load_json_flexible, _, _ = load_functions('Pre_processamento.py')
    # JSON válido
    valid = tmp_path / 'file.json'
    valid.write_text(json.dumps({'a': 1}))
    assert load_json_flexible(str(valid)) == {'a': 1}

    # JSON com vírgula extra
    invalid = tmp_path / 'file2.json'
    invalid.write_text('{"b": 2,}')
    result = load_json_flexible(str(invalid))
    assert result.get('b') == 2

