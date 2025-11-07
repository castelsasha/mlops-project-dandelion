import os
import torch
from src.model.train import build_model, save_model

def test_build_model():
    m = build_model(num_classes=2)
    assert isinstance(m, torch.nn.Module)

def test_save_model(tmp_path):
    m = build_model(num_classes=2)
    out = save_model(m, output_dir=tmp_path)
    assert os.path.isfile(out)
