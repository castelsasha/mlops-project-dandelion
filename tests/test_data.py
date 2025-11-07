import os
from src.data.dataset import PlantDataset

def test_dataset_local_csv():
    meta = "data/raw/metadata.csv"
    # si la première run n’a pas encore fetch -> on skip cleanement
    if not os.path.exists(meta):
        return

    ds = PlantDataset(meta)
    assert len(ds) > 0

    x, y = ds[0]
    # image tensor shape (3, 224, 224)
    assert x.shape[0] == 3
    assert x.shape[1] == 224
    assert x.shape[2] == 224
    # label doit être 0 ou 1
    assert y in (0, 1)
