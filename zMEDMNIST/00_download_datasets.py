
from pathlib import Path

from medmnist import (
    OrganAMNIST,
    PathMNIST,
    ChestMNIST,
    OCTMNIST
)

DATA_DIR = Path(
    "/users/project1/pt01191/MMODAL_ISIC/Data/MEDMNIST"
)

DATA_DIR.mkdir(parents=True, exist_ok=True)

datasets = {
    "OrganAMNIST": OrganAMNIST,
    "PathMNIST": PathMNIST,
    "ChestMNIST": ChestMNIST,
    "OCTMNIST": OCTMNIST,
}

SIZE = 224

splits = ["train", "val", "test"]

for name, DatasetClass in datasets.items():
    print(f"\nPobieranie zbioru: {name}")
    for split in splits:
        print(f"  Split: {split}")
        dataset = DatasetClass(
            split=split,
            root=str(DATA_DIR),
            download=True,
            size=SIZE
        )
        print(f"  Liczba próbek: {len(dataset)}")
    print(f"Zbiór {name} pobrany pomyślnie.")
print("\nPobieranie wszystkich zbiorów zakończone.")
print(f"Lokalizacja danych: {DATA_DIR}")
