import os
from collections import Counter

from datasets import (
    ClassLabel,
    DatasetDict,
    Features,
    Value,
    concatenate_datasets,
    load_dataset,
)

# Configuracion
REPO_ID = "project-oceania/planktonzilla-17M"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "planktonzilla_17M_only_plankton")

SEED = 42
TEST_FRAC = 0.2
VAL_FRAC = 0.2
MIN_CLASS_FREQ = 5  # clases con menos ejemplos se mantienen integras en train

# Rango taxonomico usado para construir la etiqueta, de mayor a menor jerarquia.
TAXONOMY_COLS = ["Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species"]


def build_only_plankton(ds, num_proc=1):
    """Filtra a solo plankton con taxonomia y codifica la etiqueta taxonomica."""

    # Mascara de plankton: marcado como plankton y con Kingdom asignado.
    ds = ds.filter(
        lambda x: x["plankton"] is True and x["Kingdom"] != "",
        num_proc=num_proc,
    )

    # La etiqueta es el camino taxonomico completo, ignorando los rangos vacios.
    def build_tax_string(example):
        tax = [example[c] for c in TAXONOMY_COLS if example[c] not in ("", None)]
        return {"tax_label": " ".join(tax)}

    ds = ds.map(build_tax_string, num_proc=num_proc)

    # Codificamos cada string taxonomico unico a un entero de clase.
    unique_labels = sorted(set(ds["tax_label"]))
    class_label = ClassLabel(names=unique_labels)

    def encode_label(example):
        return {"label": class_label.str2int(example["tax_label"])}

    ds = ds.map(encode_label, num_proc=num_proc)

    # Nos quedamos solo con lo necesario para entrenar.
    ds = ds.remove_columns(
        [c for c in ds.column_names if c not in ["image", "label", "dataset"]]
    )

    ds = ds.cast(
        Features({
            "image": ds.features["image"],
            "label": class_label,
            "dataset": Value("string"),
        })
    )

    return ds


def stratified_split_by_dataset(ds, num_proc, seed=SEED, test_frac=TEST_FRAC, val_frac=VAL_FRAC):
    """Split train/val/test estratificado por dataset y por taxonomia.

    El split se hace de forma independiente dentro de cada dataset de origen, y
    dentro de cada uno se estratifica por etiqueta. Las clases con menos de
    MIN_CLASS_FREQ ejemplos se envian completas a train.
    """
    train_splits = []
    val_splits = []
    test_splits = []

    for dname in sorted(set(ds["dataset"])):
        ds_sub = ds.filter(lambda x: x["dataset"] == dname, num_proc=num_proc)

        labels = ds_sub["label"]
        counts = Counter(labels)

        # Clases minoritarias: se reservan para train para no perderlas en val/test.
        minority = {k for k, v in counts.items() if v < MIN_CLASS_FREQ}
        minority_idx = [i for i, y in enumerate(labels) if y in minority]
        remaining_idx = [i for i, y in enumerate(labels) if y not in minority]

        ds_minority = ds_sub.select(minority_idx) if minority_idx else None
        ds_remaining = ds_sub.select(remaining_idx) if remaining_idx else None

        # Si tras quitar las minoritarias no queda nada que dividir, todo va a train.
        if ds_remaining is None or len(ds_remaining) == 0:
            train_splits.append(ds_sub)
            continue

        n = len(ds_remaining)

        # Primer corte: train contra el bloque reservado para val + test.
        try:
            splits = ds_remaining.train_test_split(
                test_size=int(n * (test_frac + val_frac)),
                shuffle=True,
                seed=seed,
                stratify_by_column="label",
            )
        except ValueError:
            splits = ds_remaining.train_test_split(
                test_size=int(n * (test_frac + val_frac)),
                shuffle=True,
                seed=seed,
            )

        train_split = splits["train"]
        val_test_split = splits["test"]

        # Segundo corte: separamos val y test dentro del bloque reservado.
        try:
            splits = val_test_split.train_test_split(
                test_size=int(n * val_frac),
                shuffle=True,
                seed=seed,
                stratify_by_column="label",
            )
        except ValueError:
            splits = val_test_split.train_test_split(
                test_size=int(n * val_frac),
                shuffle=True,
                seed=seed,
            )

        test_split = splits["train"]
        val_split = splits["test"]

        # Anadimos las clases minoritarias reservadas a train.
        if ds_minority is not None:
            train_split = concatenate_datasets([train_split, ds_minority])

        train_splits.append(train_split)
        val_splits.append(val_split)
        test_splits.append(test_split)

    train_ds = concatenate_datasets(train_splits)
    val_ds = concatenate_datasets(val_splits) if val_splits else None
    test_ds = concatenate_datasets(test_splits) if test_splits else None

    return train_ds, val_ds, test_ds


def main():
    num_proc = int(os.cpu_count()/2)

    ds = load_dataset(REPO_ID, split="train")
    ds = build_only_plankton(ds, num_proc=num_proc)

    train_ds, val_ds, test_ds = stratified_split_by_dataset(ds, num_proc=num_proc)

    dataset = DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    })

    dataset.save_to_disk(OUTPUT_DIR)
    print("DONE")

if __name__ == "__main__":
    main()