import os
import glob
import json
import pandas as pd

folder = "output"

for file in glob.glob(os.path.join(folder, "*seed*.json")):
    data = []

    # Baca file JSONL
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # abaikan baris kosong
                data.append(json.loads(line))

    # Ubah menjadi DataFrame
    df = pd.DataFrame(data)

    list_columns = [
        "best_route",
        "parent_1",
        "parent_2",
        "child_1",
        "child_2",
        "mutation_child_1",
        "mutation_child_2"
    ]

    for col in list_columns:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: "-".join(map(str, x)) if isinstance(x, list) else x)

    # Simpan ke Excel
    excel_file = file.replace(".json", ".xlsx")
    df.to_excel(excel_file, index=False)

    print(f"Berhasil: {excel_file}")