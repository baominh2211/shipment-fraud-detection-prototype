from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMN_RENAMES = {
    "Declaration ID": "declaration_id",
    "Date": "date",
    "Office ID": "office_id",
    "Process Type": "process_type",
    "Import Type": "import_type",
    "Import Use": "import_use",
    "Payment Type": "payment_type",
    "Mode of Transport": "mode_of_transport",
    "Declarant ID": "declarant_id",
    "Importer ID": "importer_id",
    "Seller ID": "seller_id",
    "Courier ID": "courier_id",
    "HS6 Code": "hs6_code",
    "Country of Departure": "country_of_departure",
    "Country of Origin": "country_of_origin",
    "Tax Rate": "tax_rate",
    "Tax Type": "tax_type",
    "Country of Origin Indicator": "country_of_origin_indicator",
    "Net Mass": "net_mass",
    "Item Price": "item_price",
    "Fraud": "fraud",
    "Critical Fraud": "critical_fraud",
}


def load_split(data_dir: Path | str, split: str) -> pd.DataFrame:
    return pd.read_csv(Path(data_dir) / f"df_syn_{split}_eng.csv")



def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=COLUMN_RENAMES).copy()
