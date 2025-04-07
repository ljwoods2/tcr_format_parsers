__all__ = ["hla_uniprot_data_cache"]

from importlib import resources
from pathlib import Path
import csv

HEADERS = ["name","sequence","accession"]

_data_ref = resources.files("tcr_format_parsers.data")
hla_uniprot_data_cache = (Path(_data_ref) / "hla_uniprot_data_cache.csv")

if not hla_uniprot_data_cache.exists():
    with open(hla_uniprot_data_cache, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(HEADERS)


del resources
