__all__ = ["hla_uniprot_data_cache"]

from importlib import resources
from pathlib import Path

_data_ref = resources.files("tcr_format_parsers.data")

hla_uniprot_data_cache = (
    Path(_data_ref) / "hla_uniprot_data_cache.csv"
).as_posix()


del resources
