
These subdirectories contain formatt parsing code for different TCR, peptide, MHC triad input sources

All formatters output into the uniform CSV format:

```
job_name,cognate,peptide,mhc_1_type,mhc_1_name,mhc_1_seq,mhc_2_type,mhc_2_name,mhc_2_seq,tcr_1_type,tcr_1_name,tcr_1_seq,tcr_2_type,tcr_2_name,tcr_2_seq
```

`job_name`: A unique name for the row which will be used as the alphafold job name.
`cognate`: 1 for true, 0 for false.


## Installation

First, create a conda environment:

```bash
conda create -n tcr_format_parsers
```

ANARCI must be installed in your environment before installing this package. See installation instructions [here](https://github.com/oxpig/ANARCI).

Then, clone this repo and install:
```bash
git clone https://github.com/ljwoods2/tcr_format_parsers
cd tcr_format_parsers
pip install -e .
```

Finally, download the necessary IMGT data for [stitchr](https://github.com/JamieHeather/stitchr) to work
```bash
# other species must be downloaded if needed
stitchrdl -s human 
```

## Usage

CRESTA Cellranger data
```bash
python tcr_format_parsers/CRESTA_CellRanger/parse_cresta_runs.py \
    -d /path/to/cellranger/gem/dir \
    -o output_csv.csv
```

IEDB data
```bash
python tcr_format_parsers/IEDB/fetch_IEDB_triads_brute.py \
    -o output_csv.csv
```