
These subdirectories contain format parsing code for different TCR, peptide, MHC triad input sources

All formatters output into the uniform CSV format:

```
job_name,cognate,peptide,mhc_class,mhc_1_chain,mhc_1_species,mhc_1_name,mhc_1_seq,mhc_2_chain,mhc_2_species,mhc_2_name,mhc_2_seq,tcr_1_chain,tcr_1_species,tcr_1_seq,tcr_2_chain,tcr_2_species,tcr_2_seq
```

`job_name`: A unique name for the row for downstream use

`cognate`: "true" or "false" string indicating if the triad binds or not

`peptide`: The peptide's amino acid sequence

`mhc_class`: "I" or "II"

`mhc_<1/2>_chain`: "alpha"/"beta"/"heavy"/"light"

`tcr_<1/2>_chain`: "alpha"/"beta"

`<type>_<1/2>_species`: Species of origin for the chain ("human"/"mouse")

`<type>_<1/2>_seq`: Amino acid sequence for the triad chain

Additionally, for pMHC-only tasks, just exclude the tcr-related columns (for example, for use with [af3-nf](https://github.com/ljwoods2/af3-nf) pMHC-only prediction pipelines)

## Installation

### Conda env

First, create a conda environment:

```bash
conda create -n tcr_format_parsers
conda env update -n tcr_format_parsers --file devtools/conda-envs/test.yaml
```

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

### Pip only

Install [hmmer=3.3.2](http://hmmer.org/) (a dependency for ANARCI)

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
