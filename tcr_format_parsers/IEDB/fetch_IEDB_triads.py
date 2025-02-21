import requests
import warnings
import polars as pl
import io
from tcr_format_parsers.common.MHCCodeConverter import (
    DQA_FOR,
    HLACodeWebConverter,
)
from tcr_format_parsers.common.TCRUnique import hash_tcr_sequence
import tidytcells as tt
import argparse
from Stitchr import stitchrfunctions as fxn
from Stitchr import stitchr as st
import math

# https://help.iedb.org/hc/en-us/articles/4402872882189-Immune-Epitope-Database-Query-API-IQ-API
BASE_URL = "https://query-api.iedb.org/"
LIMIT = 10000


def paginate_get(url, params, schema_overrides=None):
    out = []

    n_queries = 1
    start_idx = 0

    while n_queries > 0:
        headers = {
            "Accept": "application/json",
            "Range-Unit": "items",
            "Range": f"{start_idx}-{start_idx + LIMIT - 1}",
        }
        if start_idx == 0:
            headers["Prefer"] = "count=exact"

        response = requests.get(url, params=params, headers=headers)
        response.raise_for_status()

        out.append(
            pl.read_json(
                io.StringIO(response.text), schema_overrides=schema_overrides
            )
        )

        if start_idx == 0:
            tot = int(response.headers["Content-Range"].split("/")[1])
            n_queries = math.ceil(tot / LIMIT)

        n_queries -= 1
        start_idx += LIMIT

    return pl.concat(out, how="vertical")


def chunk_list(lst, chunk_size):
    """Yield successive chunk_size chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def shorten_to_fullname(mhc_name):
    colon = mhc_name.count(":")
    if colon == 1:
        return mhc_name
    elif colon >= 2:
        return mhc_name.split(":")[0] + ":" + mhc_name.split(":")[1]
    else:
        return mhc_name


def is_fullname(mhc_name):
    return mhc_name.count(":") == 1


def infer_hla_chain(mhc_name):

    nullchains = {"mhc_1_name": None, "mhc_2_name": None}

    if (
        mhc_name.startswith("HLA-A")
        or mhc_name.startswith("HLA-B")
        or mhc_name.startswith("HLA-C")
        # NOTE: verify this
        or mhc_name.startswith("HLA-E")
    ):
        # https://www.uniprot.org/uniprotkb/P61769/entry

        fullname = shorten_to_fullname(mhc_name)
        if is_fullname(fullname):
            return {
                "mhc_1_name": mhc_name.split("HLA-")[1],
                "mhc_2_name": "B2M",
            }
        else:
            return nullchains
    elif mhc_name.startswith("HLA-DRB"):
        fullname = shorten_to_fullname(mhc_name)
        if is_fullname(fullname):
            return {
                "mhc_1_name": "DRA1*01:02",
                "mhc_2_name": mhc_name.split("HLA-")[1],
            }
        else:
            return nullchains
    elif mhc_name.startswith("HLA-DQB"):
        fullname = shorten_to_fullname(mhc_name)
        if is_fullname(fullname):
            b_chain = fullname.split("HLA-")[1]
            if b_chain in DQA_FOR:
                a_chain = DQA_FOR[b_chain]
                return {
                    "mhc_1_name": a_chain,
                    "mhc_2_name": b_chain,
                }
            else:
                return nullchains
                # raise ValueError(f"Unknown DQA chain for DQB chain {b_chain}")
        else:
            return nullchains

    elif mhc_name.startswith("HLA-DPB") and mhc_name != "HLA-DPB":
        # https://www.uniprot.org/uniprotkb/P20036/entry
        return {"mhc_1_name": None, "mhc_2_name": None}
        # return {"mhc_1_name": "DPA1", "mhc_2_name": mhc_name.split("HLA-")[1]}
    else:
        return nullchains


def sort_triad_types(df, species):
    both_chains_known = (
        df.filter(
            (~pl.col("chain_1__protein_sequence").is_null())
            & (~pl.col("chain_2__protein_sequence").is_null())
        )
        .rename(
            {
                "chain_1__protein_sequence": "tcr_1_seq",
                "chain_2__protein_sequence": "tcr_2_seq",
            }
        )
        .select(
            [
                "receptor_id",
                "qualitative_measure",
                "peptide",
                "mhc_class",
                "mhc_1_name",
                "mhc_2_name",
                "mhc_1_type",
                "mhc_2_type",
                "tcr_1_type",
                "tcr_2_type",
                "tcr_1_seq",
                "tcr_2_seq",
            ]
        )
        # should be noop
        .unique()
    )

    chain_1_known = (
        df.filter(
            (~pl.col("chain_1__protein_sequence").is_null())
            & (pl.col("chain_2__protein_sequence").is_null())
        )
        .select(
            pl.exclude(
                [
                    "chain_1__curated_v_gene",
                    "chain_1__calculated_v_gene",
                    "chain_1__curated_j_gene",
                    "chain_1__calculated_j_gene",
                    "chain_1__cdr3_curated",
                    "chain_1__cdr3_calculated",
                    "chain_2__protein_sequence",
                ]
            )
        )
        .rename(
            {
                "chain_1__protein_sequence": "tcr_1_seq",
            }
        )
    )

    chain_1_known = handle_one_chain(
        chain_1_known,
        "chain_2__curated_v_gene",
        "chain_2__calculated_v_gene",
        "chain_2__curated_j_gene",
        "chain_2__calculated_j_gene",
        "chain_2__cdr3_curated",
        "chain_2__cdr3_calculated",
        "tcr_2",
    )

    # filter down to rows that have calculated genes
    chain_2_known = (
        df.filter(
            (pl.col("chain_1__protein_sequence").is_null())
            & (~pl.col("chain_2__protein_sequence").is_null())
        )
        .select(
            pl.exclude(
                [
                    "chain_2__curated_v_gene",
                    "chain_2__calculated_v_gene",
                    "chain_2__curated_j_gene",
                    "chain_2__calculated_j_gene",
                    "chain_2__cdr3_curated",
                    "chain_2__cdr3_calculated",
                    "chain_1__protein_sequence",
                ]
            )
        )
        .rename(
            {
                "chain_2__protein_sequence": "tcr_2_seq",
            }
        )
    )

    chain_2_known = handle_one_chain(
        chain_2_known,
        "chain_1__curated_v_gene",
        "chain_1__calculated_v_gene",
        "chain_1__curated_j_gene",
        "chain_1__calculated_j_gene",
        "chain_1__cdr3_curated",
        "chain_1__cdr3_calculated",
        "tcr_1",
    )

    no_chains_known = df.filter(
        (pl.col("chain_1__protein_sequence").is_null())
        & (pl.col("chain_2__protein_sequence").is_null())
    ).select(
        pl.exclude(
            [
                "chain_1__protein_sequence",
                "chain_2__protein_sequence",
            ]
        )
    )

    no_chains_known = handle_one_chain(
        no_chains_known,
        "chain_1__curated_v_gene",
        "chain_1__calculated_v_gene",
        "chain_1__curated_j_gene",
        "chain_1__calculated_j_gene",
        "chain_1__cdr3_curated",
        "chain_1__cdr3_calculated",
        "tcr_1",
    )
    no_chains_known = handle_one_chain(
        no_chains_known,
        "chain_2__curated_v_gene",
        "chain_2__calculated_v_gene",
        "chain_2__curated_j_gene",
        "chain_2__calculated_j_gene",
        "chain_2__cdr3_curated",
        "chain_2__cdr3_calculated",
        "tcr_2",
    )

    return both_chains_known, chain_1_known, chain_2_known, no_chains_known


def handle_one_chain(
    df,
    v_gene_unfav,
    v_gene_fav,
    j_gene_unfav,
    j_gene_fav,
    cdr_3_unfav,
    cdr_3_fav,
    outprefix,
    # all_unfav=False,
):
    # for each of v_gene, j_gene, cdr3, if both are abset, remove row
    df = df.filter(
        ~((pl.col(cdr_3_fav).is_null()) & (pl.col(cdr_3_unfav).is_null()))
        & ~((pl.col(v_gene_fav).is_null()) & (pl.col(v_gene_unfav).is_null()))
        & ~((pl.col(j_gene_fav).is_null()) & (pl.col(j_gene_unfav).is_null()))
    )
    # for rows where cdr3fav is present, set prefix_cdr3_seq to cdr3fav
    df = df.with_columns(
        pl.when(pl.col(cdr_3_fav).is_not_null())
        .then(pl.col(cdr_3_fav))
        .otherwise(pl.col(cdr_3_unfav))
        .alias(f"{outprefix}_cdr3_seq")
    )
    # for rows where v_gene_fav is present, set prefix_v_gene to v_gene_fav
    df = df.with_columns(
        pl.when(pl.col(v_gene_fav).is_not_null())
        .then(pl.col(v_gene_fav))
        .otherwise(pl.col(v_gene_unfav))
        .alias(f"{outprefix}_v_gene")
    )
    # for rows where j_gene_fav is present, set prefix_j_gene to j_gene_fav
    df = df.with_columns(
        pl.when(pl.col(j_gene_fav).is_not_null())
        .then(pl.col(j_gene_fav))
        .otherwise(pl.col(j_gene_unfav))
        .alias(f"{outprefix}_j_gene")
    )

    df = df.with_columns(
        pl.col(f"{outprefix}_cdr3_seq")
        .map_elements(
            tt.junction.standardize, return_dtype=pl.String, skip_nulls=True
        )
        .alias(f"{outprefix}_cdr3_seq"),
    ).filter(pl.col(f"{outprefix}_cdr3_seq").is_not_null())

    df = df.with_columns(
        pl.col(f"{outprefix}_v_gene")
        .map_elements(
            tt.tr.standardize, return_dtype=pl.String, skip_nulls=True
        )
        .alias(f"{outprefix}_v_gene"),
    ).filter(pl.col(f"{outprefix}_v_gene").is_not_null())

    df = df.with_columns(
        pl.col(f"{outprefix}_j_gene")
        .map_elements(
            tt.tr.standardize, return_dtype=pl.String, skip_nulls=True
        )
        .alias(f"{outprefix}_j_gene")
    ).filter(pl.col(f"{outprefix}_j_gene").is_not_null())

    # remove input cols
    df = df.select(
        pl.exclude(
            [
                v_gene_fav,
                v_gene_unfav,
                j_gene_fav,
                j_gene_unfav,
                cdr_3_fav,
                cdr_3_unfav,
            ]
        )
    )

    return df


def stitch_helper(
    v_gene,
    j_gene,
    cdr3,
    chain,
    tcr_dat,
    codons,
    species,
    functionality,
    partial,
):  # https://github.com/JamieHeather/stitchr/blob/c528c9f2998c0c4ea7ab561233bf412f996b0502/docs/inputdata.rst
    # https://jamieheather.github.io/stitchr/usage.html
    # _gene = "TRAC*01" if chain == "TRA" else "TRBC1*01"

    # j_gene_num = j_gene.split("*")[1]

    tcr_bits = {
        "v": v_gene,
        "j": j_gene,
        "cdr3": cdr3,
        "l": v_gene,
        # infer
        "c": "",
        "skip_c_checks": False,
        "species": species,
        "seamless": False,
        "5_prime_seq": "",
        "3_prime_seq": "",
        "name": "TCR",
    }
    tcr_bits, _ = fxn.sort_input(tcr_bits)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "error",
            message="Note: while a C-terminal CDR3:J germline match has been found, it was only the string*",
        )
        try:
            out = st.stitch(
                tcr_bits, tcr_dat, functionality, partial, codons, 3, ""
            )
        except Exception as e:
            return None
    # out[-1] contains the translation offset, which we ignore since we aren't
    # providing a 5_prime_seq
    return fxn.translate_nt(out[1])


def aa_from_tcr_chain(df, chain, species, v_gene, j_gene, cdr3, outcol):
    tcr_dat, functionality, partial = fxn.get_imgt_data(
        chain, st.gene_types, species
    )
    codons = fxn.get_optimal_codons("", species)

    df = (
        df.with_columns(
            pl.struct(v_gene, j_gene, cdr3)
            .map_elements(
                lambda x: stitch_helper(
                    x[v_gene],
                    x[j_gene],
                    x[cdr3],
                    chain,
                    tcr_dat,
                    codons,
                    species,
                    functionality,
                    partial,
                ),
                return_dtype=pl.String,
                skip_nulls=True,
            )
            .alias(outcol),
        )
        .filter(pl.col(outcol).is_not_null())
        .select(pl.exclude([v_gene, j_gene, cdr3]))
    )
    return df


def format_df(df, cognate, converter, tcr_chain_sequence_stitched):
    keep_cols = [
        "cognate",
        "job_name",
        "peptide",
        "mhc_1_type",
        "mhc_1_name",
        "mhc_1_seq",
        "mhc_2_type",
        "mhc_2_name",
        "mhc_2_seq",
        "tcr_1_type",
        "tcr_1_name",
        "tcr_1_seq",
        "tcr_2_type",
        "tcr_2_name",
        "tcr_2_seq",
        "qualitative_measure",
        "tcr_chain_sequence_stitched",
    ]
    df = (
        df.with_columns(
            [
                pl.col("mhc_1_name")
                .map_elements(
                    lambda x: converter.get_sequence(x, top_only=True),
                    return_dtype=pl.String,
                )
                .alias("mhc_1_seq"),
                pl.col("mhc_2_name")
                .map_elements(
                    lambda x: converter.get_sequence(
                        x,
                        top_only=True,
                    ),
                    return_dtype=pl.String,
                )
                .alias("mhc_2_seq"),
                pl.col("tcr_1_seq")
                .map_elements(hash_tcr_sequence, return_dtype=pl.String)
                .alias("tcr_1_name"),
                pl.col("tcr_2_seq")
                .map_elements(hash_tcr_sequence, return_dtype=pl.String)
                .alias("tcr_2_name"),
                pl.lit(cognate).alias("cognate"),
                pl.lit(tcr_chain_sequence_stitched).alias(
                    "tcr_chain_sequence_stitched"
                ),
            ]
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.col("peptide"),
                    pl.col("mhc_1_name"),
                    pl.col("mhc_2_name"),
                    pl.col("tcr_1_name"),
                    pl.col("tcr_2_name"),
                ],
                separator="_",
            ).alias("job_name")
        )
        .select(keep_cols)
        .unique()
    )
    return df


CHAIN_COLS = [
    "chain_1__curated_v_gene",
    "chain_1__calculated_v_gene",
    "chain_1__curated_j_gene",
    "chain_1__calculated_j_gene",
    "chain_1__protein_sequence",
    "chain_1__cdr3_curated",
    "chain_1__cdr3_calculated",
    "chain_2__curated_v_gene",
    "chain_2__calculated_v_gene",
    "chain_2__curated_j_gene",
    "chain_2__calculated_j_gene",
    "chain_2__protein_sequence",
    "chain_2__cdr3_curated",
    "chain_2__cdr3_calculated",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        "-o",
        help="Output directory path",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--negative",
        help="Query for negative triads",
        action="store_true",
    )
    args = parser.parse_args()

    cognate = not args.negative

    cols = [
        "tcell_ids:assay__iedb_ids",
        "receptor_id:receptor__iedb_receptor_id",
        "peptide:epitope__name",
        "mhc_allele_names:assay__mhc_allele_names",
        "chain_1__curated_v_gene",
        "chain_1__calculated_v_gene",
        "chain_1__curated_j_gene",
        "chain_1__calculated_j_gene",
        "chain_1__protein_sequence",
        "chain_1__cdr3_curated",
        "chain_1__cdr3_calculated",
        "chain_2__curated_v_gene",
        "chain_2__calculated_v_gene",
        "chain_2__curated_j_gene",
        "chain_2__calculated_j_gene",
        "chain_2__protein_sequence",
        "chain_2__cdr3_curated",
        "chain_2__cdr3_calculated",
    ]
    col_string = ",".join(cols)

    params = {
        "receptor__type": "eq.alphabeta",
        "assay__iedb_ids": "not.is.null",
        "epitope__name": "not.is.null",
        "and": "(assay__mhc_allele_names.not.is.null,assay__mhc_allele_names.not.like.*mutant*)",
        "select": col_string,
    }

    schema_overrides = {k: pl.String for k in CHAIN_COLS}

    tcr_export_dat = paginate_get(
        BASE_URL + "tcr_export", params, schema_overrides=schema_overrides
    )
    tcr_export_dat = (
        tcr_export_dat.with_columns(
            pl.col("tcell_ids")
            .str.split(",")
            .list.eval(pl.element().str.strip_chars().cast(pl.UInt32))
            .alias("tcell_ids"),
            pl.col("mhc_allele_names")
            .str.split(",")
            .list.eval(pl.element().str.strip_chars())
            .alias("mhc_allele_names"),
        )
        .filter(pl.col("peptide").str.contains(r"^[A-Z]+$"))
        .explode("mhc_allele_names")
        .explode("tcell_ids")
        .rename(
            {"mhc_allele_names": "mhc_allele_name", "tcell_ids": "tcell_id"}
        )
        .unique()
    ).lazy()

    cols = [
        "tcell_id",
        "peptide:linear_sequence",
        "receptor_ids",
        "mhc_class",
        "qualitative_measure",
        "mhc_restriction",
        "mhc_allele_evidence",
    ]
    col_string = ",".join(cols)

    if cognate:
        measure = "Positive"
    else:
        measure = "Negative"

    params = {
        # "linear_sequence": "not.is.null",
        # "receptor_ids": "not.is.null",
        "and": "(mhc_restriction.not.is.null,mhc_restriction.not.like.*mutant*)",
        # "or": "(mhc_allele_evidence.eq.Single allele present,mhc_allele_evidence.eq.MHC binding assay)",
        "mhc_allele_evidence": "eq.Single allele present",
        "qualitative_measure": f"like.{measure}*",
        "select": col_string,
    }

    schema_overrides = {
        "tcell_id": pl.UInt32,
        "receptor_ids": pl.List(pl.UInt32),
        "peptide": pl.String,
        "mhc_class": pl.String,
        "qualitative_measure": pl.String,
        "mhc_restriction": pl.String,
        "mhc_allele_evidence": pl.String,
    }

    tcell_search_dat = (
        paginate_get(
            BASE_URL + "tcell_search",
            params,
            schema_overrides=schema_overrides,
        )
        .select(pl.exclude("receptor_ids"))
        .unique()
    ).lazy()

    tbl = (
        (
            tcr_export_dat.join(
                tcell_search_dat,
                left_on=["tcell_id", "peptide", "mhc_allele_name"],
                right_on=["tcell_id", "peptide", "mhc_restriction"],
                how="inner",
            )
            .group_by(pl.exclude("mhc_allele_evidence", "qualitative_measure"))
            .agg(pl.col("mhc_allele_evidence"), pl.col("qualitative_measure"))
        )
        .select(pl.exclude("tcell_id"))
        .unique()
    )

    # pick the more specific measure
    tbl = tbl.with_columns(
        pl.col("qualitative_measure")
        .map_elements(lambda lst: max(lst, key=len), return_dtype=pl.String)
        .alias("qualitative_measure")
    )

    tbl = tbl.with_columns(
        pl.lit("tcr_alpha").alias("tcr_1_type"),
        pl.lit("tcr_beta").alias("tcr_2_type"),
    )

    tbl = tbl.collect()

    two_chain_hla = tbl.filter(
        pl.col("mhc_allele_name").str.contains("/")
    ).filter(pl.col("mhc_allele_name").str.starts_with("HLA"))

    two_chain_hla = (
        two_chain_hla.with_columns(
            pl.col("mhc_allele_name")
            .str.split_exact("/", 1)
            .alias("split_parts")
        )
        .unnest("split_parts")
        .rename({"field_0": "mhc_1_name", "field_1": "mhc_2_name"})
        .with_columns(
            pl.col("mhc_1_name").str.replace("HLA-", "").alias("mhc_1_name"),
        )
    )

    one_chain_hla = tbl.join(
        two_chain_hla, on="mhc_allele_name", how="anti"
    ).filter(pl.col("mhc_allele_name").str.starts_with("HLA"))

    tmp_struct = pl.Struct(
        {
            "mhc_1_name": pl.String,
            "mhc_2_name": pl.String,
        }
    )

    one_chain_hla = (
        one_chain_hla.with_columns(
            pl.col("mhc_allele_name")
            .map_elements(infer_hla_chain, return_dtype=tmp_struct)
            .alias("chains")
        )
        .unnest("chains")
        .filter(
            (pl.col("mhc_1_name").is_not_null())
            & (pl.col("mhc_2_name").is_not_null())
        )
    )

    human = (
        pl.concat([one_chain_hla, two_chain_hla], how="vertical")
        .select(pl.exclude("mhc_allele_name"))
        .filter(pl.col("mhc_1_name").is_not_null())
    ).with_columns(
        [
            pl.when(pl.col("mhc_class") == "I")
            .then(pl.lit("hla_I_alpha"))
            .otherwise(pl.lit("hla_II_alpha"))
            .alias(f"mhc_1_type"),
            pl.when(pl.col("mhc_class") == "I")
            .then(pl.lit("hla_I_beta"))
            .otherwise(pl.lit("hla_II_beta"))
            .alias(f"mhc_2_type"),
        ]
    )
    # Also available in dset: Mamu, Gaga
    mouse = tbl.filter(pl.col("mhc_allele_name").str.starts_with("H2"))

    both, chain_1_known, chain_2_known, neither_known = sort_triad_types(
        human, "human"
    )

    # both.write_csv(f"{args.output}/both_human.csv")

    chain_1_known = aa_from_tcr_chain(
        chain_1_known,
        "TRB",
        "HUMAN",
        "tcr_2_v_gene",
        "tcr_2_j_gene",
        "tcr_2_cdr3_seq",
        "tcr_2_seq",
    )

    chain_2_known = aa_from_tcr_chain(
        chain_2_known,
        "TRA",
        "HUMAN",
        "tcr_1_v_gene",
        "tcr_1_j_gene",
        "tcr_1_cdr3_seq",
        "tcr_1_seq",
    )

    neither_known = aa_from_tcr_chain(
        neither_known,
        "TRA",
        "HUMAN",
        "tcr_1_v_gene",
        "tcr_1_j_gene",
        "tcr_1_cdr3_seq",
        "tcr_1_seq",
    )
    neither_known = aa_from_tcr_chain(
        neither_known,
        "TRB",
        "HUMAN",
        "tcr_2_v_gene",
        "tcr_2_j_gene",
        "tcr_2_cdr3_seq",
        "tcr_2_seq",
    )

    converter = HLACodeWebConverter()

    both = format_df(both, cognate, converter, "neither chain")
    chain_1_known = format_df(chain_1_known, cognate, converter, "chain 2")
    chain_2_known = format_df(chain_2_known, cognate, converter, "chain 1")
    neither_known = format_df(neither_known, cognate, converter, "both chains")

    out = pl.concat([both, chain_1_known, chain_2_known, neither_known])
    out.write_csv(f"{args.output}")
