import editdistance
import polars as pl
from tcr_format_parsers.common.TCRUtils import hash_tcr_sequence

FORMAT_COLS = [
    "job_name",
    "cognate",
    "peptide",
    "mhc_class",
    "mhc_1_chain",
    "mhc_1_species",
    "mhc_1_name",
    "mhc_1_seq",
    "mhc_2_chain",
    "mhc_2_species",
    "mhc_2_name",
    "mhc_2_seq",
    "tcr_1_chain",
    "tcr_1_species",
    "tcr_1_seq",
    "tcr_2_chain",
    "tcr_2_species",
    "tcr_2_seq",
]

FORMAT_TCR_COLS = [
    "tcr_1_chain",
    "tcr_1_species",
    "tcr_1_seq",
    "tcr_2_chain",
    "tcr_2_species",
    "tcr_2_seq",
]

FORMAT_ANTIGEN_COLS = [
    "peptide",
    "mhc_class",
    "mhc_1_chain",
    "mhc_1_species",
    "mhc_1_name",
    "mhc_1_seq",
    "mhc_2_chain",
    "mhc_2_species",
    "mhc_2_name",
    "mhc_2_seq",
]

FORMAT_MHC_COLS = [
    "mhc_class",
    "mhc_1_chain",
    "mhc_1_species",
    "mhc_1_name",
    "mhc_1_seq",
    "mhc_2_chain",
    "mhc_2_species",
    "mhc_2_name",
    "mhc_2_seq",
]


def generate_job_name(df):
    df = df.with_columns(
        pl.concat_str(
            pl.concat_str(
                [
                    pl.col("peptide"),
                    pl.col("mhc_1_seq"),
                    pl.col("mhc_2_seq"),
                    pl.col("tcr_1_seq"),
                    pl.col("tcr_2_seq"),
                ],
            )
            .map_elements(
                lambda x: hash_tcr_sequence(x, "md5"), return_dtype=pl.String
            )
            .alias("job_name"),
        )
    )
    return df


def generate_negatives(df):

    cognate = df.select(
        "peptide",
        "mhc_1_seq",
        "mhc_2_seq",
        "tcr_1_seq",
        "tcr_2_seq",
    ).unique()

    tcr_12 = df.select(
        "tcr_1_chain",
        "tcr_1_species",
        "tcr_1_seq",
        "tcr_2_chain",
        "tcr_2_species",
        "tcr_2_seq",
    ).unique()
    pmhc = df.select(
        "peptide",
        "mhc_class",
        "mhc_1_name",
        "mhc_1_species",
        "mhc_1_chain",
        "mhc_1_seq",
        "mhc_2_name",
        "mhc_2_species",
        "mhc_2_chain",
        "mhc_2_seq",
    ).unique()

    n_samples = df.height
    edit_thresh = 3
    noncognate_list = []
    found_set = set()
    i = 0

    while i < n_samples:
        tcr = tcr_12.sample(n=1)
        antigen = pmhc.sample(n=1)

        peptide = antigen.select("peptide").item()
        mhc_1_seq = antigen.select("mhc_1_seq").item()
        mhc_2_seq = antigen.select("mhc_2_seq").item()
        tcr_1_seq = tcr.select("tcr_1_seq").item()
        tcr_2_seq = tcr.select("tcr_2_seq").item()

        noncognate_row = tcr.join(antigen, how="cross")

        # case 0: we can't have already pulled this sample
        query = peptide + mhc_1_seq + mhc_2_seq + tcr_1_seq + tcr_2_seq
        if query in found_set:
            continue

        # case 1: our random selection must not be a cognate
        if (
            cognate.filter(
                (pl.col("tcr_1_seq") == tcr_1_seq)
                & (pl.col("tcr_2_seq") == tcr_2_seq)
                & (pl.col("peptide") == peptide)
                & (pl.col("mhc_1_seq") == mhc_1_seq)
                & (pl.col("mhc_2_seq") == mhc_2_seq)
            ).height
            > 0
        ):
            continue

        # case 2: our random selection must not be within edit_thresh of
        # a cognate's tcr12 and pmhc

        pmhc_edit_dist = pmhc.with_columns(
            pl.struct(
                pl.col("peptide"), pl.col("mhc_1_seq"), pl.col("mhc_2_seq")
            )
            .map_elements(
                lambda x: (editdistance.eval(x["mhc_1_seq"], mhc_1_seq))
                + (editdistance.eval(x["mhc_2_seq"], mhc_2_seq))
                + (editdistance.eval(x["peptide"], peptide)),
                return_dtype=pl.Int64,
            )
            .alias("edit_dist")
        ).filter(pl.col("edit_dist") <= edit_thresh)

        similar_cognate = cognate.join(
            pmhc_edit_dist,
            on=["peptide", "mhc_1_seq", "mhc_2_seq"],
            how="inner",
        )

        tcr_edit_dist = (
            similar_cognate.select("tcr_1_seq", "tcr_2_seq")
            .unique()
            .with_columns(
                pl.struct(pl.col("tcr_1_seq"), pl.col("tcr_2_seq"))
                .map_elements(
                    lambda x: (editdistance.eval(x["tcr_1_seq"], tcr_1_seq))
                    + (editdistance.eval(x["tcr_2_seq"], tcr_2_seq)),
                    return_dtype=pl.Int64,
                )
                .alias("edit_dist")
            )
            .filter(pl.col("edit_dist") <= edit_thresh)
        )

        if tcr_edit_dist.height != 0:
            continue

        noncognate_list.append(noncognate_row)
        found_set.add(query)

        i += 1
    noncognate = pl.concat(noncognate_list, how="vertical")
    noncognate = generate_job_name(noncognate)
    noncognate = noncognate.with_columns(pl.lit(False).alias("cognate"))
    noncognate = noncognate.select(FORMAT_COLS)
    return noncognate
