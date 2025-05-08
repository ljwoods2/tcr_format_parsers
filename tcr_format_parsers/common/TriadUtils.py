import editdistance
import polars as pl
from tcr_format_parsers.common.TCRUtils import hash_tcr_sequence, pw_tcrdist
import numpy as np
from tcrdist.repertoire import TCRrep
from tcrdist.diversity import (
    generalized_simpsons_entropy,
    simpsons_difference,
    fuzzy_diversity,
)


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

TCRDIST_COLS = [
    "tcr_1_cdr_1",
    "tcr_1_cdr_2",
    "tcr_1_cdr_3",
    "tcr_2_cdr_1",
    "tcr_2_cdr_2",
    "tcr_2_cdr_3",
    "tcr_1_v_gene",
    "tcr_2_v_gene",
    "tcr_1_j_gene",
    "tcr_2_j_gene",
]


def generate_job_name(df, addtl_cols=[]):
    df = df.with_columns(
        pl.concat_str(
            pl.concat_str(
                [
                    pl.col("peptide"),
                    pl.col("mhc_1_seq"),
                    pl.col("mhc_2_seq"),
                    pl.col("tcr_1_seq"),
                    pl.col("tcr_2_seq"),
                    *[pl.col(colname) for colname in addtl_cols],
                ],
                ignore_nulls=True,
            )
            .map_elements(
                lambda x: hash_tcr_sequence(x, "md5"), return_dtype=pl.String
            )
            .alias("job_name"),
        )
    )
    return df


# def generate_negatives_antigen_matched(df, thresh=120):
#     """
#     DF must contain FORMAT_COLS as well as TCRDIST_COLS

#     Rows must be totally unqiue
#     """

#     edit_thresh = 3

#     tmp_dfs = []

#     all_antigens = df.select(FORMAT_ANTIGEN_COLS).unique()

#     all_tcrs = df.select(FORMAT_TCR_COLS + TCRDIST_COLS).unique()

#     all_tcrs, dist_mtx = pw_tcrdist(all_tcrs)

#     df_with_idx = df.join(all_tcrs, on=FORMAT_TCR_COLS + TCRDIST_COLS)
#     found_set = set()

#     for row in df.select(FORMAT_COLS).iter_rows(named=True):

#         antigen = pl.DataFrame(row).select(FORMAT_ANTIGEN_COLS)

#         # tcrs_for_antigen = (
#         #     df.join(antigen, on=FORMAT_ANTIGEN_COLS, how="inner")
#         #     .select("index")
#         #     .unique()
#         #     .to_series()
#         #     .to_numpy()
#         # )

#         # don't favor TCRs from a particularly well-represented antigen
#         # do this by first randomly sampling an antigen,
#         # then choosing a random TCR from it

#         # a noncognate must follow these criteria:
#         # - it cannot BOTH be within thresh TCRdist units from a cognate
#         # triad's TCR AND be within edit distance 3 of a pMHC

#         noncognate_tcr_found = False

#         while not noncognate_tcr_found:

#             other_antigens = all_antigens.join(
#                 antigen, on=FORMAT_ANTIGEN_COLS, how="anti"
#             )

#             other_antigen = other_antigens.sample(n=1, shuffle=True)

#             tcr_other_antigen = (
#                 df_with_idx.join(
#                     other_antigen, on=FORMAT_ANTIGEN_COLS, how="inner"
#                 )
#                 .select(FORMAT_TCR_COLS + TCRDIST_COLS + ["index"])
#                 .unique()
#             ).sample(n=1, shuffle=True)

#             # negs must be unqiue

#             peptide = antigen.select("peptide").item()
#             mhc_1_seq = antigen.select("mhc_1_seq").item()
#             mhc_2_seq = antigen.select("mhc_2_seq").item()
#             tcr_1_seq = tcr_other_antigen.select("tcr_1_seq").item()
#             tcr_2_seq = tcr_other_antigen.select("tcr_2_seq").item()

#             query = peptide + mhc_1_seq + mhc_2_seq + tcr_1_seq + tcr_2_seq

#             if query in found_set:
#                 continue

#             target_row = tcr_other_antigen.select("index").item()

#             # this includes self-distance
#             # so must be removed w list comprehension
#             tcr_hits = [
#                 idx
#                 for idx in np.argwhere(dist_mtx[target_row] <= thresh)[:, 0]
#                 if idx != target_row
#             ]

#             # tcr doesn't look like any tcrab in
#             # other cognate complexes
#             if len(tcr_hits) == 0:
#                 # not suspect positive
#                 tmp_dfs.append(
#                     antigen.join(
#                         tcr_other_antigen.select(pl.exclude("index")),
#                         how="cross",
#                     )
#                 )
#                 noncognate_tcr_found = True
#                 found_set.add(query)
#                 continue

#             tmp_idx = pl.DataFrame({"index": tcr_hits})

#             triad_tcr_pmhc_matched = (
#                 # filt down to only matching TCRs
#                 # now, are there matching pMHCs?
#                 df_with_idx.join(tmp_idx, on="index")
#                 .select(FORMAT_ANTIGEN_COLS)
#                 .unique()
#                 .with_columns(
#                     pl.struct(
#                         pl.col("peptide"),
#                         pl.col("mhc_1_seq"),
#                         pl.col("mhc_2_seq"),
#                     )
#                     .map_elements(
#                         lambda x: (
#                             editdistance.eval(x["mhc_1_seq"], mhc_1_seq)
#                         )
#                         + (editdistance.eval(x["mhc_2_seq"], mhc_2_seq))
#                         + (editdistance.eval(x["peptide"], peptide)),
#                         return_dtype=pl.Int64,
#                     )
#                     .alias("edit_dist")
#                 )
#                 .filter(pl.col("edit_dist") <= edit_thresh)
#             ).height > 0

#             if triad_tcr_pmhc_matched:
#                 # suspect positive
#                 continue

#             # not suspect positive
#             tmp_dfs.append(
#                 antigen.join(
#                     tcr_other_antigen.select(pl.exclude("index")),
#                     how="cross",
#                 )
#             )
#             noncognate_tcr_found = True
#             found_set.add(query)

SOURCE_ANTIGEN_COLS = ["source_" + colname for colname in FORMAT_ANTIGEN_COLS]
SOURCE_RENAME_DICT = {
    k: v for k, v in zip(FORMAT_ANTIGEN_COLS, SOURCE_ANTIGEN_COLS)
}
SOURCE_REV_RENAME_DICT = {
    v: k for k, v in zip(FORMAT_ANTIGEN_COLS, SOURCE_ANTIGEN_COLS)
}


#     noncognate = pl.concat(tmp_dfs, how="vertical")
#     noncognate = generate_job_name(noncognate)
#     noncognate = noncognate.with_columns(pl.lit(False).alias("cognate"))
#     noncognate = noncognate.select(FORMAT_COLS + TCRDIST_COLS)
#     return noncognate
def generate_all_possible_negs(
    df, tcrdist_thresh=120, pmhc_edit_thresh=3, cross_class=False
):
    """
    DF must contain FORMAT_COLS as well as TCRDIST_COLS

    Rows must be totally unqiue
    """
    tmp_dfs = []

    all_antigens = df.select(FORMAT_ANTIGEN_COLS).unique()
    all_tcrs = df.select(FORMAT_TCR_COLS + TCRDIST_COLS).unique()
    all_tcrs, dist_mtx = pw_tcrdist(all_tcrs)

    df_with_idx = df.join(all_tcrs, on=FORMAT_TCR_COLS + TCRDIST_COLS)
    found_set = set()

    for row in df.select(FORMAT_ANTIGEN_COLS).unique().iter_rows(named=True):

        antigen = pl.DataFrame(row).select(FORMAT_ANTIGEN_COLS)

        # tcrs_for_antigen = (
        #     df.join(antigen, on=FORMAT_ANTIGEN_COLS, how="inner")
        #     .select("index")
        #     .unique()
        #     .to_series()
        #     .to_numpy()
        # )

        # don't favor TCRs from a particularly well-represented antigen
        # do this by first randomly sampling an antigen,
        # then choosing a random TCR from it

        neg_found = 0

        other_antigens = all_antigens.join(
            antigen, on=FORMAT_ANTIGEN_COLS, how="anti"
        )

        if cross_class:
            other_antigens = other_antigens.filter(
                pl.col("mhc_class") != antigen.select("mhc_class").unique()
            )

        # other_antigen = other_antigens.sample(n=1, shuffle=True)

        # # do we reject based on antigen distance?

        peptide = antigen.select("peptide").item()
        mhc_1_seq = antigen.select("mhc_1_seq").item()
        mhc_2_seq = antigen.select("mhc_2_seq").item()

        other_antigens = other_antigens.with_columns(
            pl.struct(
                pl.col("peptide"),
                pl.col("mhc_1_seq"),
                pl.col("mhc_2_seq"),
            )
            .map_elements(
                lambda x: (editdistance.eval(x["mhc_1_seq"], mhc_1_seq))
                + (editdistance.eval(x["mhc_2_seq"], mhc_2_seq))
                + (editdistance.eval(x["peptide"], peptide)),
                return_dtype=pl.Int64,
            )
            .alias("edit_dist")
        ).filter(pl.col("edit_dist") > pmhc_edit_thresh)

        # if antigen_distance <= pmhc_edit_thresh:
        #     # reject, antigen too close
        #     continue

        # select a TCR
        # tcrs_other_antigen = (
        #     df_with_idx.join(
        #         other_antigens, on=FORMAT_ANTIGEN_COLS, how="inner"
        #     )
        #     .select(FORMAT_TCR_COLS + TCRDIST_COLS + ["index"])
        #     .unique()
        # )

        other_antigen_tcr_indices_boolmask = np.zeros(
            dist_mtx.shape[0], dtype=np.bool
        )
        other_antigen_tcr_indices = np.sort(
            (
                df_with_idx.join(
                    other_antigens, on=FORMAT_ANTIGEN_COLS, how="inner"
                )
                .select("index")
                .unique()
            )
            .to_series()
            .to_numpy()
        )
        other_antigen_tcr_indices_boolmask[other_antigen_tcr_indices] = True

        # tcr_1_seq = tcr_other_antigen.select("tcr_1_seq").item()
        # tcr_2_seq = tcr_other_antigen.select("tcr_2_seq").item()

        # query = peptide + mhc_1_seq + mhc_2_seq + tcr_1_seq + tcr_2_seq

        # # negs must be unqiue
        # if query in found_set:
        #     print("Miss")
        #     continue

        # target_row = tcr_other_antigen.select("index").item()

        focal_antigen_tcr_indices_boolmask = np.zeros(
            dist_mtx.shape[0], dtype=np.bool
        )
        focal_antigen_tcr_indices = (
            (
                df_with_idx.join(antigen, on=FORMAT_ANTIGEN_COLS, how="inner")
                .select("index")
                .unique()
            )
            .to_series()
            .to_numpy()
        )
        focal_antigen_tcr_indices_boolmask[focal_antigen_tcr_indices] = True

        noncognate_tcr_indices = other_antigen_tcr_indices[
            np.all(
                dist_mtx[other_antigen_tcr_indices_boolmask][
                    :, focal_antigen_tcr_indices_boolmask
                ]
                > tcrdist_thresh,
                axis=1,
            )
        ]

        tmp_idx = pl.DataFrame({"index": noncognate_tcr_indices})

        # # we only care about distance to binding TCRs
        # # for the focal antigen
        # tcr_hits = [
        #     idx
        #     for idx in np.argwhere(dist_mtx[target_row] <= tcrdist_thresh)[
        #         :, 0
        #     ]
        #     if idx in focal_antigen_tcr_indices
        # ]

        # if len(tcr_hits) != 0:
        #     print("Miss")
        #     continue

        tmp_df = (
            df_with_idx.join(tmp_idx, on="index")
            .select(FORMAT_TCR_COLS + TCRDIST_COLS + FORMAT_ANTIGEN_COLS)
            .unique()
        )

        if tmp_df.join(antigen, on=FORMAT_ANTIGEN_COLS).height != 0:
            raise ValueError

        # not suspect positive
        tmp_dfs.append(
            antigen.join(
                df_with_idx.join(tmp_idx, on="index")
                .rename(SOURCE_RENAME_DICT)
                .select(FORMAT_TCR_COLS + TCRDIST_COLS + SOURCE_ANTIGEN_COLS)
                .group_by(FORMAT_TCR_COLS + SOURCE_ANTIGEN_COLS)
                .agg(pl.col(colname).first() for colname in TCRDIST_COLS)
                .unique(),
                how="cross",
            )
        )
        # neg_found += 1
        # found_set.add(query)

    noncognate = pl.concat(tmp_dfs, how="vertical")
    noncognate = generate_job_name(noncognate)
    noncognate = noncognate.with_columns(pl.lit(False).alias("cognate"))
    noncognate = (
        noncognate.select(FORMAT_COLS + TCRDIST_COLS + SOURCE_ANTIGEN_COLS)
        .group_by(FORMAT_COLS)
        .agg(
            [pl.col(colname).first() for colname in TCRDIST_COLS]
            + SOURCE_ANTIGEN_COLS
        )
        .explode(SOURCE_ANTIGEN_COLS)
    ).select(FORMAT_COLS + TCRDIST_COLS + SOURCE_ANTIGEN_COLS)
    return noncognate


def generate_negatives_antigen_matched(
    df, tcrdist_thresh=120, pmhc_edit_thresh=3, n_neg=1, pos=None, dist=None
):
    """
    DF must contain FORMAT_COLS as well as TCRDIST_COLS

    Rows must be totally unqiue
    """

    tmp_dfs = []

    all_antigens = df.select(FORMAT_ANTIGEN_COLS).unique()

    if dist is None:
        all_tcrs = df.select(FORMAT_TCR_COLS + TCRDIST_COLS).unique()
        all_tcrs, dist_mtx = pw_tcrdist(all_tcrs)
        df_with_idx = df.join(all_tcrs, on=FORMAT_TCR_COLS + TCRDIST_COLS)

    else:
        df_with_idx = df
        dist_mtx = dist

    found_set = set()

    iter_df = (
        df.select(FORMAT_COLS) if pos is None else pos.select(FORMAT_COLS)
    )

    for row in iter_df.iter_rows(named=True):

        antigen = pl.DataFrame(row).select(FORMAT_ANTIGEN_COLS)

        # tcrs_for_antigen = (
        #     df.join(antigen, on=FORMAT_ANTIGEN_COLS, how="inner")
        #     .select("index")
        #     .unique()
        #     .to_series()
        #     .to_numpy()
        # )

        # don't favor TCRs from a particularly well-represented antigen
        # do this by first randomly sampling an antigen,
        # then choosing a random TCR from it

        neg_found = 0

        while neg_found < n_neg:

            other_antigens = all_antigens.join(
                antigen, on=FORMAT_ANTIGEN_COLS, how="anti"
            )

            other_antigen = other_antigens.sample(n=1, shuffle=True)

            # do we reject based on antigen distance?

            peptide = antigen.select("peptide").item()
            mhc_1_seq = antigen.select("mhc_1_seq").item()
            mhc_2_seq = antigen.select("mhc_2_seq").item()

            antigen_distance = (
                other_antigen.with_columns(
                    pl.struct(
                        pl.col("peptide"),
                        pl.col("mhc_1_seq"),
                        pl.col("mhc_2_seq"),
                    )
                    .map_elements(
                        lambda x: (
                            editdistance.eval(x["mhc_1_seq"], mhc_1_seq)
                        )
                        + (editdistance.eval(x["mhc_2_seq"], mhc_2_seq))
                        + (editdistance.eval(x["peptide"], peptide)),
                        return_dtype=pl.Int64,
                    )
                    .alias("edit_dist")
                )
                .select("edit_dist")
                .item()
            )

            if antigen_distance <= pmhc_edit_thresh:
                # reject, antigen too close
                continue

            # select a TCR
            tcr_other_antigen = (
                df_with_idx.join(
                    other_antigen, on=FORMAT_ANTIGEN_COLS, how="inner"
                )
                .select(FORMAT_TCR_COLS + TCRDIST_COLS + ["index"])
                .unique()
            ).sample(n=1, shuffle=True)

            tcr_1_seq = tcr_other_antigen.select("tcr_1_seq").item()
            tcr_2_seq = tcr_other_antigen.select("tcr_2_seq").item()

            query = peptide + mhc_1_seq + mhc_2_seq + tcr_1_seq + tcr_2_seq

            # negs must be unqiue
            if query in found_set:
                # print("Miss")
                continue

            target_row = tcr_other_antigen.select("index").item()

            focal_antigen_tcr_indices = set(
                (
                    (
                        df_with_idx.join(
                            antigen, on=FORMAT_ANTIGEN_COLS, how="inner"
                        )
                        .select("index")
                        .unique()
                    )
                    .to_series()
                    .to_list()
                )
            )

            # we only care about distance to binding TCRs
            # for the focal antigen
            tcr_hits = [
                idx
                for idx in np.argwhere(dist_mtx[target_row] <= tcrdist_thresh)[
                    :, 0
                ]
                if idx in focal_antigen_tcr_indices
            ]

            if len(tcr_hits) != 0:
                print("Miss")
                continue

            # not suspect positive
            tmp_dfs.append(
                antigen.join(
                    tcr_other_antigen.select(pl.exclude("index")),
                    how="cross",
                )
            )
            neg_found += 1
            found_set.add(query)

    noncognate = pl.concat(tmp_dfs, how="vertical")
    noncognate = generate_job_name(noncognate)
    noncognate = noncognate.with_columns(pl.lit(False).alias("cognate"))
    noncognate = noncognate.select(FORMAT_COLS + TCRDIST_COLS)
    return noncognate


def generate_negatives_random(df):

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


def per_antigen_diversity(df, species=None, chains=None, thresh=120):

    if species is None:
        species = df[0].select("tcr_1_species").item()

    if chains is None:
        chain_1 = df[0].select("tcr_1_chain").item()
        chain_2 = df[0].select("tcr_2_chain").item()

        chains = [chain_1, chain_2]

    by_antigen_list = df.partition_by(
        FORMAT_ANTIGEN_COLS,
    )

    tmp_dfs = []

    for antigen_df in by_antigen_list:

        tcr_df_pd = (
            antigen_df.select(FORMAT_TCR_COLS + TCRDIST_COLS)
            .unique()
            .with_columns(
                pl.col("tcr_1_cdr_3").alias("cdr3_a_aa"),
                pl.col("tcr_2_cdr_3").alias("cdr3_b_aa"),
                pl.col("tcr_1_v_gene").alias("v_a_gene"),
                pl.col("tcr_2_v_gene").alias("v_b_gene"),
                pl.col("tcr_1_j_gene").alias("j_a_gene"),
                pl.col("tcr_2_j_gene").alias("j_b_gene"),
                pl.lit(1).alias("count"),
            )
            .to_pandas()
        )

        tr = TCRrep(
            cell_df=tcr_df_pd,
            organism=species,
            chains=chains,
            db_file="alphabeta_gammadelta_db.tsv",
        )

        # https://github.com/phbradley/TCRdock/blob/c5a7af42eeb0c2a4492a4d4fe803f1f9aafb6193/algorithms_from_the_paper.py#L5
        def compute_tcrdiv(D, sigma=120):
            """D is a symmetric matrix of TCRdist distances
            sigma is the width of the Gaussian smoothing term
            """
            N = D.shape[0]
            if N == 1:
                return None

            D = D.copy()  # don't change passed array
            D[np.arange(N), np.arange(N)] = (
                1e6  # set diagonal to a very large value
            )
            return -1 * np.log(
                np.sum(np.exp(-1 * (D / sigma) ** 2)) / (N * (N - 1))
            )

        # not a perfect measure since count is set to 1
        # fdiv = fuzzy_diversity(
        #     tr.clone_df["count"],
        #     tr.pw_alpha + tr.pw_beta,
        #     order=2,
        #     threshold=thresh,
        # )

        fdiv = compute_tcrdiv((tr.pw_alpha + tr.pw_beta).astype(np.int64))

        tmp_dfs.append(
            antigen_df.select(FORMAT_ANTIGEN_COLS)[0].with_columns(
                pl.lit(fdiv).alias("TCRdiv"),
                pl.lit(antigen_df.height).alias("TCRdiv_samples"),
            )
        )

    return pl.concat(tmp_dfs, how="vertical_relaxed")
