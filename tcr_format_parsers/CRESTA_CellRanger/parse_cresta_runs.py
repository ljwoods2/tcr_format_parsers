"""
Takes path to cellranger "outs" directory.

Overall steps:

    given clonotypes.csv, for a particular run (which is probably one donor)
    we can associate clonotype_id to a list of CDR3 amino acid sequences 
    (or nucleotide sequences). we can also get the number of these clonotype
    cells in the overall proportion and the proportion relative to the entire cell pop

    filtered_contig_annotations gives CELL barcode to clonotype id to cdr3_aa to entire sequenc

    -d /scratch/lwoods/10x/manuscript/30168neg


    may need conda-forge::gcc 
    conda install -c conda-forge libstdcxx-ng=12

Identify and remove transmembrane region from HLAs (use fixed position probably)
Identify peptide core using Alphafold (in later step)
Only include clonotypes with cutoff num cells and which are associated with 
exactly one TCR alpha and beta

Convert MHC code to sequence (this step)
Merge MHCs based on overlap (this step)
"""

import numpy as np
from pathlib import Path
import polars as pl
import argparse
from scipy.stats import fisher_exact, kruskal, mannwhitneyu
from scipy.cluster.hierarchy import linkage, fcluster
from math import comb
import dask.bag as db
from dask.distributed import Client
from difflib import SequenceMatcher
from tcr_format_parsers.common.MHCCodeConverter import (
    HLACodeWebConverter,
    DRA_EXTRACELLULAR_TOPOLOGICAL_SEQ,
    DRA_NAME,
    DQA_FOR,
)
from tcr_format_parsers.common.TCRUnique import hash_tcr_sequence
from CellRangerOutput import CellRangerOutput, FeatBCMatrix
import collections
import itertools
import tidytcells as tt

# client = Client()


def interpolate_chains(filt_annot_df):
    """For two barcodes (cells) with a matching raw_clonotype_id and chain_code,
    ensure that each barcode is associated with the exact same set of chain_codes.
    i.e. if they match on one chain code, we insert rows so that they match on all of them

    we call this "read interpolation"

    This is done for consistency with John's original code
    """
    # Gets raw clonotype ids that match on at least one chain code
    rawctid_with_match = (
        filt_annot_df.select(
            ["raw_clonotype_id", "chain_code", "barcode", "chain"]
        )
        .group_by(["raw_clonotype_id", "chain_code", "chain"])
        .agg([pl.col("barcode").count().alias("count")])
        .filter(pl.col("count") > 1)
        .select("raw_clonotype_id")
        .unique()
    )

    rawctid_chain_allbc_allccode = (
        filt_annot_df.select(
            ["raw_clonotype_id", "chain_code", "barcode", "chain"]
        )
        .join(rawctid_with_match, on="raw_clonotype_id", how="inner")
        .group_by(["raw_clonotype_id", "chain"])
        .agg(
            [
                pl.col("barcode").unique().alias("all_barcodes"),
                pl.col("chain_code").unique().alias("all_chain_codes"),
            ]
        )
    )
    # explode all associated barcodes and chaincodes for reach rawctid to interpolate
    rawctid_chain_bc_ccode_exploded = (
        rawctid_chain_allbc_allccode.explode("all_barcodes")
        .explode("all_chain_codes")
        .rename({"all_barcodes": "barcode", "all_chain_codes": "chain_code"})
    )

    # join explided df with original df
    rawctid_bc_ccode_chain = pl.concat(
        [
            rawctid_chain_bc_ccode_exploded,
            filt_annot_df.select(
                ["raw_clonotype_id", "chain_code", "barcode", "chain"]
            ),
        ]
    ).unique()
    return rawctid_bc_ccode_chain


def cluster_clonotypes(bc_chaincode_chain):
    """
    Reassign clonotypes based on hclustering that uses
    Fisher's exact test p-value as the distance metric

    Filter out:
    - singletons (cells that only contain chains that appear in only 1 cell),
    - doublets (cells that contain chains from more than 1 (reassigned) clonotype)

    Parameters
    ----------
    bc_chaincode_chain : pl.LazyFrame
        A LazyFrame with columns "barcode", "chain_code", "chain"
    """
    chaincode_bclist_bccount = bc_chaincode_chain.group_by("chain_code").agg(
        pl.col("barcode").alias("bc_list"),
        pl.col("barcode").count().alias("bc_count"),
    )
    # filter out chains that only appear in 1 cell
    # these will be assigned a placeholder clonotype of -1 later
    chaincode_bclist_bccount_filtered = chaincode_bclist_bccount.filter(
        pl.col("bc_count") > 1
    )

    # Get total number of cells (number of barcodes)
    n_cells = bc_chaincode_chain.select("barcode").unique().collect().height

    # sort for consistency
    chaincode_bclist_bccount_filtered = (
        chaincode_bclist_bccount_filtered.collect().sort("chain_code")
    )

    n_unique_chains = chaincode_bclist_bccount_filtered.height

    bc_col = chaincode_bclist_bccount_filtered.select("bc_list")

    bc_sets = [set(bc) for bc in bc_col.to_series().to_list()]
    bc_set_sizes = [len(bc_set) for bc_set in bc_sets]

    pairs = [
        (i, j)
        for i in range(n_unique_chains - 1)
        for j in range(i + 1, n_unique_chains)
    ]

    pair_bag = db.from_sequence(pairs, npartitions=8)

    def fisher_for_pair(pair, bc_sets, bc_sizes, n_cells):
        i, j = pair
        set_i, set_j = bc_sets[i], bc_sets[j]
        c1 = len(set_i.intersection(set_j))

        if c1 == 0:
            return None

        c2 = bc_sizes[i] - c1
        c3 = bc_sizes[j] - c1
        c4 = n_cells - (c1 + c2 + c3)
        _, p_val = fisher_exact([[c1, c2], [c3, c4]])
        return (i, j, p_val)

    p_vals_unordered = (
        pair_bag.map(
            fisher_for_pair,
            bc_sets=bc_sets,
            bc_sizes=bc_set_sizes,
            n_cells=n_cells,
        )
        .filter(lambda x: x is not None)
        .compute()
    )

    indexing_choose_constant = comb(n_unique_chains, 2)
    indexing_choose_i_array = [
        comb(n_unique_chains - i, 2) for i in range(len(bc_sets) - 1)
    ]
    # use floor val to convert to int
    # distance vector more efficent than nxn matrix
    p_vals = np.ones(
        n_unique_chains * (n_unique_chains - 1) // 2, dtype=np.float64
    )

    for i, j, p_val in p_vals_unordered:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html
        p_val_idx = (
            indexing_choose_constant - indexing_choose_i_array[i] + (j - i - 1)
        )
        p_vals[p_val_idx] = p_val

    z = linkage(p_vals, method="complete")

    # clusters is a length n_unique_chains array where each element is the cluster number
    # of the TCR seq
    out_clonotypes = fcluster(z, t=1e-6, criterion="distance")

    chaincode_bclist_bccount_outclonotype_filtered = (
        chaincode_bclist_bccount_filtered.with_columns(
            pl.Series("out_clonotype", out_clonotypes)
        ).lazy()
    )

    # assign clonotypes to barcodes
    # any chain that we didn't give a clonotype because it
    # only appeared in 1 cell
    # is given a clonotype of pl.UInt32.max
    # so that it can be considered in finding doublets
    bc_chaincode_chain_outclonotype = bc_chaincode_chain.join(
        chaincode_bclist_bccount_outclonotype_filtered.select(
            ["chain_code", "out_clonotype"]
        ),
        on=["chain_code"],
        how="left",
    ).fill_null(pl.UInt32.max())

    # find doublets- cells containing more than one clonotype
    doublet_bcs = (
        bc_chaincode_chain_outclonotype.select(["barcode", "out_clonotype"])
        .group_by("barcode")
        .agg([pl.col("out_clonotype").n_unique().alias("n_clonotypes")])
        .filter(pl.col("n_clonotypes") > 1)
    ).select("barcode")

    # find singletons- cells containing only chains wtih clonotype pl.UInt32.max
    # meaning those chains only appeared in that cell
    singleton_bcs = (
        (
            bc_chaincode_chain_outclonotype.select(
                ["barcode", "out_clonotype"]
            )
            .group_by("barcode")
            .agg([pl.col("out_clonotype").n_unique().alias("n_clonotypes")])
            .filter(pl.col("n_clonotypes") == 1)
            .select("barcode")
        )
        .join(
            bc_chaincode_chain_outclonotype.select(
                ["barcode", "out_clonotype"]
            ).unique(),
            on="barcode",
            how="inner",
        )
        .filter(pl.col("out_clonotype") == pl.UInt32.max())
        .select("barcode")
    )

    # filter out singletons and doublets
    bc_chaincode_chain_outclonotype = bc_chaincode_chain_outclonotype.join(
        doublet_bcs, on="barcode", how="anti"
    ).join(singleton_bcs, on="barcode", how="anti")

    # recombine TRA and TRB chains, filtering out clonotypes associated with >1 chain
    bc_tra_outclonotype = (
        bc_chaincode_chain_outclonotype.filter(pl.col("chain") == "TRA")
        .select(["barcode", "out_clonotype", "chain_code"])
        .rename({"chain_code": "TRA"})
    )
    bc_tra_outclonotype = bc_tra_outclonotype.join(
        bc_tra_outclonotype.select("out_clonotype", "TRA")
        .unique()
        .group_by("out_clonotype")
        .agg(pl.col("TRA").count())
        .filter(pl.col("TRA") == 1)
        .select("out_clonotype"),
        on="out_clonotype",
        how="inner",
    )

    bc_trb_outclonotype = (
        bc_chaincode_chain_outclonotype.filter(pl.col("chain") == "TRB")
        .select(["barcode", "out_clonotype", "chain_code"])
        .rename({"chain_code": "TRB"})
    )

    bc_trb_outclonotype = bc_trb_outclonotype.join(
        bc_trb_outclonotype.select("out_clonotype", "TRB")
        .unique()
        .group_by("out_clonotype")
        .agg(pl.col("TRB").count())
        .filter(pl.col("TRB") == 1)
        .select("out_clonotype"),
        on="out_clonotype",
        how="inner",
    )

    bc_tra_trb_outclonotype = bc_tra_outclonotype.join(
        bc_trb_outclonotype,
        on=["barcode", "out_clonotype"],
        how="inner",
    )

    # count number of cells per clonotype
    outclonotype_newclonotype_frequency = (
        (
            bc_tra_trb_outclonotype.select(["out_clonotype", "barcode"])
            .unique()
            .group_by("out_clonotype")
            .agg(pl.col("barcode").count().alias("clonotype_frequency"))
        )
        .sort("clonotype_frequency", descending=True)
        .collect()
    )

    outclonotype_newclonotype_frequency = (
        outclonotype_newclonotype_frequency.with_columns(
            pl.Series(
                "new_clonotype",
                np.arange(
                    outclonotype_newclonotype_frequency.height,
                    dtype=np.int32,
                ),
            )
        )
    ).lazy()

    clonotypeid_frequency = outclonotype_newclonotype_frequency.select(
        ["new_clonotype", "clonotype_frequency"]
    ).rename({"new_clonotype": "clonotype_id"})

    bc_tra_trb_clonotypeid = (
        bc_tra_trb_outclonotype.join(
            outclonotype_newclonotype_frequency,
            on="out_clonotype",
            how="inner",
        )
        .select(["barcode", "TRA", "TRB", "new_clonotype"])
        .rename({"new_clonotype": "clonotype_id"})
    )

    return bc_tra_trb_clonotypeid, clonotypeid_frequency


def construct_cognate_df(cro_list):

    tmp_dfs = []
    withvdj_norm_matrices = []
    withvdj_matrices = []
    filt_annot_dfs = []
    featnames_idx_df = None
    offset = 0

    for cro in cro_list:
        filt_annot_df = cro.get_filtered_contig_df()

        # May have uneccesary logic here- doesn't hurt
        filt_annot_df = filt_annot_df.filter(
            (~pl.col("raw_clonotype_id").is_in(["None", ""]))
            & (pl.col("productive") == "true")
            & (~pl.col("raw_consensus_id").is_in(["None", ""])),
        )

        # Create a TRA/B chain_code column
        filt_annot_df = filt_annot_df.with_columns(
            pl.concat_str(
                ["v_gene", "cdr3", "j_gene", "cdr3_nt"],
                separator=":",
            ).alias("chain_code")
        )

        # can remove later- just rename filt_annot_df to rawctid_...
        rawctid_bc_chaincode_chain = interpolate_chains(filt_annot_df)

        # Barcode, TRA, TRB
        bc_chaincode_chain_alpha = (
            rawctid_bc_chaincode_chain.filter(pl.col("chain") == "TRA").select(
                ["barcode", "chain_code", "chain"]
            )
        ).unique()
        bc_chaincode_chain_beta = (
            rawctid_bc_chaincode_chain.filter(pl.col("chain") == "TRB").select(
                ["barcode", "chain_code", "chain"]
            )
        ).unique()

        # all barcodes that contain VDJ evidence
        # we only care about the barcodes for which
        # we have feature counts & VDJ annot

        bc_chaincode_chain_with_vdj = pl.concat(
            [bc_chaincode_chain_alpha, bc_chaincode_chain_beta], how="vertical"
        )

        # remove barcodes without VDJ annotations from norm mtx
        original_matrix = cro.get_featbcmatrix_obj("orig")

        withvdj_matrix = original_matrix.create_child_matrix(
            "withvdj",
            bc_df=bc_chaincode_chain_with_vdj.select("barcode")
            .unique()
            .collect(),
        )

        # These index dfs should be the same across runs
        if featnames_idx_df is None:
            featnames_idx_df = withvdj_matrix.featnames_idx_df

        # then normalize feature counts
        withvdj_norm_matrix = FeatBCMatrix(
            "withvdj_norm",
            normalize_feature_counts(withvdj_matrix),
            withvdj_matrix.featnames_idx_df.rename(
                {withvdj_matrix.idx_name: "withvdj_norm_idx"}
            ),
            withvdj_matrix.bc_idx_df.rename(
                {withvdj_matrix.idx_name: "withvdj_norm_idx"},
            ),
        )

        # only reclonotype rows with feature counts and vdj annot
        # replace barcode with unique integer
        # that doubles as index into concatenated np array
        bc_chaincode_chain_idx = (
            withvdj_matrix.bc_idx_df.lazy()
            .join(bc_chaincode_chain_with_vdj, on="barcode", how="inner")
            .select("barcode", "withvdj_idx", "chain_code", "chain")
        ).with_columns((pl.col("withvdj_idx") + offset).alias("withvdj_idx"))

        filt_annot_df = (
            filt_annot_df.join(
                bc_chaincode_chain_idx.select("barcode", "withvdj_idx"),
                on="barcode",
                how="inner",
            )
            .select(pl.exclude("barcode"))
            .with_columns(pl.col("withvdj_idx").alias("barcode"))
            .select(pl.exclude("withvdj_idx"))
        )

        bc_chaincode_chain = (
            bc_chaincode_chain_idx.select(pl.exclude("barcode"))
            .with_columns(pl.col("withvdj_idx").alias("barcode"))
            .select(pl.exclude("withvdj_idx"))
        )

        filt_annot_dfs.append(filt_annot_df)
        withvdj_matrices.append(withvdj_matrix)
        withvdj_norm_matrices.append(withvdj_norm_matrix)
        tmp_dfs.append(bc_chaincode_chain)

        offset += withvdj_matrix.mtx.shape[1]

    # master filt_annot_df
    filt_annot_df = pl.concat(filt_annot_dfs, how="vertical")

    # combine withvdj/withvdjnorm matrices
    withvdj_matrix_ndarr = np.concatenate(
        [fbcm.mtx for fbcm in withvdj_matrices]
    )
    withvdj_norm_matrix_ndarr = np.concatenate(
        [fbcm.mtx for fbcm in withvdj_norm_matrices]
    )

    bc_chaincode_chain = pl.concat(tmp_dfs, how="vertical")
    bc_idx_df = (
        bc_chaincode_chain.select("barcode")
        .with_columns(pl.col("barcode").alias("withvdj_idx"))
        .collect()
    )

    withvdj_matrix = FeatBCMatrix(
        "withvdj", withvdj_matrix_ndarr, featnames_idx_df, bc_idx_df
    )

    withvdj_norm_matrix = FeatBCMatrix(
        "withvdj_norm",
        withvdj_norm_matrix_ndarr,
        featnames_idx_df.rename({"withvdj_idx": "withvdj_norm_idx"}),
        bc_idx_df.rename({"withvdj_idx": "withvdj_norm_idx"}),
    )

    # CellRanger known limitation is that when 1 cell contains
    # (only) a TRA which exactly matches a TRA from another cell
    # which contains both a TRA and TRB, the cells might not
    # be called as the same clonotype. This algorithm
    # will reassign clonotypes based on Fisher's exact test
    bc_tra_trb_clonotypeid, clonotypeid_frequency = cluster_clonotypes(
        bc_chaincode_chain
    )

    # filter out clonotypes by frequency
    # singletons and doublets have been removed
    bc_clonotypeid_tmp = (
        bc_tra_trb_clonotypeid.join(
            clonotypeid_frequency.filter(pl.col("clonotype_frequency") >= 3),
            on="clonotype_id",
            how="inner",
        )
        .select(["barcode", "clonotype_id"])
        .unique()
        .collect()
    )

    # find mhcs with a max count >=2 in the unnormalized data
    feature_maxes = np.max(withvdj_matrix.mtx, axis=1)
    feature_indices = np.where(feature_maxes >= 2)[0]
    fname_df = pl.DataFrame(
        {
            "feature_name": withvdj_matrix.get_featnames_ndarr()[
                feature_indices
            ],
        }
    )

    withvdjfilt_matrix = withvdj_matrix.create_child_matrix(
        "withvdj_filt",
        bc_df=bc_clonotypeid_tmp,
        bc_cols=["clonotype_id"],
        featnames_df=fname_df,
    )

    # child matx of withvdj_norm with only 'filt' barcodes and focal features

    focalfeatnames = get_focal_features(
        withvdjfilt_matrix,
    )

    bc_ctypeid_filt = withvdjfilt_matrix.bc_idx_df.select(
        ["barcode", "clonotype_id"]
    )

    cognate_matrix = withvdj_norm_matrix.create_child_matrix(
        "cognate",
        bc_df=bc_ctypeid_filt,
        bc_cols=["clonotype_id"],
        featnames_df=focalfeatnames,
    )

    ctypeid_featname = find_binding_features_and_clonotypes(
        cognate_matrix,
    )

    featname_mhc1_mhc2_peptide_ctypeid = extract_cresta_peptide_mhc_seqs(
        ctypeid_featname.select("feature_name").unique(),
    ).join(ctypeid_featname, on="feature_name", how="inner")

    mhc1 = featname_mhc1_mhc2_peptide_ctypeid.select(
        ["mhc_1_seq", "mhc_1_name", "mhc_1_type"]
    ).unique()

    mhc2 = featname_mhc1_mhc2_peptide_ctypeid.select(
        ["mhc_2_seq", "mhc_2_name", "mhc_2_type"]
    ).unique()

    # if two hla,hlb,peptide rows are considered matching (if they have overlapping peptides)
    # then we combine them into one row, which we consider to bind with the superset
    # of the peptides that the two rows bind with
    featname_mhc1_mhc2_peptide_ctypeid = (
        combine_features(featname_mhc1_mhc2_peptide_ctypeid)
        .join(mhc1, on="mhc_1_name", how="inner")
        .join(mhc2, on="mhc_2_name", how="inner")
    )

    ctypeid_mhc1_mhc2_peptide = featname_mhc1_mhc2_peptide_ctypeid.select(
        [
            "clonotype_id",
            "mhc_1_seq",
            "mhc_1_name",
            "mhc_1_type",
            "mhc_2_seq",
            "mhc_2_name",
            "mhc_2_type",
            "peptide",
        ]
    ).unique()

    # for now, if two barcodes contain the same chain_code, we consider them to
    # have the same ACTUAL CHAIN, which is not true- their FWR or CDR1/2 regions can differ
    # (though this is rare)
    ctypeid_tcrseq_chain = (
        filt_annot_df.join(bc_ctypeid_filt.lazy(), on="barcode", how="inner")
        .with_columns(
            pl.concat_str(
                ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"],
            ).alias("tcr_seq")
        )
        .select(
            "clonotype_id", "tcr_seq", "v_gene", "j_gene", "cdr3_nt", "chain"
        )
        .group_by("v_gene", "j_gene", "cdr3_nt", "clonotype_id", "chain")
        .agg(pl.col("tcr_seq").first())
        .select("clonotype_id", "tcr_seq", "chain")
    )
    # this only makes sense if we only allow clonotypes with exactly 1
    # alpha and 1 beta and filter out singletons + doublets
    ctypeid_tcr1 = (
        ctypeid_tcrseq_chain.filter(pl.col("chain") == "TRA")
        .select("clonotype_id", "tcr_seq")
        .rename({"tcr_seq": "tcr_1_seq"})
    )

    ctypeid_tcr2 = (
        ctypeid_tcrseq_chain.filter(pl.col("chain") == "TRB")
        .select("clonotype_id", "tcr_seq")
        .rename({"tcr_seq": "tcr_2_seq"})
    )

    # this should be a strict subset of bc_tra_trb_clonotypeid
    ctypeid_tcra_tcrb = ctypeid_tcr1.join(
        ctypeid_tcr2, on="clonotype_id", how="inner"
    )

    tcr1_tcr1_mhc1_mhc2_peptide = (
        (
            ctypeid_tcra_tcrb.join(
                ctypeid_mhc1_mhc2_peptide.lazy(),
                on="clonotype_id",
                how="inner",
            )
        )
        .select(
            "tcr_1_seq",
            "tcr_2_seq",
            "mhc_1_seq",
            "mhc_1_name",
            "mhc_1_type",
            "mhc_2_seq",
            "mhc_2_name",
            "mhc_2_type",
            "peptide",
        )
        .with_columns(
            pl.lit("tcr_alpha").alias("tcr_1_type"),
            pl.lit("tcr_beta").alias("tcr_2_type"),
            pl.col("tcr_1_seq")
            .map_elements(
                lambda x: hash_tcr_sequence(x, "md5"), return_dtype=pl.String
            )
            .alias("tcr_1_name"),
            pl.col("tcr_2_seq")
            .map_elements(
                lambda x: hash_tcr_sequence(x, "md5"), return_dtype=pl.String
            )
            .alias("tcr_2_name"),
        )
    )

    return tcr1_tcr1_mhc1_mhc2_peptide


def combine_features(featname_mhc1_mhc2_peptide_ctypeid, overlap_thresh=9):

    mhc1name_mhc2name_plist = (
        featname_mhc1_mhc2_peptide_ctypeid.select(
            ["mhc_1_name", "mhc_2_name", "peptide"]
        )
        .group_by(["mhc_1_name", "mhc_2_name"])
        .agg(
            [
                pl.col("peptide").alias("peptide_list"),
            ]
        )
    )

    hla_a_names = mhc1name_mhc2name_plist.select("mhc_1_name").to_series()
    hla_b_names = mhc1name_mhc2name_plist.select("mhc_2_name").to_series()
    peptide_lists = mhc1name_mhc2name_plist.select("peptide_list").to_series()

    collapse_hla_bs = []
    collapse_hla_as = []
    collapse_peptides = []
    collapse_ctypids = []

    for i in range(len(hla_a_names)):
        hla_a = hla_a_names[i]
        hla_b = hla_b_names[i]
        peptide_list = peptide_lists[i]

        # keep a row if there's only one peptide
        # if len(peptide_list) == 1:
        #     # collapse_hla_bs.append(hla_b)
        #     # collapse_hla_as.append(hla_a)
        #     # collapse_peptides.append(peptide_list[0])
        #     # collapse_ctypids.append(clonotype_id_list[0].to_list())
        #     continue

        # all_peptides = set(p for p in peptide_list)

        match_dict = dict(
            (peptide, set([peptide])) for peptide in peptide_list
        )

        def merge_keys(d):
            merge_found = False
            overlap = None
            k1, k2 = None, None
            for k1, k2 in itertools.combinations(d, 2):
                m = SequenceMatcher(
                    None, k1, k2, autojunk=False
                ).find_longest_match()
                overlap_len = m.size
                overlap = k1[m.a : m.a + m.size]
                if overlap_len >= overlap_thresh:
                    merge_found = True
                    break

            if merge_found:
                d[overlap] = d[k1].union(d[k2])
                del d[k1]
                del d[k2]
                return merge_keys(d)

            return d

        match_dict = merge_keys(match_dict)

        # if the indices of any two keys overlap, raise an error
        # this means that A and B overlap and B and C overlap
        # but A and C do not overlap for some A, B, C peptides

        # otherwise condense all rows in each match except the first one
        seen = set()
        for k in match_dict.keys():
            v = match_dict[k]
            for pep in v:
                if pep in seen:
                    raise ValueError(
                        f"Complex overlap for peptide {pep}."
                        f"for HLAA {hla_a} and HLAB {hla_b}. Manually review"
                    )
            seen.update(v)
            keep = min(v)
            superset_ctypeids = (
                featname_mhc1_mhc2_peptide_ctypeid.filter(
                    (pl.col("mhc_1_name") == hla_a)
                    & (pl.col("mhc_2_name") == hla_b)
                    & (pl.col("peptide").is_in(v))
                )
                .select("peptide", "clonotype_id")
                .group_by("peptide")
                .agg(
                    [
                        pl.col("clonotype_id").alias("clonotype_id_list"),
                    ]
                )
                .select("clonotype_id_list")
                .to_series()
                .to_list()
            )
            # ensure peptides all share at least one clonotype
            # else raise an error as it doesn't make sense to merge them
            tmp = set(superset_ctypeids[0])
            for i in range(1, len(superset_ctypeids)):
                if len(tmp.intersection(set(superset_ctypeids[i]))) == 0:
                    raise ValueError(
                        f"Peptides {v[0]} and {v[i]} do not share any clonotypes"
                    )
            superset_ctypeids = [x for xs in superset_ctypeids for x in xs]

            collapse_hla_bs.append(hla_b)
            collapse_hla_as.append(hla_a)
            collapse_peptides.append(keep)
            collapse_ctypids.append(
                np.array(superset_ctypeids, dtype=np.int32)
            )

    mhc1_mhc2_peptide_ctypids = pl.DataFrame(
        {
            "mhc_1_name": collapse_hla_as,
            "mhc_2_name": collapse_hla_bs,
            "peptide": collapse_peptides,
            "clonotype_id": collapse_ctypids,
        }
    )

    mhc1_mhc2_peptide_ctypid = mhc1_mhc2_peptide_ctypids.explode(
        "clonotype_id"
    )
    return mhc1_mhc2_peptide_ctypid


def get_focal_features(
    featbc_mtx,
):

    mtx = featbc_mtx.mtx
    feature_name_ndarr = featbc_mtx.get_featnames_ndarr()
    clonotypes_ndarr = featbc_mtx.get_ndarr_for_col("bc", "clonotype_id")

    p_vals = []
    fold_changes = []

    # for each fbc, perform a kruskal test on clonotype ID vs read count
    for idx in range(mtx.shape[0]):
        read_count = mtx[idx]

        clonotype_rc_list = (
            pl.DataFrame(
                {
                    "read_count": read_count,
                    "clonotype_id": clonotypes_ndarr,
                }
            )
            .group_by("clonotype_id")
            .agg([pl.col("read_count").alias("read_count_list")])
        )
        pvalue = kruskal(
            *clonotype_rc_list.select("read_count_list").to_series().to_list()
        ).pvalue
        if pvalue == 0.0:
            pvalue = 1e-250

        clonotype_rc_list_medians = clonotype_rc_list.with_columns(
            median=pl.col("read_count_list").list.median()
        )

        median_max = clonotype_rc_list_medians.select("median").max().item()
        median_min = clonotype_rc_list_medians.select("median").min().item()
        ratio_of_extremes = 1 + median_max / 1 + median_min

        p_vals.append(pvalue)
        fold_changes.append(ratio_of_extremes)

    fname_pval_fchange = pl.DataFrame(
        {
            "feature_name": feature_name_ndarr,
            "p_val": p_vals,
            "fold_change": fold_changes,
        }
    )

    focalfnames = fname_pval_fchange.filter(
        ((pl.col("p_val").log(base=10) * -1) > 10)
        & (
            (pl.col("fold_change").log(base=2) > 0.3)
            | (pl.col("fold_change").log(base=2) < -0.3)
        )
    ).select("feature_name")

    return focalfnames


def normalize_feature_counts(featbc_mtx):
    mtx_ndarr = featbc_mtx.mtx

    n_barcodes = mtx_ndarr.shape[1]
    n_features = mtx_ndarr.shape[0]

    mtx_ndarr = mtx_ndarr + 1
    mtx_ndarr = mtx_ndarr.astype(np.float32)

    # Per-FBC relative abundance
    # calculate sum per fbc
    feature_count_sums = mtx_ndarr.sum(axis=1)

    # divide each fbc by its barcode count sum to get relative abundance
    # then multiply by n_barcodes so sum adds to n_barcodes
    mtx_ndarr[:] = (
        np.divide(mtx_ndarr, feature_count_sums[:, np.newaxis]) * n_barcodes
    )

    # Per-cell norm
    mtx_median_norm = np.empty(mtx_ndarr.shape)

    row_arange = np.arange(0, n_features)
    for i in range(n_features):
        # exclude the current row in calculating the col-wise medians
        col_wise_medians = np.median(
            mtx_ndarr[row_arange[row_arange != i]], axis=0
        )
        mtx_median_norm[i] = mtx_ndarr[i] / col_wise_medians

    # Apply a ceiling
    # parameterize later if needed
    top_val = n_barcodes // 1000

    for i in range(n_features):
        # get the top_val indices of the top_val largest elements
        # O(nlogn)
        k_largest_ind = np.argpartition(mtx_median_norm[i], -top_val)[
            -top_val:
        ]
        # gives indices of k largets from smallest to largest
        k_largest_ind[:] = k_largest_ind[
            np.argsort(mtx_median_norm[i][k_largest_ind])
        ]
        mtx_median_norm[i][k_largest_ind[1:]] = mtx_median_norm[i][
            k_largest_ind[0]
        ]

    # normalize FBCs by max
    row_wise_fbc_maxes = np.max(mtx_median_norm[:], axis=1) + 1
    mtx_norm = mtx_median_norm / row_wise_fbc_maxes[:, np.newaxis]
    return mtx_norm


def find_binding_features_and_clonotypes(featbc_mtx):
    # find_binding_features_and_clonotypes
    clonotype_cltypeidxls = (
        featbc_mtx.bc_idx_df.select(["clonotype_id", featbc_mtx.idx_name])
        .group_by("clonotype_id")
        .agg([pl.col(featbc_mtx.idx_name).alias("index_list")])
    )
    clonotypes = clonotype_cltypeidxls.select("clonotype_id").to_series()
    cltypeidx_lists = clonotype_cltypeidxls.select("index_list").to_series()

    # for each feature, find the range
    feature_names = featbc_mtx.get_featnames_ndarr()

    # idxs of 40th and 60th percentile in sorted array's 2nd dimension
    p1 = int((0.4 * (featbc_mtx.mtx.shape[1])) - 1)
    p2 = int(0.6 * (featbc_mtx.mtx.shape[1]))

    tmp_dfs = []

    for i in range(featbc_mtx.mtx.shape[0]):
        # feature_idx = feature_idxs[i]
        feature_name = feature_names[i]

        # this should be filtered down to JUST the remaining barcodes
        baseline_counts = np.round(np.sort(featbc_mtx.mtx[i])[p1:p2], 3)
        baseline_mean = np.mean(baseline_counts)

        p_vals = []
        mean_ratios = []

        for j in range(len(clonotypes)):
            cltypeidx_list = cltypeidx_lists[j]
            # REMOVE ROUNDING- reduces number of binders
            per_clonotype_counts = np.round(
                featbc_mtx.mtx[i][cltypeidx_list], 3
            )
            # unpaired wilcoxon is equivalent to Mann-Whitney U test
            p_val = mannwhitneyu(
                per_clonotype_counts, baseline_counts, alternative="greater"
            ).pvalue
            mean_ratio = np.mean(per_clonotype_counts) / baseline_mean

            p_vals.append(p_val)
            mean_ratios.append(mean_ratio)

        tmp_df = pl.DataFrame(
            {
                "clonotype_id": clonotypes,
                "p_val": p_vals,
                "fold_change": mean_ratios,
                "feature_name": feature_name,
            }
        )
        tmp_dfs.append(tmp_df)

    clonotypeid_fname_pval_fchange = pl.concat(tmp_dfs, how="vertical")
    clonotypeid_fname_binding = clonotypeid_fname_pval_fchange.filter(
        ((pl.col("p_val").log(base=10) * -1) > 2.5)
        & (pl.col("fold_change").log(base=2) > 1.25)
    ).select(["clonotype_id", "feature_name"])

    return clonotypeid_fname_binding


KNOWN_FEATNAMES = {
    "DRB1150": "DRB1*15:03",
}


def extract_cresta_peptide_mhc_seqs(featname):
    # create code, hla alpha, hla beta, peptide df
    # merge matches
    featname_fbcid_hlacode_peptide = (
        featname.select("feature_name")
        .with_columns(
            pl.col("feature_name")
            .str.split_exact("_", 3)
            .struct.rename_fields(
                ["deleteme", "fbc_id", "mhc_code", "peptide"]
            )
            .alias("struct")
        )
        .unnest("struct")
    )
    featname_hlacode_peptide = featname_fbcid_hlacode_peptide.drop(
        ["deleteme", "fbc_id"]
    )

    # convert from fbc to official format
    mhc_codes = (
        featname_hlacode_peptide.select("mhc_code").to_series().to_list()
    )
    beta_seqs = []
    beta_names = []
    beta_types = []
    alpha_seqs = []
    alpha_names = []
    alpha_types = []
    mhc_converter = HLACodeWebConverter()

    # convert MHC code to transmembrane-removed AA sequence
    for mhc_code in mhc_codes:
        if mhc_code in KNOWN_FEATNAMES:
            officialname = KNOWN_FEATNAMES[mhc_code]
        else:
            # standardize is fine here since we guaranteed a fully-qualified name
            # in cresta
            officialname = tt.mh.standardize(
                mhc_code, precision="allele"
            ).split("HLA-")[1]

            # gene = hla_code[:4]
            # allele_group = hla_code[4:6]
            # # ignore synonymous mutations
            # allele = hla_code[6:].split(":")[0]
            # officialname = gene + "*" + allele_group + ":" + allele

        seq = mhc_converter.get_sequence(officialname, top_only=True)
        beta_seq = None
        alpha_seq = None

        if officialname.startswith("DRB"):

            alpha_seq = DRA_EXTRACELLULAR_TOPOLOGICAL_SEQ
            alpha_name = DRA_NAME
            alpha_type = "hla_II_alpha"
            beta_seq = seq
            beta_name = officialname
            beta_type = "hla_II_beta"

        elif officialname.startswith("DQB"):
            alpha_seq = mhc_converter.get_sequence(
                DQA_FOR[officialname], top_only=True
            )
            alpha_name = DQA_FOR[officialname]
            alpha_type = "hla_II_alpha"
            beta_seq = seq
            beta_name = officialname
            beta_type = "hla_II_beta"
        else:
            raise ValueError(f"Unexpected HLA gene {officialname}")

        beta_seqs.append(beta_seq)
        beta_names.append(beta_name)
        beta_types.append(beta_type)
        alpha_seqs.append(alpha_seq)
        alpha_names.append(alpha_name)
        alpha_types.append(alpha_type)

    featname_mhc1_mhc2_peptide = featname_hlacode_peptide.with_columns(
        [
            pl.Series("mhc_1_seq", alpha_seqs),
            pl.Series("mhc_1_name", alpha_names),
            pl.Series("mhc_1_type", alpha_types),
            pl.Series("mhc_2_seq", beta_seqs),
            pl.Series("mhc_2_name", beta_names),
            pl.Series("mhc_2_type", beta_types),
        ]
    )
    featname_mhc1_mhc2_peptide.drop_in_place("mhc_code")
    return featname_mhc1_mhc2_peptide


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare CRESTA data for 5LR pipeline"
    )
    parser.add_argument(
        "-d",
        "--directories",
        help="CellRanger directories to parse",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output CSV path",
        default="cresta_output.csv",
    )
    # add args for logfold change thresh and p val thresh
    # parser.add_argument(
    #     "-s2", "--structure_2", help="Second structure", required=True
    # )
    args = parser.parse_args()

    directory_paths = [d.strip() for d in args.directories.split(",")]

    cro_list = []
    donor_num = None
    for directory_path in directory_paths:
        cro = CellRangerOutput(directory_path)
        if donor_num is None:
            donor_num = cro.donor_num
        elif cro.donor_num != donor_num:
            raise ValueError(
                "This script is intended for multiple runs of the "
                f"same donor. Donors {donor_num} and {cro.donor_num} "
                "were detected."
            )
        cro_list.append(cro)
    cognate_df = construct_cognate_df(cro_list).with_columns(
        pl.lit(1, dtype=pl.Boolean).alias("cognate"),
        pl.lit(donor_num, dtype=pl.Int32).alias("participant_id"),
    )

    cognate_df.collect().write_csv(args.output)
