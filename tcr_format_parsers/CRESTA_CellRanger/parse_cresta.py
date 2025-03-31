"""
Takes path to cellranger "outs" directory.

Overall steps:

    given clonotypes.csv, for a particular run (which is probably one donor)
    we can associate clonotype_id to a list of CDR3 amino acid sequences 
    (or nucleotide sequences). we can also get the number of these clonotype
    cells in the overall proportion and the proportion relative to the entire cell pop

    filtered_contig_annotations gives CELL barcode to clonotype id to cdr3_aa to entire sequence

    -d /scratch/lwoods/10x/10x_run_043/30107neg


    -d /scratch/lwoods/10x/IM0144_10x/30168neg

    --directories /tnorth_labs/Immunology/ekelley/10x/10x_run_043/30107pos
    /tnorth_labs/Immunology/ekelley/10x/10x_run_043/30107pos/outs/per_sample_outs/30107pos

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
from CellRangerOutput import CellRangerOutput, FeatBCMatrix
import collections
import itertools

# client = Client()


def interpolate_reads(filt_annot_df):
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


def cluster_clonotypes(bc_chaincode_chain, strict=False):
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
    strict : bool, optional
        Whether to only keep clonotypes with exactly 1 alpha and 1 beta chain [[False]]

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

    # # tmp
    # chaincodes_sorted = (
    #     chaincode_bclist_bccount_filtered.select("chain_code")
    #     .collect()
    #     .to_series()
    #     .to_list()
    # )
    # p_vals_mtx = squareform(p_vals)
    # chain_1 = []
    # chain_2 = []
    # p_val_ls = []
    # for i in range(n_unique_chains - 1):
    #     for j in range(i + 1, n_unique_chains):
    #         p_val = p_vals_mtx[i, j]
    #         if p_val != 0 and p_val != 1:
    #             chain1 = chaincodes_sorted[i]
    #             chain2 = chaincodes_sorted[j]
    #             chain_1.append(chain1)
    #             chain_2.append(chain2)
    #             p_val_ls.append(p_val)
    # p_vals_df = pl.DataFrame(
    #     {
    #         "chain1": chain_1,
    #         "chain2": chain_2,
    #         "p_val": p_val_ls,
    #     }
    # )

    # for i in range(len(bc_sets) - 1):
    #     bc_set_1 = bc_sets[i]
    #     bc_set_1_size = len(bc_set_1)
    #     for j in range(i + 1, len(bc_sets)):
    #         bc_set_2 = bc_sets[j]
    #         bc_set_2_size = len(bc_set_2)

    #         c1 = len(bc_set_1.intersection(bc_set_2))
    #         c2 = bc_set_1_size - c1
    #         c3 = bc_set_2_size - c1
    #         c4 = n_cells - c1 - c2 - c3

    #         _, p_val = fisher_exact([[c1, c2], [c3, c4]])

    #         # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html
    #         p_val_idx = (
    #             indexing_choose_constant
    #             - indexing_choose_i_array[i]
    #             + (j - i - 1)
    #         )
    #         p_vals[p_val_idx] = p_val

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
    # is given a clonotype of -1
    # so that it can be considered in finding doublets
    bc_chaincode_chain_outclonotype = bc_chaincode_chain.join(
        chaincode_bclist_bccount_outclonotype_filtered.select(
            ["chain_code", "out_clonotype"]
        ),
        on=["chain_code"],
        how="left",
    ).fill_null(-1)

    # find doublets- cells containing more than one clonotype
    doublet_bcs = (
        bc_chaincode_chain_outclonotype.select(["barcode", "out_clonotype"])
        .group_by("barcode")
        .agg([pl.col("out_clonotype").n_unique().alias("n_clonotypes")])
        .filter(pl.col("n_clonotypes") > 1)
    ).select("barcode")

    # find singletons- cells containing only chains wtih clonotype -1
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
        .filter(pl.col("out_clonotype") == -1)
        .select("barcode")
    )

    # # find singletons- chains that only appear in one cell
    # singleton_chaincodes = (
    #     bc_chaincode_chain_newclonotype.select(["barcode", "chain_code"])
    #     .group_by("chain_code")
    #     .agg(pl.col("barcode").count().alias("n_barcodes"))
    #     .filter(pl.col("n_barcodes") == 1)
    # ).select("chain_code")

    # filter out singletons and doublets
    bc_chaincode_chain_outclonotype = bc_chaincode_chain_outclonotype.join(
        doublet_bcs, on="barcode", how="anti"
    ).join(singleton_bcs, on="barcode", how="anti")

    # recombine TRA and TRB chains
    bc_tra_outclonotype = (
        bc_chaincode_chain_outclonotype.filter(pl.col("chain") == "TRA")
        .select(["barcode", "out_clonotype", "chain_code"])
        .rename({"chain_code": "TRA"})
    )
    bc_trb_outclonotype = (
        bc_chaincode_chain_outclonotype.filter(pl.col("chain") == "TRB")
        .select(["barcode", "out_clonotype", "chain_code"])
        .rename({"chain_code": "TRB"})
    )

    # later, we will only care about barcode with both chains
    # and exactly one of each
    # keep this for now for validation
    bc_tra_trb_outclonotype = pl.concat(
        [
            (
                bc_tra_outclonotype.join(
                    bc_trb_outclonotype,
                    on=["barcode", "out_clonotype"],
                    how="inner",
                )
            ),
            # (
            #     bc_tra_outclonotype.join(
            #         bc_trb_outclonotype,
            #         on=["barcode", "out_clonotype"],
            #         how="anti",
            #     )
            # ),
            # (
            #     bc_trb_outclonotype.join(
            #         bc_tra_outclonotype,
            #         on=["barcode", "out_clonotype"],
            #         how="anti",
            #     )
            # ),
        ],
        how="diagonal",
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
    # outclonotype_newclonotype_frequency = (
    #     outclonotype_newclonotype_frequency.sort(
    #         "clonotype_frequency", descending=True
    #     )
    # )
    # outclonotype_newclonotype_frequency = (
    #     outclonotype_newclonotype_frequency.with_columns(
    #         pl.Series(
    #             "new_clonotype",
    #             np.arange(
    #                 outclonotype_newclonotype_frequency.collect().height,
    #                 dtype=np.int32,
    #             ),
    #         ),
    #     )
    # )

    # chaincode_newclonotype_frequency_filtered = (
    #     chaincode_bclist_bccount_outclonotype_filtered.select(
    #         ["chain_code", "out_clonotype"]
    #     ).join(
    #         outclonotype_newclonotype_frequency,
    #         on="out_clonotype",
    #         how="inner",
    #     )
    # )

    # chaincode_newclonotype_frequency_filtered.drop("out_clonotype")

    # filter out clonotypes which have != 1 alpha or != 1 beta TCR chain
    # clonotypes_1_alpha = (
    #     bc_chaincode_chain_outclonotype.filter(pl.col("chain") == "TRA")
    #     .select(["chain_code", "new_clonotype"])
    #     .group_by("new_clonotype")
    #     .agg([pl.col("chain_code").count().alias("count")])
    #     .filter(pl.col("count") == 1)
    #     .select("new_clonotype")
    # )

    # clonotypes_1_beta = (
    #     bc_chaincode_chain_outclonotype.filter(pl.col("chain") == "TRB")
    #     .select(["chain_code", "new_clonotype"])
    #     .group_by("new_clonotype")
    #     .agg([pl.col("chain_code").count().alias("count")])
    #     .filter(pl.col("count") == 1)
    #     .select("new_clonotype")
    # )

    # bc_chaincode_chain_outclonotype = bc_chaincode_chain_outclonotype.join(
    #     clonotypes_1_alpha, on="new_clonotype", how="inner"
    # ).join(clonotypes_1_beta, on="new_clonotype", how="inner")

    # bc_newclonotype_tra = (
    #     bc_chaincode_chain_outclonotype.filter(pl.col("chain") == "TRA")
    #     .select(["barcode", "new_clonotype", "chain_code"])
    #     .rename({"chain_code": "TRA"})
    # )
    # bc_newclonotype_trb = (
    #     bc_chaincode_chain_outclonotype.filter(pl.col("chain") == "TRB")
    #     .select(["barcode", "new_clonotype", "chain_code"])
    #     .rename({"chain_code": "TRB"})
    # )

    # bc_tra_trb_clonotypeid = bc_newclonotype_tra.join(
    #     bc_newclonotype_trb, on=["barcode", "new_clonotype"], how="inner"
    # ).rename({"new_clonotype": "clonotype_id"})
    return bc_tra_trb_clonotypeid, clonotypeid_frequency


""""
Ps[indecestotest]=lapply(Ctables[indecestotest],function(X) fisher.test(X)$p.value)
Pmat=matrix(Ps,nrow=length)
clusteredobject=hclust(as.dist(Pmat))
groups=cutree(clusteredobject,h=1e-6)
"""


def construct_cognate_df(cro):

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
    rawctid_bc_chaincode_chain = interpolate_reads(filt_annot_df)

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

    # remove barcodes without VDJ annotations from mtx
    original_matrix = cro.get_featbcmatrix_obj("orig")

    withvdj_matrix = original_matrix.create_child_matrix(
        "withvdj",
        bc_df=bc_chaincode_chain_with_vdj.select("barcode").unique().collect(),
    )

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
    bc_chaincode_chain_with_feat_vdj = (
        original_matrix.bc_idx_df.lazy()
        .select("barcode")
        .join(bc_chaincode_chain_with_vdj, on="barcode", how="inner")
    )

    # CellRanger known limitation is that when 1 cell contains
    # (only) a TRA which exactly matches a TRA from another cell
    # which contains both a TRA and TRB, the cells might not
    # be called as the same clonotype. This algorithm
    # will reassign clonotypes based on Fisher's exact test
    bc_tra_trb_clonotypeid, clonotypeid_frequency = cluster_clonotypes(
        bc_chaincode_chain_with_feat_vdj, strict=False
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

    # find mhcs with a count >=2 in the unnormalized data
    feature_maxes = np.max(original_matrix.mtx, axis=1)
    feature_indices = np.where(feature_maxes >= 2)[0]
    fname_df = pl.DataFrame(
        {
            "feature_name": original_matrix.get_featnames_ndarr()[
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

    featname_hlaa_hlab_peptide_ctypeid = extract_cresta_peptide_mhc_seqs(
        ctypeid_featname.select("feature_name").unique(),
    ).join(ctypeid_featname, on="feature_name", how="inner")

    hlaaseq_hlaaname = featname_hlaa_hlab_peptide_ctypeid.select(
        ["hla_a_seq", "hla_a_name"]
    ).unique()

    hlabseq_hlabname = featname_hlaa_hlab_peptide_ctypeid.select(
        ["hla_b_seq", "hla_b_name"]
    ).unique()

    # if two hla,hlb,peptide rows are considered matching (if they have overlapping peptides)
    # then we combine them into one row, which we consider to bind with the superset
    # of the peptides that the two rows bind with
    featname_hlaa_hlab_peptide_ctypeid = (
        combine_features(featname_hlaa_hlab_peptide_ctypeid)
        .join(hlaaseq_hlaaname, on="hla_a_name", how="inner")
        .join(hlabseq_hlabname, on="hla_b_name", how="inner")
    )

    ctypeid_hlaa_hlab_peptide = featname_hlaa_hlab_peptide_ctypeid.select(
        [
            "clonotype_id",
            "hla_a_seq",
            "hla_a_name",
            "hla_b_seq",
            "hla_b_name",
            "peptide",
        ]
    ).unique()

    # bc, ctypeid, chain sequence
    bc_ctypeid_tcrseq_chain = (
        filt_annot_df.join(bc_ctypeid_filt.lazy(), on="barcode", how="inner")
        .with_columns(
            pl.concat_str(
                ["fwr1", "cdr1", "fwr2", "cdr2", "fwr3", "cdr3", "fwr4"],
            ).alias("tcr_seq")
        )
        .select(["barcode", "clonotype_id", "chain", "tcr_seq"])
    )
    # this only makes sense if we only allow clonotypes with exactly 1
    # alpha and 1 beta and filter out singletons + doublets
    bc_cltypeid_tcra_tcrb = (
        bc_tra_trb_clonotypeid.join(
            bc_ctypeid_tcrseq_chain.filter(pl.col("chain") == "TRA"),
            on="barcode",
            how="inner",
        )
        .rename({"tcr_seq": "tcr_alpha_seq"})
        .join(
            bc_ctypeid_tcrseq_chain.filter(pl.col("chain") == "TRB"),
            on="barcode",
            how="inner",
        )
        .rename({"tcr_seq": "tcr_beta_seq"})
        .select("barcode", "clonotype_id", "tcr_alpha_seq", "tcr_beta_seq")
    )

    tcra_tcrb_hlaa_hlab_peptide = (
        (
            bc_cltypeid_tcra_tcrb.select(
                ["clonotype_id", "tcr_alpha_seq", "tcr_beta_seq"]
            )
            .unique()
            .join(
                ctypeid_hlaa_hlab_peptide.lazy(),
                on="clonotype_id",
                how="inner",
            )
        )
        .select(
            "tcr_alpha_seq",
            "tcr_beta_seq",
            "hla_a_seq",
            "hla_a_name",
            "hla_b_seq",
            "hla_b_name",
            "peptide",
        )
        .rename(
            {
                "tcr_alpha_seq": "tcr_1_seq",
                "tcr_beta_seq": "tcr_2_seq",
                "hla_a_seq": "mhc_1_seq",
                "hla_a_name": "mhc_1_name",
                "hla_b_seq": "mhc_2_seq",
                "hla_b_name": "mhc_2_name",
            }
        )
    )

    return tcra_tcrb_hlaa_hlab_peptide

    # join with clonotype_id featurename pairings
    # bc_cltypeid_tcra_tcrb_hlaa_hlab_peptide = bc_cltypeid_tcra_tcrb.join(
    #     ctypeid_hlaa_hlab_peptide,
    #     on="clonotype_id",
    #     how="inner",
    # )


def combine_features(featname_hlaa_hlab_peptide_ctypeid, overlap_thresh=9):
    # to make this method deterministic, sort input dataframe by
    # feature name
    # featname_hlaa_hlab_peptide = featname_hlaaseq_hlabseq_peptide.sort(
    #     by="feature_name"
    # )

    hlaaname_hlabname_plist = (
        featname_hlaa_hlab_peptide_ctypeid.select(
            ["hla_a_name", "hla_b_name", "peptide"]
        )
        .group_by(["hla_a_name", "hla_b_name"])
        .agg(
            [
                pl.col("peptide").alias("peptide_list"),
            ]
        )
    )

    hla_a_names = hlaaname_hlabname_plist.select("hla_a_name").to_series()
    hla_b_names = hlaaname_hlabname_plist.select("hla_b_name").to_series()
    peptide_lists = hlaaname_hlabname_plist.select("peptide_list").to_series()

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

        all_peptides = set(p for p in peptide_list)

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

        # # is this match a substring of a match we already found or vice versa?
        # delete = set()
        # add = []
        # substring = False
        # for key in match_dict.keys():
        #     if (
        #         overlap in key or key in overlap
        #     ) and key not in delete:
        #         substring = True
        #         if overlap_len >= len(key):
        #             match_dict[key].update([peptide_1, peptide_2])
        #         else:
        #             delete.add(key)
        #             add.append((overlap, [j, k, *match_dict[key]]))

        # for key in delete:
        #     del match_dict[key]

        # for shorter, indices in add:
        #     match_dict[shorter].update(indices)

        # # if it's not, just add it
        # if not substring:
        #     match_dict[overlap].update([j, k])

        # # we also keep a row if there are no matches for it
        # if no_match:
        #     collapse_hla_bs.append(hla_b)
        #     collapse_hla_as.append(hla_a)
        #     collapse_peptides.append(peptide_1)
        #     collapse_ctypids.append(clonotype_id_list[j].to_list())
        # remaining_peptides = list(all_peptides)
        # for l in remaining_peptides:
        #     collapse_hla_as.append(hla_a)
        #     collapse_hla_bs.append(hla_b)
        #     collapse_peptides.append(peptide_list[l])
        #     collapse_ctypids.append(
        #         np.array(clonotype_id_list[l].to_list(), dtype=np.int32)
        #     )

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
                featname_hlaa_hlab_peptide_ctypeid.filter(
                    (pl.col("hla_a_name") == hla_a)
                    & (pl.col("hla_b_name") == hla_b)
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
            # ensure each peptide shares at least one clonotype
            # else raise an error
            # tmp = set(superset_ctypeids[0])
            # for i in range(1, len(superset_ctypeids)):
            #     if len(tmp.intersection(set(superset_ctypeids[i]))) == 0:
            #         raise ValueError(
            #             f"Peptides {v[0]} and {v[i]} do not share any clonotypes"
            #         )
            # superset_ctypeids = [].extend(
            #     ctypeids for ctypeids in superset_ctypeids
            # )
            superset_ctypeids = [x for xs in superset_ctypeids for x in xs]

            collapse_hla_bs.append(hla_b)
            collapse_hla_as.append(hla_a)
            collapse_peptides.append(keep)
            collapse_ctypids.append(
                np.array(list(superset_ctypeids), dtype=np.int32)
            )

    hlaaseq_hlabseq_peptide_ctypids = pl.DataFrame(
        {
            "hla_a_name": collapse_hla_as,
            "hla_b_name": collapse_hla_bs,
            "peptide": collapse_peptides,
            "clonotype_id": collapse_ctypids,
        }
    )

    hlaaseq_hlabseq_peptide_ctypid = hlaaseq_hlabseq_peptide_ctypids.explode(
        "clonotype_id"
    )
    return hlaaseq_hlabseq_peptide_ctypid


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
            # "index": fname_idx_ndarr,
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

    # bc_preidx_postidx_clonotypeid = bc_preidx_postidx_clonotypeid.sort(
    #     by="post_idx"
    # ).with_row_index(
    #     name="clonotyped_idx",
    # )
    # bc_postidx_ndarr = (
    #     bc_preidx_postidx_clonotypeid.select("post_idx").to_series().to_numpy()
    # )

    # mtx_norm_clonotyped = mtx_norm_with_vdj[:, bc_postidx_ndarr]

    # find_binding_features_and_clonotypes
    clonotype_cltypeidxls = (
        featbc_mtx.bc_idx_df.select(["clonotype_id", featbc_mtx.idx_name])
        .group_by("clonotype_id")
        .agg([pl.col(featbc_mtx.idx_name).alias("index_list")])
    )
    clonotypes = clonotype_cltypeidxls.select("clonotype_id").to_series()
    cltypeidx_lists = clonotype_cltypeidxls.select("index_list").to_series()

    # for each feature, find the range
    # feature_idxs = focalfnames_idx.select("index").to_series()
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
                ["deleteme", "fbc_id", "hla_code", "peptide"]
            )
            .alias("struct")
        )
        .unnest("struct")
    )
    featname_hlacode_peptide = featname_fbcid_hlacode_peptide.drop(
        ["deleteme", "fbc_id"]
    )

    # convert from fbc to official format
    hla_codes = (
        featname_hlacode_peptide.select("hla_code").to_series().to_list()
    )
    beta_seqs = []
    beta_names = []
    alpha_seqs = []
    alpha_names = []
    mhc_converter = HLACodeWebConverter()

    # convert MHC code to transmembrane-removed AA sequence
    for hla_code in hla_codes:
        if hla_code in KNOWN_FEATNAMES:
            officialname = KNOWN_FEATNAMES[hla_code]
        else:
            gene = hla_code[:4]
            allele_group = hla_code[4:6]
            # ignore synonymous mutations
            allele = hla_code[6:].split(":")[0]
            officialname = gene + "*" + allele_group + ":" + allele

        seq = mhc_converter.get_sequence(officialname, top_only=True)
        beta_seq = None
        alpha_seq = None

        if officialname.startswith("DRB"):

            alpha_seq = DRA_EXTRACELLULAR_TOPOLOGICAL_SEQ
            alpha_name = DRA_NAME
            beta_seq = seq
            beta_name = officialname

        elif officialname.startswith("DQB"):
            alpha_seq = mhc_converter.get_sequence(
                DQA_FOR[officialname], top_only=True
            )
            alpha_name = DQA_FOR[officialname]
            beta_seq = seq
            beta_name = officialname
        else:
            raise ValueError(f"Unexpected HLA gene {officialname}")

        beta_seqs.append(beta_seq)
        beta_names.append(beta_name)
        alpha_seqs.append(alpha_seq)
        alpha_names.append(alpha_name)

    featname_hlaa_hlab_peptide = featname_hlacode_peptide.with_columns(
        [
            pl.Series("hla_a_seq", alpha_seqs),
            pl.Series("hla_a_name", alpha_names),
            pl.Series("hla_b_seq", beta_seqs),
            pl.Series("hla_b_name", beta_names),
        ]
    )
    featname_hlaa_hlab_peptide.drop_in_place("hla_code")
    return featname_hlaa_hlab_peptide


# class HLACodeCSVConverter(MHCCodeConverter):
#     def __init__(self, dir="./HLA_seqs"):
#         dir_path = Path(dir)
#         dqa = dir_path / ("DQA_allele_seqs.csv")
#         self.dqa = pl.read_csv(
#             dqa, has_header=False, new_columns=["code", "seq"]
#         )

#         dqb = dir_path / ("DQB_allele_seqs.csv")
#         self.dqb = pl.read_csv(
#             dqb, has_header=False, new_columns=["code", "seq"]
#         )

#         drb = dir_path / ("DRB_allele_seqs.csv")
#         self.drb = pl.read_csv(
#             drb, has_header=False, new_columns=["code", "seq"]
#         )

#     def get_sequence(self, official_name, topological_only=False):
#         if official_name.startswith("DQA"):
#             query = self.dqa.filter(pl.col("code") == official_name)
#         elif official_name.startswith("DQB"):
#             query = self.dqb.filter(pl.col("code") == official_name)
#         elif official_name.startswith("DRB"):
#             query = self.drb.filter(pl.col("code") == official_name)
#         else:
#             raise ValueError("Invalid MHC allele")

#         if query.height == 0:
#             raise ValueError(f"No sequence found for allele {official_name}")
#         if query.height > 1:
#             raise ValueError(
#                 f"Multiple sequences found for allele {official_name}"
#             )

#         seq = query.select("seq").to_series().item()

#         if official_name.startswith("DRB") and topological_only:
#             seq = seq[DRB_TOPOLOGY_DOMAIN]

#         return seq


def parse_cresta_dirs(directory_paths: list):
    tmp_dfs = []
    for directory_path in directory_paths:
        cro = CellRangerOutput(directory_path)
        donor_num = cro.donor_num
        cognate_df = construct_cognate_df(cro).with_columns(
            pl.lit(1, dtype=pl.Boolean).alias("cognate"),
            pl.lit(donor_num, dtype=pl.Int32).alias("participant_id"),
        )
        tmp_dfs.append(cognate_df)
    return pl.concat(tmp_dfs)


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
        "-p",
        "--p_value",
        help="P value threshold",
        required=False,
        default=10,
    )
    parser.add_argument(
        "-f",
        "--fold_value",
        help="Fold change threshold",
        required=False,
        default=0.3,
    )
    # parser.add_argument(
    #     "-s2", "--structure_2", help="Second structure", required=True
    # )
    args = parser.parse_args()

    dirs = [d.strip() for d in args.directories.split(",")]
    out_df = parse_cresta_dirs(dirs)
    out_df.to_csv("cresta_data.csv")
