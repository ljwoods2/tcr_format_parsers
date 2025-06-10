import hashlib
import anarci
import numpy as np
import warnings
import difflib
import polars as pl
from tcrdist.repertoire import TCRrep


def hash_tcr_sequence(tcr_seq: str, hash_type: str = "md5") -> str:
    """
    Hash a TCR sequence using the specified hash function.

    Args:
        tcr_seq (str): The TCR sequence string.
        hash_type (str): The hash function to use ('md5', 'sha1', 'sha256', etc.)

    Returns:
        str: The hexadecimal digest of the hashed sequence.
    """
    # Select the hash function
    if hash_type.lower() == "md5":
        h = hashlib.md5()
    elif hash_type.lower() == "sha1":
        h = hashlib.sha1()
    elif hash_type.lower() == "sha256":
        h = hashlib.sha256()
    else:
        raise ValueError("Unsupported hash type")

    # Encode the sequence and compute the hash
    h.update(tcr_seq.encode("utf-8"))
    return h.hexdigest()


# https://www.uniprot.org/uniprotkb/P01848/entry
# Stitchr and uniprot agree exactly
HUMAN_TRAC_SEQ = "IQNPDPAVYQLRDSKSSDKSVCLFTDFDSQTNVSQSKDSDVYITDKTVLDMRSMDFKSNSAVAWSNKSDFACANAFNNSIIPEDTFFPSPESSCDVKLVEKSFETDTNLNFQNLSVIGFRILLLKVAGFNLLMTLRLWSS"
HUMAN_TRAC_TOP = slice(0, 115)
HUMAN_TRAC_PREFIX = ["IQNPD", "IQNPE", "IQKPD"]

# https://www.uniprot.org/uniprotkb/P01850/entry
# Stitchr and uniprot agree exactly
HUMAN_TRBC1_SEQ = "DLNKVFPPEVAVFEPSEAEISHTQKATLVCLATGFFPDHVELSWWVNGKEVHSGVSTDPQPLKEQPALNDSRYCLSSRLRVSATFWQNPRNHFRCQVQFYGLSENDEWTQDRAKPVTQIVSAEAWGRADCGFTSVSYQQGVLSATILYEILLGKATLYAVLVSALVLMAMVKRKDF"
HUMAN_TRBC1_TOP = slice(0, 149)
HUMAN_TRBC1_PREFIX = ["DLNKV", "DLKNV", "DLRNV"]

# https://www.uniprot.org/uniprotkb/A0A5B9/entry
# modified residue 8 (0-indexed) K->E for consistency with Stitchr seq
HUMAN_TRBC2_SEQ = "DLKNVFPPEVAVFEPSEAEISHTQKATLVCLATGFYPDHVELSWWVNGKEVHSGVSTDPQPLKEQPALNDSRYCLSSRLRVSATFWQNPRNHFRCQVQFYGLSENDEWTQDRAKPVTQIVSAEAWGRADCGFTSESYQQGVLSATILYEILLGKATLYAVLVSALVLMAMVKRKDSRG"
HUMAN_TRBC2_TOP = slice(0, 144)

# https://www.uniprot.org/uniprotkb/P01849/entry
MOUSE_TRAC_SEQ = "IQNPEPAVYQLKDPRSQDSTLCLFTDFDSQINVPKTMESGTFITDKTVLDMKAMDSKSNGAIAWSNQTSFTCQDIFKETNATYPSSDVPCDATLTEKSFETDMNLNFQNLSVMGLRILLLKVAGFNLLMTLRLWSS"
MOUSE_TRAC_TOP = slice(0, 110)
MOUSE_TRAC_PREFIX = ["IQNPE"]
# MOUSE_TRBC1_SEQ
# MOUSE_TRBC1_TOP
# MOUSE_TRBC1_PREFIX = ["IQNPE"]
MOUSE1 = "DLRNVTPPKVSLFEPSKAEIANKQKATLVCLARGFFPDHVELSWWVNGKEVHSGVSTDPQAYKESNYSYCLSSRLRVSATFWHNPRNHFRCQVQFHGLSEEDKWPEGSPKPVTQNISAEAWGRADCGITSASYQQGVLSATILYEILLGKATLYAVLVSTLVVMAMVKRKNS"
MOUSE2 = "DLRNVTPPKVSLFEPSKAEIANKQKATLVCLARGFFPDHVELSWWVNGKEVHSGVSTDPQAYKESNYSYCLSSRLRVSATFWHNPRNHFRCQVQFHGLSEEDKWPEGSPKPVTQNISAEAWGRADCGITSASYHQGVLSATILYEILLGKATLYAVLVSGLVLMAMVKKKNS"

STANDARDIZE_TCR_STRUCT = pl.Struct(
    {
        "seq": pl.String,
        "c_motif_match_len": pl.Int64,
    }
)


# def standardize_tcr(tcr_seq, tcr_chain, species, v_region_only=True):
#     """
#     Use IMGT numbering to identify the variable portion of the TCR.
#     Then, based on the species and the chain, add a fixed constant
#     region sequence with the transmembrane portion removed.

#     Returns:
#         str: The standardized TCR sequence.
#         int: The length of the prefix used to identify the C region.
#     """
#     if species not in ["human", "mouse"]:
#         raise ValueError(f"Invalid species '{species}'")
#     if tcr_chain == "alpha":
#         if species == "human":
#             c_region = HUMAN_TRAC_SEQ[HUMAN_TRAC_TOP]
#             prefix = HUMAN_TRAC_PREFIX
#         elif species == "mouse":
#             c_region = MOUSE_TRAC_SEQ[MOUSE_TRAC_TOP]
#             prefix = MOUSE_TRAC_PREFIX
#     elif tcr_chain == "beta":
#         if species == "human":
#             c_region = HUMAN_TRBC1_SEQ[HUMAN_TRBC1_TOP]
#             prefix = HUMAN_TRBC1_PREFIX
#         # elif species == "mouse":
#         #     c_region = MOUSE_TRBC1_SEQ[MOUSE_TRBC1_TOP]
#     else:
#         raise ValueError(f"Invalid chain '{tcr_chain}'")

#     tcr_seq_np = np.array(list(tcr_seq))

#     try:
#         residue_indices, imgt_number = annotate_tcr(
#             tcr_seq, np.arange(len(tcr_seq)), tcr_chain, species
#         )
#     except ValueError as e:
#         warnings.warn(f"Failed to annotate TCR seq {tcr_seq}: {e}")
#         return {
#             "seq": None,
#             "c_motif_match_len": None,
#         }

#     imgt_numbered_region = "".join(tcr_seq_np[residue_indices])

#     if v_region_only:
#         return imgt_numbered_region
#     ##  validate the junction has not changed

#     # first tcr seq idx after identified v region
#     j_start = residue_indices[-1] + 1

#     # case where input TCR is already sliced to V region only
#     if j_start > len(tcr_seq_np) - 1:
#         # assume the junction is OK
#         warnings.warn(
#             f"TCR seq {tcr_seq} has no included C region fragment, assuming "
#             "entire V region is included and not cut off."
#         )
#         return {
#             "seq": imgt_numbered_region + c_region,
#             "c_motif_match_len": 0,
#         }

#     # naiive c region (anything to the right of IMGT-defined v region)
#     orig_c_region = tcr_seq[j_start:]

#     j_len = len(tcr_seq) - j_start
#     # arbitrarily use length 5 prefix to test against
#     # known canonical and alt c region prefixes
#     # if 5 isn't possible because orig c region is < 5 AAs,
#     # use min
#     test_len = min(j_len, 5)
#     if test_len < 5:
#         warnings.warn(
#             f"Matching prefixes of length {test_len}, likely nonsense result for "
#             f"{tcr_seq}"
#         )

#     # shorten prefixes if needed
#     prefix_test = [prefix[i][:test_len] for i in range(len(prefix))]

#     idx = -1
#     for p in prefix_test:
#         # look for the prefix
#         # anything between the IMGT-defined V region
#         # and the canonical C region we call a junction
#         # and add back in
#         try:
#             idx = orig_c_region.index(p)
#         except ValueError:
#             continue
#         break

#     if idx == -1:
#         # print(f"{orig_c_region}")
#         warnings.warn(
#             f"Failed to find C region start for {tcr_seq}. Assuming entire V region and junction are included."
#         )
#         return {
#             "seq": imgt_numbered_region + orig_c_region + c_region,
#             "c_motif_match_len": 0,
#         }

#         # raise ValueError(
#         #     f"Failed to find C region start for {tcr_chain} seq: {tcr_seq}"
#         # )

#     # potentially the empty string if anarci performed V-region identificaiton perfectly
#     junction = orig_c_region[0:idx]

#     return {
#         "seq": imgt_numbered_region + junction + c_region,
#         "c_motif_match_len": test_len,
#     }
#     # if m.b != 0:
#     #     # fall back on a more expensive hamming loss calculation
#     #     orig_c_ndarr = np.array(list(orig_c_region))
#     #     c_ndarr = np.array(list(c_region))

#     #     thresh = 0.2
#     #     loss = float("inf")

#     #     orig_start = 0
#     #     str_len = min(len(orig_c_ndarr[orig_start:]), len(c_ndarr))
#     #     orig_end = orig_start + str_len

#     #     while loss > thresh and str_len > 3:
#     #         sliced_orig = orig_c_ndarr[orig_start:orig_end]
#     #         sliced_true = c_ndarr[:str_len]

#     #         loss = hamming_loss(sliced_orig, sliced_true)

#     #         if loss <= thresh:
#     #             return (
#     #                 "".join(tcr_seq_np[residue_indices])
#     #                 + orig_c_region[:orig_start]
#     #                 + c_region
#     #             )

#     #         orig_start += 1
#     #         str_len = min(len(orig_c_ndarr[orig_start:]), len(c_ndarr))
#     #         orig_end = orig_start + str_len

#     # raise ValueError(
#     #     f"IMGT numbering failed to identify V region for seq {tcr_seq}"
#     # )

#     # # i.e. j_start + j_len = end of slice to compare
#     # j_len = len(tcr_seq) - j_start
#     # j_test = min(j_len, 5)

#     # orig_junction = tcr_seq[j_start : j_start + j_test]
#     # proposed_junction = c_region[:j_test]

#     # if orig_junction == proposed_junction:
#     #     return "".join(tcr_seq_np[residue_indices]) + c_region

#     # # trickiest case.


# def standardize_stitchr_tcr(tcr_seq, tcr_chain, species, j_gene):
#     # 1. Remove signal peptide if present
#     residue_indices, imgt_number = annotate_tcr(
#         tcr_seq, np.arange(len(tcr_seq)), tcr_chain, species
#     )
#     tcr_seq_np = np.array(list(tcr_seq))
#     non_signal_start = residue_indices[0]

#     tcr_seq = "".join(tcr_seq_np[non_signal_start:])

#     # 2. Replace the entire constant region AA seq with just the constant region
#     # up to but not including the transmembrane helicies

#     if tcr_chain == "alpha":
#         if species == "human":
#             const_start = tcr_seq.index(HUMAN_TRAC_SEQ)
#             tcr_seq = tcr_seq[:const_start] + HUMAN_TRAC_SEQ[HUMAN_TRAC_TOP]
#             return tcr_seq
#         elif species == "mouse":
#             const_start = tcr_seq.index(MOUSE_TRAC_SEQ)
#             tcr_seq = tcr_seq[:const_start] + MOUSE_TRAC_SEQ[MOUSE_TRAC_TOP]
#             return tcr_seq
#     else:
#         if species == "human":
#             if "TRBJ1" in j_gene:
#                 const_start = tcr_seq.index(HUMAN_TRBC1_SEQ)
#                 tcr_seq = (
#                     tcr_seq[:const_start] + HUMAN_TRBC1_SEQ[HUMAN_TRBC1_TOP]
#                 )
#                 return tcr_seq
#             elif "TRBJ2" in j_gene:
#                 const_start = tcr_seq.index(HUMAN_TRBC2_SEQ)
#                 tcr_seq = (
#                     tcr_seq[:const_start] + HUMAN_TRBC2_SEQ[HUMAN_TRBC2_TOP]
#                 )
#                 return tcr_seq
#         # elif species == "mouse":
#         #     if "TRBJ1" in j_gene:
#         #         const_start = tcr_seq.index(MOUSE_TRBC1_SEQ)
#         #         tcr_seq = tcr_seq[:const_start] + MOUSE_TRBC1_SEQ[MOUSE_TRBC1_TOP]
#         #         return tcr_seq
#         #     elif "TRBJ2" in j_gene:
#         #         const_start = tcr_seq.index(MOUSE_TRBC2_SEQ)
#         #         tcr_seq = tcr_seq[:const_start] + MOUSE_TRBC2_SEQ[MOUSE_TRBC2_TOP]
#         #         return tcr_seq


def shorten_tcr_to_vregion(tcr_seq, tcr_chain, species, strict=False):

    res_slice, _, _ = annotate_tcr(
        tcr_seq, np.arange(len(tcr_seq)), tcr_chain, species, strict=strict
    )

    return "".join(np.array(list(tcr_seq))[res_slice])


def extract_tcrdist_cols(row):

    for tcr in [1, 2]:
        regions = tcr_by_imgt_region(
            row[f"tcr_{tcr}_seq"],
            np.arange(len(row[f"tcr_{tcr}_seq"])),
            row[f"tcr_{tcr}_chain"],
            row[f"tcr_{tcr}_species"],
        )
        for region in ["cdr_1", "cdr_2", "cdr_2_5", "cdr_3"]:
            row[f"tcr_{tcr}_{region}"] = "".join(
                np.array(list(row[f"tcr_{tcr}_seq"]))[regions[region]]
            )

        v_gene, j_gene = extract_v_j_genes(
            row[f"tcr_{tcr}_seq"],
            row[f"tcr_{tcr}_chain"],
            row[f"tcr_{tcr}_species"],
        )
        row[f"tcr_{tcr}_v_gene"] = v_gene
        row[f"tcr_{tcr}_j_gene"] = j_gene

    return row


def extract_v_j_genes(tcr_seq, tcr_chain, species):

    if tcr_chain == "alpha":
        # quirk of ANARCI
        # https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/anarci/
        allow = set(["A", "D"])
    elif tcr_chain == "beta":
        allow = set(["B"])
    else:
        raise ValueError(f"Invalid chain '{tcr_chain}'")

    genes = anarci.run_anarci(
        [("AAA", tcr_seq)],
        scheme="imgt",
        allowed_species=[species],
        allow=allow,
        assign_germline=True,
    )[2][0][0]["germlines"]

    return genes["v_gene"][0][1], genes["j_gene"][0][1]


def annotate_tcr(tcr_seq, resindices, tcr_chain, species, strict=False):
    """Return the resindices of a contiguous substring of resindices
    which aligned to a TCR seq using ANARCI"""
    if len(tcr_seq) != len(resindices):
        raise ValueError(
            "TCR sequence length must match length of " "residue index array"
        )

    if tcr_chain == "alpha":
        # quirk of ANARCI
        # https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/anarci/
        allow = set(["A", "D"])
    elif tcr_chain == "beta":
        allow = set(["B"])
    else:
        raise ValueError(f"Invalid chain '{tcr_chain}'")

    results = anarci.run_anarci(
        [("AAA", tcr_seq)],
        scheme="imgt",
        allowed_species=[species],
        allow=allow,
    )
    # check for None
    if results[1][0] is None:
        raise ValueError(f"No domain found for sequence '{tcr_seq}'")
    elif len(results[1][0]) > 1:
        raise ValueError("Multiple domains found for sequence")

    # validate that the species provided is what anarci found
    if strict:
        if results[2][0][0]["species"] != species:
            raise ValueError(
                f"Species mismatch: {species} vs {results[1][0][0][1]}"
            )

    numbering = results[1][0][0]

    sub_start = numbering[1]
    sub_stop = numbering[2] + 1

    # this entire substring will be numbered
    # however, there may be gaps in the sequence
    # which are given an IMGT number
    numbered_substring = tcr_seq[sub_start:sub_stop]
    resindices_slice = resindices[sub_start:sub_stop]
    imgt_num = np.zeros((len(range(sub_start, sub_stop)),), dtype=np.int32)
    imgt_tuples = numbering[0]
    j = 0
    for i in range(len(numbered_substring)):
        aa = numbered_substring[i]
        while j < len(imgt_tuples) and imgt_tuples[j][1] != aa:
            j += 1
        if j < len(imgt_tuples):
            imgt_num[i] = imgt_tuples[j][0][0]
            j += 1

    # zero is not an IMGT number, so we use this as a quick error cehck
    n_zeroes = np.count_nonzero(imgt_num == 0)
    if n_zeroes != 0:
        raise ValueError(
            f"0 is not an IMGT number. Numbering failed for TCR '{tcr_seq}'"
        )

    return resindices_slice, imgt_num, (sub_start, sub_stop)


def tcr_by_imgt_region(tcr_seq, tcr_resindices, tcr_chain, tcr_species):
    tcr_indices, imgt_num, _ = annotate_tcr(
        tcr_seq,
        tcr_resindices,
        tcr_chain,
        tcr_species,
    )

    tcr_residx_dict = {}

    tcr_residx_dict["fwr_1"] = tcr_indices[
        ((imgt_num >= 1) & (imgt_num <= 26))
    ]

    tcr_residx_dict["cdr_1"] = tcr_indices[
        ((imgt_num >= 27) & (imgt_num <= 38))
    ]

    tcr_residx_dict["fwr_2"] = tcr_indices[
        ((imgt_num >= 39) & (imgt_num <= 55))
    ]

    tcr_residx_dict["cdr_2"] = tcr_indices[
        ((imgt_num >= 56) & (imgt_num <= 65))
    ]

    tcr_residx_dict["fwr_3"] = tcr_indices[
        ((imgt_num >= 66) & (imgt_num <= 103))
    ]

    tcr_residx_dict["cdr_2_5"] = tcr_indices[
        ((imgt_num >= 81) & (imgt_num <= 86))
    ]

    tcr_residx_dict["cdr_3"] = tcr_indices[
        ((imgt_num >= 104) & (imgt_num <= 118))
    ]

    tcr_residx_dict["fwr_4"] = tcr_indices[
        ((imgt_num >= 119) & (imgt_num <= 129))
    ]

    return tcr_residx_dict


def pw_tcrdist(tcr_df, species=None, chains=None, use_provided_cdr=False):

    if species is None:
        species = tcr_df[0].select("tcr_1_species").item()

    if chains is None:
        chain_1 = tcr_df[0].select("tcr_1_chain").item()
        chain_2 = tcr_df[0].select("tcr_2_chain").item()

        chains = [chain_1, chain_2]

    if use_provided_cdr:
        tcr_df_pd = tcr_df.with_columns(
            pl.col("tcr_1_cdr_1").alias("cdr1_a_aa"),
            pl.col("tcr_1_cdr_2").alias("cdr2_a_aa"),
            pl.col("tcr_1_cdr_2_5").alias("pmhc_a_aa"),
            pl.col("tcr_1_cdr_3").alias("cdr3_a_aa"),
            pl.col("tcr_2_cdr_1").alias("cdr1_b_aa"),
            pl.col("tcr_2_cdr_2").alias("cdr2_b_aa"),
            pl.col("tcr_2_cdr_2_5").alias("pmhc_b_aa"),
            pl.col("tcr_2_cdr_3").alias("cdr3_b_aa"),
            pl.lit(1).alias("count"),
        ).to_pandas()

        tr = TCRrep(
            cell_df=tcr_df_pd,
            organism=species,
            chains=chains,
            db_file="alphabeta_gammadelta_db.tsv",
            infer_cdrs=False,
        )

    else:

        tcr_df_pd = tcr_df.with_columns(
            pl.col("tcr_1_cdr_3").alias("cdr3_a_aa"),
            pl.col("tcr_2_cdr_3").alias("cdr3_b_aa"),
            pl.col("tcr_1_v_gene").alias("v_a_gene"),
            pl.col("tcr_2_v_gene").alias("v_b_gene"),
            pl.col("tcr_1_j_gene").alias("j_a_gene"),
            pl.col("tcr_2_j_gene").alias("j_b_gene"),
            pl.lit(1).alias("count"),
        ).to_pandas()

        #     {
        #         # allow these to be inferred
        #         # "tcr_1_cdr_1": "cdr1_a_aa",
        #         # "tcr_1_cdr_2": "cdr2_a_aa",
        #         "tcr_1_cdr_3": "cdr3_a_aa",
        #         # "tcr_2_cdr_1": "cdr1_b_aa",
        #         # "tcr_2_cdr_2": "cdr2_b_aa",
        #         "tcr_2_cdr_3": "cdr3_b_aa",
        #         "tcr_1_v_gene": "v_a_gene",
        #         "tcr_2_v_gene": "v_b_gene",
        #         "tcr_1_j_gene": "j_a_gene",
        #         "tcr_2_j_gene": "j_b_gene",
        #     }
        #

        tr = TCRrep(
            cell_df=tcr_df_pd,
            organism=species,
            chains=chains,
            db_file="alphabeta_gammadelta_db.tsv",
        )

    # assert tr.pw_alpha.shape[0] == len(tcr_df_pd)

    dists = tr.pw_alpha + tr.pw_beta

    out_df = tr.clone_df.copy()

    out_df["index"] = out_df.index

    out_df.drop(
        labels=[
            "v_a_gene",
            "j_a_gene",
            "cdr3_a_aa",
            "v_b_gene",
            "j_b_gene",
            "cdr3_b_aa",
            "cdr1_a_aa",
            "cdr2_a_aa",
            "pmhc_a_aa",
            "cdr1_b_aa",
            "cdr2_b_aa",
            "pmhc_b_aa",
            "count",
            "clone_id",
        ],
        axis=1,
        inplace=True,
    )

    return pl.DataFrame(out_df), dists


# https://github.com/phbradley/TCRdock/blob/c5a7af42eeb0c2a4492a4d4fe803f1f9aafb6193/algorithms_from_the_paper.py#L5
def pick_reps(D, num_reps=50, sdev_big=120.0, sdev_small=36.0, min_size=0.5):
    """D is a symmetric distance matrix (e.g., of TCRdist distances)
    num_reps is the number of representatives to choose
    sdev_big defines the neighbor-density sum used for ranking
    sdev_small limits the redundancy
    both sdev_big and sdev_small are in distance units (ie same units as D)
    """
    # the weight remaining for each instance
    wts = np.array([1.0] * D.shape[0])

    reps, sizes = [], []
    for ii in range(num_reps):
        if np.sum(wts) < 1e-2:
            break
        gauss_big = (
            np.exp(-1 * (D / sdev_big) ** 2) * wts[:, None] * wts[None, :]
        )
        gauss_small = (
            np.exp(-1 * (D / sdev_small) ** 2) * wts[:, None] * wts[None, :]
        )
        nbr_sum = np.sum(gauss_big, axis=1)
        rep = np.argmax(nbr_sum)
        size = nbr_sum[rep]
        if size < min_size:
            break
        wts = np.maximum(0.0, wts - gauss_small[rep, :] / wts[rep])
        assert wts[rep] < 1e-3
        reps.append(rep)
        sizes.append(size)
    return reps, sizes
