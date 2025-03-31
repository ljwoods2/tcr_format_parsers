import hashlib
import anarci
import numpy as np
import warnings
import difflib
import polars as pl


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


def standardize_tcr(tcr_seq, tcr_chain, species, v_region_only=True):
    """
    Use IMGT numbering to identify the variable portion of the TCR.
    Then, based on the species and the chain, add a fixed constant
    region sequence with the transmembrane portion removed.

    Returns:
        str: The standardized TCR sequence.
        int: The length of the prefix used to identify the C region.
    """
    if species not in ["human", "mouse"]:
        raise ValueError(f"Invalid species '{species}'")
    if tcr_chain == "alpha":
        if species == "human":
            c_region = HUMAN_TRAC_SEQ[HUMAN_TRAC_TOP]
            prefix = HUMAN_TRAC_PREFIX
        elif species == "mouse":
            c_region = MOUSE_TRAC_SEQ[MOUSE_TRAC_TOP]
            prefix = MOUSE_TRAC_PREFIX
    elif tcr_chain == "beta":
        if species == "human":
            c_region = HUMAN_TRBC1_SEQ[HUMAN_TRBC1_TOP]
            prefix = HUMAN_TRBC1_PREFIX
        # elif species == "mouse":
        #     c_region = MOUSE_TRBC1_SEQ[MOUSE_TRBC1_TOP]
    else:
        raise ValueError(f"Invalid chain '{tcr_chain}'")

    tcr_seq_np = np.array(list(tcr_seq))

    try:
        residue_indices, imgt_number = annotate_tcr(
            tcr_seq, np.arange(len(tcr_seq)), tcr_chain, species
        )
    except ValueError as e:
        warnings.warn(f"Failed to annotate TCR seq {tcr_seq}: {e}")
        return {
            "seq": None,
            "c_motif_match_len": None,
        }

    imgt_numbered_region = "".join(tcr_seq_np[residue_indices])

    if v_region_only:
        return imgt_numbered_region
    ##  validate the junction has not changed

    # first tcr seq idx after identified v region
    j_start = residue_indices[-1] + 1

    # case where input TCR is already sliced to V region only
    if j_start > len(tcr_seq_np) - 1:
        # assume the junction is OK
        warnings.warn(
            f"TCR seq {tcr_seq} has no included C region fragment, assuming "
            "entire V region is included and not cut off."
        )
        return {
            "seq": imgt_numbered_region + c_region,
            "c_motif_match_len": 0,
        }

    # naiive c region (anything to the right of IMGT-defined v region)
    orig_c_region = tcr_seq[j_start:]

    j_len = len(tcr_seq) - j_start
    # arbitrarily use length 5 prefix to test against
    # known canonical and alt c region prefixes
    # if 5 isn't possible because orig c region is < 5 AAs,
    # use min
    test_len = min(j_len, 5)
    if test_len < 5:
        warnings.warn(
            f"Matching prefixes of length {test_len}, likely nonsense result for "
            f"{tcr_seq}"
        )

    # shorten prefixes if needed
    prefix_test = [prefix[i][:test_len] for i in range(len(prefix))]

    idx = -1
    for p in prefix_test:
        # look for the prefix
        # anything between the IMGT-defined V region
        # and the canonical C region we call a junction
        # and add back in
        try:
            idx = orig_c_region.index(p)
        except ValueError:
            continue
        break

    if idx == -1:
        # print(f"{orig_c_region}")
        warnings.warn(
            f"Failed to find C region start for {tcr_seq}. Assuming entire V region and junction are included."
        )
        return {
            "seq": imgt_numbered_region + orig_c_region + c_region,
            "c_motif_match_len": 0,
        }

        # raise ValueError(
        #     f"Failed to find C region start for {tcr_chain} seq: {tcr_seq}"
        # )

    # potentially the empty string if anarci performed V-region identificaiton perfectly
    junction = orig_c_region[0:idx]

    return {
        "seq": imgt_numbered_region + junction + c_region,
        "c_motif_match_len": test_len,
    }
    # if m.b != 0:
    #     # fall back on a more expensive hamming loss calculation
    #     orig_c_ndarr = np.array(list(orig_c_region))
    #     c_ndarr = np.array(list(c_region))

    #     thresh = 0.2
    #     loss = float("inf")

    #     orig_start = 0
    #     str_len = min(len(orig_c_ndarr[orig_start:]), len(c_ndarr))
    #     orig_end = orig_start + str_len

    #     while loss > thresh and str_len > 3:
    #         sliced_orig = orig_c_ndarr[orig_start:orig_end]
    #         sliced_true = c_ndarr[:str_len]

    #         loss = hamming_loss(sliced_orig, sliced_true)

    #         if loss <= thresh:
    #             return (
    #                 "".join(tcr_seq_np[residue_indices])
    #                 + orig_c_region[:orig_start]
    #                 + c_region
    #             )

    #         orig_start += 1
    #         str_len = min(len(orig_c_ndarr[orig_start:]), len(c_ndarr))
    #         orig_end = orig_start + str_len

    # raise ValueError(
    #     f"IMGT numbering failed to identify V region for seq {tcr_seq}"
    # )

    # # i.e. j_start + j_len = end of slice to compare
    # j_len = len(tcr_seq) - j_start
    # j_test = min(j_len, 5)

    # orig_junction = tcr_seq[j_start : j_start + j_test]
    # proposed_junction = c_region[:j_test]

    # if orig_junction == proposed_junction:
    #     return "".join(tcr_seq_np[residue_indices]) + c_region

    # # trickiest case.


def standardize_stitchr_tcr(tcr_seq, tcr_chain, species, j_gene):
    # 1. Remove signal peptide if present
    residue_indices, imgt_number = annotate_tcr(
        tcr_seq, np.arange(len(tcr_seq)), tcr_chain, species
    )
    tcr_seq_np = np.array(list(tcr_seq))
    non_signal_start = residue_indices[0]

    tcr_seq = "".join(tcr_seq_np[non_signal_start:])

    # 2. Replace the entire constant region AA seq with just the constant region
    # up to but not including the transmembrane helicies

    if tcr_chain == "alpha":
        if species == "human":
            const_start = tcr_seq.index(HUMAN_TRAC_SEQ)
            tcr_seq = tcr_seq[:const_start] + HUMAN_TRAC_SEQ[HUMAN_TRAC_TOP]
            return tcr_seq
        elif species == "mouse":
            const_start = tcr_seq.index(MOUSE_TRAC_SEQ)
            tcr_seq = tcr_seq[:const_start] + MOUSE_TRAC_SEQ[MOUSE_TRAC_TOP]
            return tcr_seq
    else:
        if species == "human":
            if "TRBJ1" in j_gene:
                const_start = tcr_seq.index(HUMAN_TRBC1_SEQ)
                tcr_seq = (
                    tcr_seq[:const_start] + HUMAN_TRBC1_SEQ[HUMAN_TRBC1_TOP]
                )
                return tcr_seq
            elif "TRBJ2" in j_gene:
                const_start = tcr_seq.index(HUMAN_TRBC2_SEQ)
                tcr_seq = (
                    tcr_seq[:const_start] + HUMAN_TRBC2_SEQ[HUMAN_TRBC2_TOP]
                )
                return tcr_seq
        # elif species == "mouse":
        #     if "TRBJ1" in j_gene:
        #         const_start = tcr_seq.index(MOUSE_TRBC1_SEQ)
        #         tcr_seq = tcr_seq[:const_start] + MOUSE_TRBC1_SEQ[MOUSE_TRBC1_TOP]
        #         return tcr_seq
        #     elif "TRBJ2" in j_gene:
        #         const_start = tcr_seq.index(MOUSE_TRBC2_SEQ)
        #         tcr_seq = tcr_seq[:const_start] + MOUSE_TRBC2_SEQ[MOUSE_TRBC2_TOP]
        #         return tcr_seq


def annotate_tcr(tcr_seq, resindices, tcr_chain, species):
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

    numbering = results[1][0][0]

    sub_start = numbering[1]
    sub_stop = numbering[2] + 1
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

    return resindices_slice, imgt_num
