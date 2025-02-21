import requests
import polars as pl
from tcr_format_parsers.data.datafiles import hla_uniprot_data_cache
import threading


class MHCCodeConverter:
    def __init__(self):
        pass

    def get_sequence(self, official_name):
        pass


# https://www.uniprot.org/uniprotkb/P01911/entry
# https://www.uniprot.org/uniprotkb/P79483/entry
# https://www.uniprot.org/uniprotkb/Q30154/entry
DRB_EXTRACELLULAR_TOPOLOGICAL_DOMAIN = slice(29, 227)

# https://www.uniprot.org/uniprotkb/P01903/entry#sequences
DRA_SEQ = "MAISGVPVLGFFIIAVLMSAQESWAIKEEHVIIQAEFYLNPDQSGEFMFDFDGDEIFHVDMAKKETVWRLEEFGRFASFEAQGALANIAVDKANLEIMTKRSNYTPITNVPPEVTVLTNSPVELREPNVLICFIDKFTPPVVNVTWLRNGKPVTTGVSETVFLPREDHLFRKFHYLPFLPSTEDVYDCRVEHWGLDEPLLKHWEFDAPSPLPETTENVVCALGLTVGLVGIIIGTIFIIKGLRKSNAAERRGPL"
DRA_NAME = "DRA1*01:02"

# https://www.uniprot.org/uniprotkb/P01903/entry
DRA_EXTRACELLULAR_TOPOLOGICAL_SEQ = DRA_SEQ[26:216]

# https://www.uniprot.org/uniprotkb/P01906/entry
DQA2_EXTRACELLULAR_TOPOLOGICAL_DOMAIN = slice(23, 217)

# https://www.uniprot.org/uniprotkb/P01909/entry
DQA1_EXTRACELLULAR_TOPOLOGICAL_DOMAIN = slice(23, 216)

# https://www.uniprot.org/uniprotkb/P01920/entry
DQB1_EXTRACELLULAR_TOPOLOGICAL_DOMAIN = slice(32, 230)

# https://www.uniprot.org/uniprotkb/P05538/entry
DQB2_EXTRACELLULAR_TOPOLOGICAL_DOMAIN = slice(33, 229)

# https://www.uniprot.org/uniprotkb/P04439/entry
A_EXTRACELLULAR_TOPOLOGICAL_DOMAIN = slice(24, 308)

# https://www.uniprot.org/uniprotkb/P01889/entry
B_EXTRACELLULAR_TOPOLOGICAL_DOMAIN = slice(24, 309)

# https://www.uniprot.org/uniprotkb/P10321/entry
C_EXTRACELLULAR_TOPOLOGICAL_DOMAIN = slice(24, 308)

# https://www.uniprot.org/uniprotkb/P13747/entry
E_EXTRACELLULAR_TOPOLOGICAL_DOMAIN = slice(21, 305)

# https://www.uniprot.org/uniprotkb/P04440/entry#subcellular_location
DPB1_EXTRACELLULAR_TOPOLOGICAL_DOMAIN = slice(29, 225)

# https://www.uniprot.org/uniprotkb/P20036/entry
DPA1_EXTRACELLULAR_TOPOLOGICAL_DOMAIN = slice(28, 222)

DQA_FOR = {
    "DQB1*06:02": "DQA1*01:02",
    "DQB1*02:01": "DQA1*05:01",
    "DQB1*02:02": "DQA1*02:01",
    "DQB1*03:01": "DQA1*05:01",
    "DQB1*05:01": "DQA1*01:01",
    "DQB1*03:02": "DQA1*03:01",
    "DQB1*03:01": "DQA1*03:02",
    "DQB1*06:03": "DQA1*01:03",
    "DQB1*03:03": "DQA1*02:01",
    "DQB1*06:04": "DQA1*01:02",
    "DQB1*04:02": "DQA1*04:01",
    "DQB1*05:03": "DQA1*01:04",
    "DQB1*05:02": "DQA1*01:02",
    # "DQB1*06:01"
}

# DPB_FOR = {
#     "DPB1*11:01",
#     "DPB1*04:01",
#     "DPB1*02:01",
# }

B2M_HUMAN_SEQ = "MSRSVALAVLALLSLSGLEAIQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYYTEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"


class HLACodeWebConverter(MHCCodeConverter):
    def __init__(self):

        self.cache = pl.read_csv(hla_uniprot_data_cache)
        self.cache_lock = threading.Lock()

    def get_sequence(self, official_name, top_only=False):
        """
        https://hla.alleles.org/nomenclature/naming.html
        """
        if official_name == "B2M":
            return B2M_HUMAN_SEQ
        elif official_name.startswith("DRA"):
            sequence = DRA_SEQ
        else:
            with self.cache_lock:
                # search cache
                search = self.cache.filter(
                    pl.col("name") == official_name
                ).select("sequence")
                if search.height == 1:
                    sequence = search.item()
                else:

                    params = {
                        "query": f"or(startsWith(name, '{official_name}:'), eq(name, '{official_name}'))",
                    }
                    # https://www.ebi.ac.uk/ipd/imgt/hla/alleles/
                    result = requests.get(
                        "https://www.ebi.ac.uk/cgi-bin/ipd/api/allele",
                        params=params,
                    )
                    result.raise_for_status()

                    json_result = result.json()

                    if json_result["meta"]["total"] == 0:
                        raise ValueError("No allele found in IPD-IMGT/HLA")

                    accessions = [
                        json_result["data"][i]["accession"]
                        for i in range(len(json_result["data"]))
                    ]
                    sequence = None

                    for accession in accessions:
                        result = requests.get(
                            f"https://www.ebi.ac.uk/cgi-bin/ipd/api/allele/{accession}"
                        )

                        result.raise_for_status()

                        json_result = result.json()
                        if json_result["status"] == "Public":
                            sequence = json_result["sequence"]["protein"]
                            break

                    if sequence is None:
                        raise ValueError(
                            f"No public sequence found for name {official_name}"
                        )

                    # cache the result
                    to_cache = pl.DataFrame(
                        {
                            "name": [official_name],
                            "sequence": [sequence],
                            "accession": [accession],
                        }
                    )
                    self.cache = pl.concat([self.cache, to_cache])
                    self.cache.write_csv(hla_uniprot_data_cache)

        if top_only:
            sequence = self._slice_to_topological(official_name, sequence)

        return sequence

    def _slice_to_topological(self, official_name, sequence):
        if official_name.startswith("DRB"):
            return sequence[DRB_EXTRACELLULAR_TOPOLOGICAL_DOMAIN]
        elif official_name.startswith("DRA"):
            return DRA_EXTRACELLULAR_TOPOLOGICAL_SEQ
        elif official_name == "B2M":
            return sequence
        elif official_name.startswith("DQA1"):
            return sequence[DQA1_EXTRACELLULAR_TOPOLOGICAL_DOMAIN]
        elif official_name.startswith("DQA2"):
            return sequence[DQA2_EXTRACELLULAR_TOPOLOGICAL_DOMAIN]
        elif official_name.startswith("DQB1"):
            return sequence[DQB1_EXTRACELLULAR_TOPOLOGICAL_DOMAIN]
        elif official_name.startswith("DQB2"):
            return sequence[DQB2_EXTRACELLULAR_TOPOLOGICAL_DOMAIN]
        elif official_name.startswith("A"):
            return sequence[A_EXTRACELLULAR_TOPOLOGICAL_DOMAIN]
        elif official_name.startswith("B"):
            return sequence[B_EXTRACELLULAR_TOPOLOGICAL_DOMAIN]
        elif official_name.startswith("C"):
            return sequence[C_EXTRACELLULAR_TOPOLOGICAL_DOMAIN]
        elif official_name.startswith("E"):
            return sequence[E_EXTRACELLULAR_TOPOLOGICAL_DOMAIN]
        elif official_name.startswith("DPB1"):
            return sequence[DPB1_EXTRACELLULAR_TOPOLOGICAL_DOMAIN]
        elif official_name.startswith("DPA1"):
            return sequence[DPA1_EXTRACELLULAR_TOPOLOGICAL_DOMAIN]
        else:
            raise ValueError("No topological domain for sequence")


# These are class I molecules
# Bind to B2M
H2_CLASS_1_SEQ_FOR = {
    # https://www.uniprot.org/uniprotkb/P01899/entry
    "H2-Db": "MGAMAPRTLLLLLAAALAPTQTRAGPHSMRYFETAVSRPGLEEPRYISVGYVDNKEFVRFDSDAENPRYEPRAPWMEQEGPEYWERETQKAKGQEQWFRVSLRNLLGYYNQSAGGSHTLQQMSGCDLGSDWRLLRGYLQFAYEGRDYIALNEDLKTWTAADMAAQITRRKWEQSGAAEHYKAYLEGECVEWLHRYLKNGNATLLRTDSPKAHVTHHPRSKGEVTLRCWALGFYPADITLTWQLNGEELTQDMELVETRPAGDGTFQKWASVVVPLGKEQNYTCRVYHEGLPEPLTLRWEPPPSTDSYMVIVAVLGVLGAMAIIGAVVAFVMKRRRNTGGKGGDYALAPGSQSSEMSLRDCKA",
    # https://www.uniprot.org/uniprotkb/P01901/entry
    "H2-Kb": "MVPCTLLLLLAAALAPTQTRAGPHSLRYFVTAVSRPGLGEPRYMEVGYVDDTEFVRFDSDAENPRYEPRARWMEQEGPEYWERETQKAKGNEQSFRVDLRTLLGYYNQSKGGSHTIQVISGCEVGSDGRLLRGYQQYAYDGCDYIALNEDLKTWTAADMAALITKHKWEQAGEAERLRAYLEGTCVEWLRRYLKNGNATLLRTDSPKAHVTHHSRPEDKVTLRCWALGFYPADITLTWQLNGEELIQDMELVETRPAGDGTFQKWASVVVPLGKEQYYTCHVYHQGLPEPLTLRWEPPPSTVSNMATVAVLVVLGAAIVTGAVVAFVMKMRRRNTGGKGGDYALAPGSQTSDLSLPDCKVMVHDPHSLA",
    # https://www.uniprot.org/uniprotkb/P01897/entry
    "H2-Ld": "MGAMAPRTLLLLLAAALAPTQTRAGPHSMRYFETAVSRPGLGEPRYISVGYVDNKEFVRFDSDAENPRYEPQAPWMEQEGPEYWERITQIAKGQEQWFRVNLRTLLGYYNQSAGGTHTLQWMYGCDVGSDGRLLRGYEQFAYDGCDYIALNEDLKTWTAADMAAQITRRKWEQAGAAEYYRAYLEGECVEWLHRYLKNGNATLLRTDSPKAHVTHHPRSKGEVTLRCWALGFYPADITLTWQLNGEELTQDMELVETRPAGDGTFQKWASVVVPLGKEQNYTCRVYHEGLPEPLTLRWEPPPSTDSYMVIVAVLGVLGAMAIIGAVVAFVMKRRRNTGGKGGDYALAPGSQSSEMSLRDCKA",
    # https://www.uniprot.org/uniprotkb/P01902/entry
    "H2-Kd": "MAPCTLLLLLAAALAPTQTRAGPHSLRYFVTAVSRPGLGEPRFIAVGYVDDTQFVRFDSDADNPRFEPRAPWMEQEGPEYWEEQTQRAKSDEQWFRVSLRTAQRYYNQSKGGSHTFQRMFGCDVGSDWRLLRGYQQFAYDGRDYIALNEDLKTWTAADTAALITRRKWEQAGDAEYYRAYLEGECVEWLRRYLELGNETLLRTDSPKAHVTYHPRSQVDVTLRCWALGFYPADITLTWQLNGEDLTQDMELVETRPAGDGTFQKWAAVVVPLGKEQNYTCHVHHKGLPEPLTLRWKLPPSTVSNTVIIAVLVVLGAAIVTGAVVAFVMKMRRNTGGKGVNYALAPGSQTSDLSLPDGKVMVHDPHSLA",
}
H2_CLASS_1_EXTRACELLULAR_TOP_DOMAIN_FOR = {
    "H2-Db": slice(24, 309),
    "H2-Kb": slice(21, 305),
    "H2-Ld": slice(24, 309),
    "H2-Kd": slice(21, 305),
}

# Class 2 molecules
H2_CLASS_2_SEQ_FOR = {
    # https://www.uniprot.org/uniprotkb/Q9CQ70/entry
    # https://www.informatics.jax.org/marker/MGI:103070
    "H2-IAb": "MARKRQRRRRRKVTRSQRAELQFPVSRVDRFLREGNYSRRLSSSAPVFLAGVLEYLTSNILELAGEVAHTTGRKRIAPEHVCRVVQNNEQLHQLFKQGGTSVFEPPEPDDN"
}
H2_CLASS_2_EXTRACELLULAR_TOP_DOMAIN_FOR = {}


class H2CodeConverter(MHCCodeConverter):
    # https://biology.stackexchange.com/questions/76589/mouse-h2-allele-sequences
    # https://www.informatics.jax.org/
    def get_sequence(self, official_name):
        return super().get_sequence(official_name)
