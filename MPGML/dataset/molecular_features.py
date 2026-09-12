"""RDKit fingerprints and the 21 physicochemical descriptors used by MPGML.

The legacy ``pubchem`` option computes an 881-bit Morgan fingerprint. It is
retained to reproduce existing feature caches and is not a PubChem key set.
"""

import numpy as np

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, MACCSkeys, rdMolDescriptors
from rdkit import RDLogger as _RDLogger

_RDLogger.DisableLog("rdApp.*")

try:
    from rdkit.Chem import rdFingerprintGenerator

    USE_NEW_FP_API = True
except ImportError:
    rdFingerprintGenerator = None
    USE_NEW_FP_API = False


FINGERPRINT_DIMS = {
    "maccs": 167,
    "erg": 441,
    "pubchem": 881,
    "morgan": 1024,
    "mixed": 1489,
}

DESCRIPTOR_DIM = 21
DESCRIPTOR_NAMES = [
    "MolWt",
    "NumAtoms",
    "NumHeavyAtoms",
    "NumHeteroatoms",
    "MolLogP",
    "TPSA",
    "NumHDonors",
    "NumHAcceptors",
    "NumAromaticRings",
    "NumSaturatedRings",
    "NumAliphaticRings",
    "NumRings",
    "NumRotatableBonds",
    "NumAromaticBonds",
    "NumSaturatedBonds",
    "BertzCT",
    "BalabanJ",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
]
_MORGAN_GENERATORS = {}


def get_fingerprint_dim(fp_type):
    fp_type = fp_type.lower()
    if fp_type not in FINGERPRINT_DIMS:
        raise ValueError(f"Unknown fingerprint type: {fp_type}")
    return FINGERPRINT_DIMS[fp_type]


def get_descriptor_dim():
    return DESCRIPTOR_DIM


def get_descriptor_names():
    return tuple(DESCRIPTOR_NAMES)


def _bitvect_to_numpy(fp, n_bits):
    arr = np.zeros((n_bits,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def _get_morgan_generator(radius, n_bits):
    key = (radius, n_bits)
    if key not in _MORGAN_GENERATORS:
        _MORGAN_GENERATORS[key] = rdFingerprintGenerator.GetMorganGenerator(
            radius=radius, fpSize=n_bits
        )
    return _MORGAN_GENERATORS[key]


def calculate_maccs_fingerprint(mol):
    if mol is None:
        return np.zeros(FINGERPRINT_DIMS["maccs"], dtype=np.float32)
    try:
        fp = MACCSkeys.GenMACCSKeys(mol)
        return _bitvect_to_numpy(fp, FINGERPRINT_DIMS["maccs"])
    except Exception:
        return np.zeros(FINGERPRINT_DIMS["maccs"], dtype=np.float32)


def calculate_erg_fingerprint(mol):
    if mol is None:
        return np.zeros(FINGERPRINT_DIMS["erg"], dtype=np.float32)
    try:
        fp = AllChem.GetErGFingerprint(
            mol, fuzzIncrement=0.3, maxPath=21, minPath=1
        )
        arr = np.asarray(fp, dtype=np.float32)
        dim = FINGERPRINT_DIMS["erg"]
        if arr.shape[0] < dim:
            arr = np.pad(arr, (0, dim - arr.shape[0]), mode="constant")
        elif arr.shape[0] > dim:
            arr = arr[:dim]
        return arr
    except Exception:
        return np.zeros(FINGERPRINT_DIMS["erg"], dtype=np.float32)


def calculate_morgan_fingerprint(mol, radius=2, n_bits=1024):
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    try:
        if USE_NEW_FP_API:
            fp = _get_morgan_generator(radius=radius, n_bits=n_bits).GetFingerprint(mol)
            return _bitvect_to_numpy(fp, n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        return _bitvect_to_numpy(fp, n_bits)
    except Exception:
        return np.zeros(n_bits, dtype=np.float32)


def calculate_pubchem_fingerprint(mol):
    """Return the existing 881-bit Morgan substitute, not PubChem keys.

    The option name and vector width are retained for checkpoint and cache
    compatibility. Changing this to true PubChem keys requires retraining.
    """
    return calculate_morgan_fingerprint(
        mol, radius=2, n_bits=FINGERPRINT_DIMS["pubchem"]
    )


def calculate_fingerprint(mol, fp_type="mixed"):
    """Build one fingerprint or concatenate MACCS, ErG, and Morgan-881."""
    fp_type = fp_type.lower()
    if fp_type == "maccs":
        return calculate_maccs_fingerprint(mol)
    if fp_type == "erg":
        return calculate_erg_fingerprint(mol)
    if fp_type == "pubchem":
        return calculate_pubchem_fingerprint(mol)
    if fp_type == "morgan":
        return calculate_morgan_fingerprint(mol)
    if fp_type == "mixed":
        return np.concatenate(
            [
                calculate_maccs_fingerprint(mol),
                calculate_erg_fingerprint(mol),
                calculate_pubchem_fingerprint(mol),
            ]
        ).astype(np.float32)
    raise ValueError(f"Unknown fingerprint type: {fp_type}")


def calculate_molecular_descriptors(mol):
    """Return raw descriptor values in ``DESCRIPTOR_NAMES`` order."""
    if mol is None:
        return np.zeros(DESCRIPTOR_DIM, dtype=np.float32)

    try:
        aromatic_bonds = 0
        saturated_bonds = 0
        for bond in mol.GetBonds():
            if bond.GetIsAromatic():
                aromatic_bonds += 1
            if (
                not bond.GetIsAromatic()
                and bond.GetBondType() == Chem.rdchem.BondType.SINGLE
            ):
                saturated_bonds += 1

        balaban_j = Descriptors.BalabanJ(mol) if mol.GetNumBonds() > 0 else 0.0
        values = [
            Descriptors.MolWt(mol),
            mol.GetNumAtoms(),
            mol.GetNumHeavyAtoms(),
            rdMolDescriptors.CalcNumHeteroatoms(mol),
            Crippen.MolLogP(mol),
            Descriptors.TPSA(mol),
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            rdMolDescriptors.CalcNumSaturatedRings(mol),
            rdMolDescriptors.CalcNumAliphaticRings(mol),
            rdMolDescriptors.CalcNumRings(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            aromatic_bonds,
            saturated_bonds,
            Descriptors.BertzCT(mol),
            balaban_j,
            Descriptors.HallKierAlpha(mol),
            Descriptors.Kappa1(mol),
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
        ]
        arr = np.asarray(values, dtype=np.float32)
        return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        return np.zeros(DESCRIPTOR_DIM, dtype=np.float32)


def smiles_to_mol(smiles):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None
