# Data preparation

MPGML reads local assay CSV files. It does not download datasets, reconstruct
ToxCast provider groups, or infer task splits from assay names. A public archive
for the exact processed input CSVs and their preparation procedure has not yet
been specified. Obtain the experiment CSVs from the authors before attempting
to reproduce the manuscript results; a similarly named upstream dataset may
have different rows or column ordering.

Local CSVs and caches are preserved in this workspace and excluded from Git.
This document is the only file under `data/` intended for the source repository.

## Directory structure and CSV format

The default `--data_root` is `data`. For any supported dataset name, use:

```text
data/
  muv/
    muv.csv
  toxcast-TOX21/
    toxcast-TOX21.csv
  toxcast-Tanguay/
    toxcast-Tanguay.csv
```

The general path is `<data_root>/<dataset>/<dataset>.csv`. Dataset names are
accepted without regard to case and normalized to the spelling in the table
below. Directory and file names must use that canonical spelling on systems
with case-sensitive paths. `toxcast` alone is not a supported dataset name.

CSV files must contain a header followed by molecule rows. Labels are numeric
binary values, `0` or `1`; an empty label cell denotes a missing observation.
Missing labels are stored as NaN and excluded from that target task's sampling
pools. Quoted CSV fields and blank lines are supported. Column positions are
fixed by the loader, so retain the original assay order and metadata columns.

Column positions in this table are **one-based**:

| Dataset family | SMILES column | Label columns | Other columns |
| --- | --- | --- | --- |
| `pcba` | Last | 1–128 | Supplied CSV includes a molecule identifier before SMILES |
| `muv` | Last | 1–17 | Supplied CSV includes a molecule identifier before SMILES |
| `tox21` | Last | 1–12 | Supplied CSV includes a molecule identifier before SMILES |
| `sider` | 1 | 2–28 | None expected |
| `toxcast-*` | 1 | 2 through the number of tasks plus 1 | None expected |

The loader parses SMILES with RDKit, skips molecules that cannot be parsed, and
kekulizes the retained molecules. Graph nodes contain categorical atomic-number
and chirality indices. Each bond becomes two directed edges carrying bond-type
and bond-direction indices. Edge indices are sorted before caching.

The loader does not deduplicate molecules, rebalance assays globally, or create
a molecule/scaffold holdout. The split is over **assay tasks**; different tasks
can refer to the same molecules.

## Fixed task splits

Task IDs below are **zero-based indices into the label columns**, excluding
SMILES and molecule metadata. For all datasets except PCBA, the first training
tasks are followed by the test tasks. PCBA holds out the first and last five
tasks. Changing CSV label-column order changes the experimental split.

| Dataset | Total tasks | Training tasks | Test tasks | Training task IDs | Test task IDs |
| --- | ---: | ---: | ---: | --- | --- |
| `pcba` | 128 | 118 | 10 | 5–122 | 0–4, 123–127 |
| `muv` | 17 | 12 | 5 | 0–11 | 12–16 |
| `tox21` | 12 | 9 | 3 | 0–8 | 9–11 |
| `sider` | 27 | 21 | 6 | 0–20 | 21–26 |
| `toxcast-APR` | 43 | 33 | 10 | 0–32 | 33–42 |
| `toxcast-ATG` | 146 | 106 | 40 | 0–105 | 106–145 |
| `toxcast-BSK` | 115 | 84 | 31 | 0–83 | 84–114 |
| `toxcast-CEETOX` | 14 | 10 | 4 | 0–9 | 10–13 |
| `toxcast-CLD` | 19 | 14 | 5 | 0–13 | 14–18 |
| `toxcast-NVS` | 139 | 100 | 39 | 0–99 | 100–138 |
| `toxcast-OT` | 15 | 11 | 4 | 0–10 | 11–14 |
| `toxcast-TOX21` | 100 | 80 | 20 | 0–79 | 80–99 |
| `toxcast-Tanguay` | 18 | 13 | 5 | 0–12 | 13–17 |

These definitions are in [`dataset/dataset.py`](../dataset/dataset.py). There is
no separate validation-task partition in this loader.

## Support and query episodes

[`dataset/sampler.py`](../dataset/sampler.py) samples without replacement within
each episode, using only molecules whose target assay label is available.
Support and query sets are disjoint within an episode; molecules may appear
again in other episodes or under other target tasks.

- `--n_support K` normally selects `K` negative and `K` positive molecules:
  `2K` support molecules in total. Thus `--n_support 10` means 20 support
  molecules.
- If a class has at most `K` labeled molecules, the sampler reserves one of
  that class for the query set and fills the remaining support slots from the
  other class. Support can therefore be unbalanced, and a class with only one
  labeled molecule contributes no support molecule.
- A training query contains `--n_query` molecules in total. One molecule from
  each class is drawn first, and the remaining query slots are sampled from
  the combined pool of unused molecules. The default query size is 16.
- During inductive evaluation, the query set contains **all** remaining
  labeled molecules for that target assay. `--n_query` controls the query
  batch used during adaptation; `--test_batch_size` controls prediction
  batches. Neither limits the total evaluation query set.
- Invalid requests raise `ValueError`: `K` must be positive, both query
  classes must be present, at least `2K + 2` labeled molecules are required,
  and a training query must fit the remaining pool and contain at least two
  molecules. The sampler does not oversample with replacement or silently
  skip these tasks.

Training draws two independently sampled episodes per selected target task
for the contrastive objective. The second draw is not constrained to be
disjoint from the first draw.

## Molecular fingerprints and descriptors

Auxiliary features are computed from the retained graph records' SMILES by
[`dataset/molecular_features.py`](../dataset/molecular_features.py).

| `--fp_type` | Width | Actual implementation |
| --- | ---: | --- |
| `maccs` | 167 | RDKit MACCS keys |
| `erg` | 441 | RDKit ErG, `fuzzIncrement=0.3`, paths 1–21 |
| `pubchem` | 881 | Morgan fingerprint, radius 2, 881 bits; historical option name |
| `morgan` | 1024 | Morgan fingerprint, radius 2, 1024 bits |
| `mixed` | 1489 | Concatenation of MACCS-167, ErG-441, and Morgan-881 in that order |

**The `pubchem` option does not calculate the PubChem 881 structural keys.**
The existing implementation uses an 881-bit Morgan substitute, including in
`mixed`. Its option name, dimensions, and computation are retained for
compatibility with existing runs. Substituting a real PubChem implementation
would change model inputs, invalidate feature caches, and require retraining.

`--use_descriptor` supplies these 21 descriptors in this exact order:

```text
MolWt, NumAtoms, NumHeavyAtoms, NumHeteroatoms, MolLogP, TPSA,
NumHDonors, NumHAcceptors, NumAromaticRings, NumSaturatedRings,
NumAliphaticRings, NumRings, NumRotatableBonds, NumAromaticBonds,
NumSaturatedBonds, BertzCT, BalabanJ, HallKierAlpha, Kappa1,
Kappa2, Kappa3
```

Here `NumSaturatedBonds` counts non-aromatic single bonds. Descriptor values
that are NaN or infinite are replaced by zero. A failed feature calculation
returns a zero vector for that fingerprint or descriptor calculation.

Descriptors are standardized with a per-column mean and population standard
deviation fitted on **all retained molecules in the selected dataset**. This
is label-free, dataset-wide preprocessing; the statistics are not fitted on
training episodes or support molecules alone. Standard deviations below
`1e-6` are replaced by 1. Both statistics are saved in the feature cache.
Fingerprints are not standardized by the dataset loader.

## Generated caches and invalidation

The first dataset construction creates `<data_root>/<dataset>/processed/`:

| File | Contents |
| --- | --- |
| `data.pt` | Collated PyG graphs, labels, SMILES, and molecule IDs |
| `index_list.pt` | Pickled molecule-index lists for each task and class |
| `label_matrix.npz` | Label matrix; despite the historical suffix, this is a single NPY array |
| `molecular_features_<fp_type>.pt` | Enabled fingerprint/descriptor tensors and descriptor normalization statistics |
| `pre_filter.pt`, `pre_transform.pt` | PyG preprocessing metadata |

Feature caches are built only when fingerprints or descriptors are enabled.
The cache is reused when its dataset name, fingerprint type, molecule count,
and requested feature dimensions match. Enabling a modality absent from an
existing cache triggers feature rebuilding.

`--rebuild_feature_cache` rebuilds the fingerprint/descriptor cache only. It
does **not** rebuild molecular graphs, labels, task-class index lists, or the
underlying list of molecules.

When a raw CSV, row order, task-column order, graph transformation, or RDKit
preprocessing changes, remove that dataset's complete `processed/` directory
before constructing the dataset again. When only fingerprint or descriptor
computation changes, rebuild its feature cache and retrain any affected model.
The current cache format does not hash raw CSV contents or record an RDKit
version, so equal matrix dimensions alone do not prove cache freshness.

## Local snapshot inspected during repository cleanup

The following counts describe the supplied workspace on 2026-09-12. They are
not asserted to be universal benchmark sizes. CSV counts exclude the header
and empty lines. Cached molecule counts were read from existing NPY headers;
the cleanup did not regenerate caches or execute the dataset loader.

| Dataset | Local CSV molecule rows | Existing cached molecules |
| --- | ---: | ---: |
| `pcba` | 437929 | 437929 |
| `muv` | 93087 | 93087 |
| `tox21` | 7831 | 7831 |
| `sider` | 1427 | 1427 |
| `toxcast-APR` | 1039 | 1034 |
| `toxcast-ATG` | 3423 | 3412 |
| `toxcast-BSK` | 1445 | 1439 |
| `toxcast-CEETOX` | 508 | 502 |
| `toxcast-CLD` | 305 | 302 |
| `toxcast-NVS` | 2130 | 2122 |
| `toxcast-OT` | 1782 | 1773 |
| `toxcast-TOX21` | 8241 | 8223 |
| `toxcast-Tanguay` | 1039 | 1034 |

The local MUV CSV contains 93087 rows, whereas the manuscript reports 93127
molecules. The exact preparation history of that difference is not recorded
in the supplied source files. Reconcile the manuscript counts and release
the corresponding preparation procedure together with the data archive.

Several ToxCast caches contain fewer molecules than their raw CSVs; the
loader's invalid-SMILES filtering can reduce the row count, but these existing
caches were not rebuilt to attribute each excluded row during cleanup.
