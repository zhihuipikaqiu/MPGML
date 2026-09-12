# Pretrained graph encoder

The training presets expect `supervised_contextpred.pth` in this directory. It is a PyTorch state dictionary for the molecular GNN, loaded into `GNN_Encoder.gnn` with strict key and shape matching.

The supplied configuration uses a five-layer GIN, embedding width 300, batch normalization, and the atom/bond vocabularies in `dataset/mol_features.py`. A full MPGML experiment checkpoint (`best_model.pkl`) is a different artifact and cannot replace this backbone-only file.

The architecture and checkpoint naming follow [Strategies for Pre-Training Graph Neural Networks](https://github.com/snap-stanford/pretrain-gnns). Consult that project's chemistry instructions for upstream artifacts. Confirm the exact origin and redistribution record of the local file when preparing the final release; the filename alone does not establish artifact identity.

## Local snapshot identification

The file present during source organization had:

| Field | Value |
| --- | --- |
| Filename | `supervised_contextpred.pth` |
| Size | 7,452,448 bytes |
| SHA-256 | `107197f159e9a26ed9a026b655560ac341d2d7ddec4a8d2540e83c9f87e256ac` |

This checksum identifies the local artifact; it has not been compared against an upstream download. Weights are excluded from Git and the source archive. A release download location has not been supplied.

Use `--mol_pretrain_load_path /path/to/checkpoint.pth` to select a backbone, or `--no_pretrain` to train from scratch. Changing the pretraining choice changes the experiment. A complete MPGML checkpoint already includes the backbone weights, so `explain.py` does not require this separate file.
