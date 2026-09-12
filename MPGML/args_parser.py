"""Command-line configuration shared by training and checkpoint analysis."""

import argparse

from dataset import FewshotMolDataset


def build_parser():
    parser = argparse.ArgumentParser(
        description="MPGML: multimodal episodic molecular property prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    # Experiment output and device.
    parser.add_argument("--exp_name", default="run", type=str,
                        help="Experiment name")
    parser.add_argument("--dump_path", default="dump", type=str,
                        help="Experiment output directory")
    parser.add_argument("--exp_id", default="", type=str,
                        help="Experiment ID")
    parser.add_argument("--gpu", default='0', type=str, help="CUDA device index; use -1 for CPU")
    parser.add_argument("--random_seed", default=0, type=int)

    # Dataset layout.
    parser.add_argument("--data_root", default='data', type=str)
    parser.add_argument("--dataset", default='sider', type=str,
                        help="Dataset name. ToxCast is split into toxcast-* groups.")
    # Molecular graph encoder. Width-changing pooling is not supported downstream.
    parser.add_argument("--mol_num_layer", default=5, type=int)
    parser.add_argument("--emb_dim", default=300, type=int)
    parser.add_argument("--JK", default='last', choices=['last', 'max'])
    parser.add_argument("--mol_dropout", default=0.1, type=float)
    parser.add_argument("--mol_graph_pooling", default='mean', choices=['mean', 'sum', 'max', 'attention'])
    parser.add_argument("--mol_gnn_type", default='gin', choices=['gin', 'gcn', 'gat', 'graphsage'])
    parser.add_argument("--mol_batch_norm", default=1, type=int, choices=[0, 1])
    parser.add_argument("--mol_pretrain_load_path", default='./pretrained/supervised_contextpred.pth')
    parser.add_argument("--no_pretrain", action='store_const', const=None,
                        dest='mol_pretrain_load_path', help="Initialize the graph encoder from scratch")

    # molecular fingerprint / descriptor fusion
    parser.add_argument("--use_fingerprint", "--USE_FINGERPRINT", action="store_true", default=False,
                        dest="use_fingerprint")
    parser.add_argument("--use_descriptor", "--USE_DESCRIPTOR", action="store_true", default=False,
                        dest="use_descriptor")
    parser.add_argument("--fp_type", "--FP_TYPE", default='mixed', type=str.lower,
                        choices=['maccs', 'erg', 'pubchem', 'morgan', 'mixed'])
    parser.add_argument("--fp_encoder_type", "--FP_ENCODER_TYPE", default='typewise', type=str.lower,
                        choices=['mlp', 'typewise'])
    parser.add_argument("--fp_intermediate_dim", "--FP_INTERMEDIATE_DIM", default=256, type=int)
    parser.add_argument("--desc_intermediate_dim", "--DESC_INTERMEDIATE_DIM", default=128, type=int)
    parser.add_argument("--fusion_method", "--FUSION_METHOD", default='transformer', type=str.lower,
                        choices=['transformer', 'attention', 'concat', 'sum'])
    parser.add_argument("--fusion_layers", "--FUSION_LAYERS", default=2, type=int)
    parser.add_argument("--fusion_heads", "--FUSION_HEADS", default=4, type=int)
    parser.add_argument("--fusion_dropout", "--FUSION_DROPOUT", default=0.1, type=float)
    parser.add_argument("--fusion_temperature", "--FUSION_TEMPERATURE", default=1.0, type=float)
    parser.add_argument("--use_cross_modal_attn", "--USE_CROSS_MODAL_ATTN", action="store_true", default=False,
                        dest="use_cross_modal_attn")
    parser.add_argument("--use_l2_norm", "--USE_L2_NORM", action="store_true", default=False,
                        dest="use_l2_norm")
    parser.add_argument("--modality_dropout", "--MODALITY_DROPOUT", default=0.0, type=float)
    parser.add_argument("--no_fusion_residual", action="store_false", default=True,
                        dest="fusion_residual")
    parser.add_argument("--fusion_residual_gate_init", "--FUSION_RESIDUAL_GATE_INIT",
                        default=-3.0, type=float)
    parser.add_argument("--no_fp_gate", action="store_false", default=True, dest="use_fp_gate")
    parser.add_argument("--no_desc_gate", action="store_false", default=True, dest="use_desc_gate")
    parser.add_argument("--fp_gate_reg", default=0.0, type=float)
    parser.add_argument("--desc_gate_reg", default=0.0, type=float)
    parser.add_argument("--rebuild_feature_cache", "--REBUILD_FEATURE_CACHE", action="store_true", default=False)

    # relation net
    parser.add_argument("--rel_layer", default=2, type=int)
    parser.add_argument("--rel_edge_n_layer", default=2, type=int)
    parser.add_argument("--rel_top_k", default=None, type=int)
    parser.add_argument("--rel_edge_hidden_dim", default=100, type=int)
    parser.add_argument("--rel_dropout", default=0.1, type=float)
    parser.add_argument("--rel_pre_dropout", default=0.1, type=float)
    parser.add_argument("--rel_nan_w", default=1., type=float)
    parser.add_argument("--rel_nan_type", default='nan', type=str, choices=['nan', '0', '1'])
    parser.add_argument("--rel_batch_norm", default=1, type=int)
    parser.add_argument("--rel_edge_type", default=1, type=int)

    # maml
    parser.add_argument("--inner_lr", default=0.5, type=float)
    parser.add_argument("--meta_lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--second_order", default=1, type=int, choices=[0, 1])
    parser.add_argument("--inner_update_step", default=1, type=int)
    parser.add_argument("--lr_check_flag", action="store_true", default=False, help="dump with learning rate check")


    # few-shot
    parser.add_argument("--episode", default=2000, type=int)
    parser.add_argument("--pool_num", default=5, type=int,
                        help="Number of training tasks sampled per outer update")
    parser.add_argument("--n_support", default=10, type=int,
                        help="Support examples per class when available (2-way K-shot)")
    parser.add_argument("--n_query", default=16, type=int,
                        help="Total training query examples per episode")
    parser.add_argument("--eval_step", default=100, type=int)
    parser.add_argument("--test_batch_size", default=64, type=int)
    parser.add_argument("--test_num_workers", default=0, type=int,
                        help="Number of DataLoader workers used during evaluation. Use 0 on Windows.")
    parser.add_argument("--train_auxi_task_num", default=None, type=int)
    parser.add_argument("--test_auxi_task_num", default=None, type=int)
    parser.add_argument("--auxi_task_check_flag", action="store_true", default=False, help="dump with auxiliary task number")

    # contrastive
    parser.add_argument("--nce_t", default=0.08, type=float)
    parser.add_argument("--contr_w", default=0.05, type=float)

    # relation learning
    parser.add_argument("--rel_aug", default=1, type=int, choices=[0, 1])
    parser.add_argument("--rel_ratio", default=1, type=float)

    # information bottleneck
    parser.add_argument("--ib_aug", default=1, type=int, choices=[0, 1])
    parser.add_argument("--ib_ratio", default=5e-2, type=float)

    parser.add_argument("--no_save_best_model", action="store_false", default=True,
                        dest="save_best_model",
                        help="Disable saving best_model.pkl when held-out-task ROC-AUC improves.")
    parser.add_argument("--save_last_model", action="store_true", default=False,
                        help="Save last_model.pkl at the end of training.")

    return parser


def finalize_args(args, parser=None):
    """Resolve dataset names and reject structurally invalid configurations."""
    def fail(message):
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    dataset_name = FewshotMolDataset.canonical_name(args.dataset)
    if dataset_name is None:
        toxcast_names = [name for name in FewshotMolDataset.valid_names() if name.startswith('toxcast-')]
        if str(args.dataset).lower() == 'toxcast':
            message = (
                '--dataset toxcast is not a standalone dataset in this repo. '
                f'Use one ToxCast group instead: {", ".join(toxcast_names)}.'
            )
        else:
            message = (
                f'unknown --dataset {args.dataset!r}. '
                f'Valid datasets: {", ".join(FewshotMolDataset.valid_names())}.'
            )
        fail(message)
    args.dataset = dataset_name

    if args.rel_top_k is None:
        args.rel_top_k = args.n_support - 1 if args.n_support > 1 else 1

    for name in ('episode', 'eval_step', 'pool_num', 'n_support', 'inner_update_step',
                 'test_batch_size', 'emb_dim', 'fusion_layers', 'fusion_heads'):
        if getattr(args, name) < 1:
            fail(f'--{name} must be positive.')
    if args.n_query < 2:
        fail('--n_query must be at least 2 to sample both classes.')
    if args.mol_num_layer < 2:
        fail('--mol_num_layer must be at least 2.')
    if args.JK not in ('last', 'max') or args.mol_graph_pooling not in ('mean', 'sum', 'max', 'attention'):
        fail('Use --JK last/max and pooling mean/sum/max/attention; other paths are not supported by MPGML.')
    uses_attention = (args.use_fingerprint or args.use_descriptor) and (
        args.fusion_method == 'transformer' or args.use_cross_modal_attn
    )
    if uses_attention and args.emb_dim % args.fusion_heads:
        fail('--emb_dim must be divisible by --fusion_heads for attention.')
    for name in ('train_auxi_task_num', 'test_auxi_task_num'):
        value = getattr(args, name)
        if value is not None and value < 0:
            fail(f'--{name} must be non-negative.')
    return args


def args_parser(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return finalize_args(args, parser)
