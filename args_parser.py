"""Command-line configuration shared by training and checkpoint analysis."""

import argparse
import warnings

from dataset import FewshotMolDataset


def build_parser():
    parser = argparse.ArgumentParser(
        description="MPGML: prototype-guided few-shot molecular property prediction.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--exp_name", default="run")
    parser.add_argument("--dump_path", default="dump")
    parser.add_argument("--exp_id", default="")
    parser.add_argument("--gpu", default="0", help="CUDA device index. Use -1 for CPU.")
    parser.add_argument("--random_seed", default=0, type=int)

    parser.add_argument("--data_root", default="data")
    parser.add_argument("--dataset", default="sider", help="ToxCast uses separate toxcast-* groups.")
    parser.add_argument("--mol_num_layer", default=5, type=int)
    parser.add_argument("--emb_dim", default=300, type=int)
    parser.add_argument("--JK", default="last", choices=["last", "max"])
    parser.add_argument("--mol_dropout", default=0.1, type=float)
    parser.add_argument("--mol_graph_pooling", default="mean", choices=["mean", "sum", "max", "attention"])
    parser.add_argument("--mol_gnn_type", default="gin", choices=["gin", "gcn", "gat", "graphsage"])
    parser.add_argument("--mol_batch_norm", default=1, type=int, choices=[0, 1])
    parser.add_argument("--mol_pretrain_load_path", default="./pretrained/supervised_contextpred.pth")
    parser.add_argument("--no_pretrain", action="store_const", const=None,
                        dest="mol_pretrain_load_path", help="Initialize the graph encoder from scratch.")

    parser.add_argument("--use_fingerprint", "--USE_FINGERPRINT", action="store_true",
                        dest="use_fingerprint")
    parser.add_argument("--no_fingerprint", action="store_false", dest="use_fingerprint")
    parser.add_argument("--use_descriptor", "--USE_DESCRIPTOR", action="store_true",
                        dest="use_descriptor")
    parser.add_argument("--no_descriptor", action="store_false", dest="use_descriptor")
    parser.add_argument("--fp_type", "--FP_TYPE", default="mixed", type=str.lower,
                        choices=["maccs", "erg", "pubchem", "morgan", "mixed"])
    parser.add_argument("--fp_encoder_type", "--FP_ENCODER_TYPE", default="typewise", type=str.lower,
                        choices=["mlp", "typewise"], help="Use mlp for the w/o-FR ablation.")
    parser.add_argument("--fp_intermediate_dim", "--FP_INTERMEDIATE_DIM", default=256, type=int)
    parser.add_argument("--desc_intermediate_dim", "--DESC_INTERMEDIATE_DIM", default=128, type=int)
    parser.add_argument("--fusion_method", "--FUSION_METHOD", default="transformer", type=str.lower,
                        choices=["transformer", "attention", "concat", "sum"])
    parser.add_argument("--fusion_layers", "--FUSION_LAYERS", default=3, type=int)
    parser.add_argument("--fusion_heads", "--FUSION_HEADS", default=4, type=int)
    parser.add_argument("--fusion_dropout", "--FUSION_DROPOUT", default=0.1, type=float)
    parser.add_argument("--fusion_temperature", "--FUSION_TEMPERATURE", default=1.0, type=float)
    parser.add_argument("--use_cross_modal_attn", "--USE_CROSS_MODAL_ATTN", action="store_true",
                        dest="use_cross_modal_attn")
    parser.add_argument("--no_cross_modal_attn", action="store_false", dest="use_cross_modal_attn")
    parser.add_argument("--use_l2_norm", "--USE_L2_NORM", action="store_true", dest="use_l2_norm")
    parser.add_argument("--no_l2_norm", action="store_false", dest="use_l2_norm")
    parser.add_argument("--modality_dropout", "--MODALITY_DROPOUT", default=0.0, type=float)
    parser.add_argument("--no_fusion_residual", action="store_false", default=True, dest="fusion_residual")
    parser.add_argument("--fusion_residual_gate_init", "--FUSION_RESIDUAL_GATE_INIT",
                        default=-3.0, type=float)
    parser.add_argument("--no_fp_gate", action="store_false", default=True, dest="use_fp_gate")
    parser.add_argument("--no_desc_gate", action="store_false", default=True, dest="use_desc_gate")
    parser.add_argument("--fp_gate_reg", default=1e-4, type=float)
    parser.add_argument("--desc_gate_reg", default=1e-4, type=float)
    parser.add_argument("--rebuild_feature_cache", "--REBUILD_FEATURE_CACHE", action="store_true")
    parser.set_defaults(
        use_fingerprint=True, use_descriptor=True, use_cross_modal_attn=True, use_l2_norm=True,
    )

    parser.add_argument("--inner_lr", default=0.5, type=float)
    parser.add_argument("--meta_lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=5e-5, type=float)
    parser.add_argument("--second_order", default=1, type=int, choices=[0, 1])
    parser.add_argument("--inner_update_step", default=1, type=int)
    parser.add_argument("--focal_gamma", default=2.0, type=float)
    parser.add_argument("--no_prototype_guidance", action="store_false", dest="prototype_guidance",
                        help="Adapt a shared linear initialization for the w/o-PG ablation.")
    parser.add_argument("--lr_check_flag", action="store_true", help="Group runs by learning rates.")

    parser.add_argument("--episode", default=2000, type=int)
    parser.add_argument("--pool_num", default=5, type=int, help="Maximum tasks per outer update.")
    parser.add_argument("--n_support", default=10, type=int, help="Support examples per class when available.")
    parser.add_argument("--n_query", default=16, type=int, help="Total training query examples per episode.")
    parser.add_argument("--eval_step", default=100, type=int)
    parser.add_argument("--test_batch_size", default=64, type=int)
    parser.add_argument("--test_num_workers", default=0, type=int)
    parser.add_argument("--no_save_best_model", action="store_false", default=True, dest="save_best_model")
    parser.add_argument("--save_last_model", action="store_true")

    # Accept the option found in older public training commands.
    parser.add_argument("--ib_ratio", type=float, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    return parser


def finalize_args(args, parser=None):
    """Resolve dataset names and reject structurally invalid configurations."""
    def fail(message):
        if parser is not None:
            parser.error(message)
        raise ValueError(message)

    dataset_name = FewshotMolDataset.canonical_name(args.dataset)
    if dataset_name is None:
        if str(args.dataset).lower() == "toxcast":
            groups = [name for name in FewshotMolDataset.valid_names() if name.startswith("toxcast-")]
            fail(f"Choose one ToxCast group with --dataset: {', '.join(groups)}.")
        fail(f"Unknown dataset {args.dataset!r}. Valid datasets: {', '.join(FewshotMolDataset.valid_names())}.")
    args.dataset = dataset_name

    for name in (
        "episode", "eval_step", "pool_num", "n_support", "inner_update_step",
        "test_batch_size", "emb_dim", "fusion_layers", "fusion_heads",
        "fp_intermediate_dim", "desc_intermediate_dim",
    ):
        if getattr(args, name) < 1:
            fail(f"--{name} must be positive.")
    if args.n_query < 2:
        fail("--n_query must be at least 2 to sample both classes.")
    if args.mol_num_layer < 2:
        fail("--mol_num_layer must be at least 2.")
    if args.test_num_workers < 0:
        fail("--test_num_workers must be non-negative.")
    for name in ("focal_gamma", "fp_gate_reg", "desc_gate_reg"):
        if getattr(args, name) < 0:
            fail(f"--{name} must be non-negative.")
    if args.use_l2_norm and not args.fusion_temperature > 0:
        fail("--fusion_temperature must be positive.")
    uses_attention = args.use_fingerprint and (
        args.fusion_method == "transformer" or args.use_cross_modal_attn
    )
    if uses_attention and args.emb_dim % args.fusion_heads:
        fail("--emb_dim must be divisible by --fusion_heads for attention.")
    if hasattr(args, "ib_ratio"):
        warnings.warn("--ib_ratio is a deprecated option from the previous training entry point.", FutureWarning)
        del args.ib_ratio
    return args


def args_parser(argv=None):
    parser = build_parser()
    return finalize_args(parser.parse_args(argv), parser)
