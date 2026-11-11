<##
Run one supplied MPGML configuration.
Example: .\scripts\train.ps1 -Dataset muv -Seed 2 -Gpu 0
Additional settings: -ExtraArgs @('--episode', '2000', '--save_last_model')
##>
param(
    [ValidateSet('toxcast-TOX21', 'muv', 'toxcast-Tanguay')]
    [string]$Dataset = 'toxcast-TOX21',
    [int]$Seed = 2,
    [int]$Gpu = 0,
    [string[]]$ExtraArgs = @()
)

$ErrorActionPreference = 'Stop'
$preset = switch ($Dataset) {
    'toxcast-TOX21' { @{ Inner = '1e-1'; Meta = '1e-3' } }
    'muv' { @{ Inner = '5e-1'; Meta = '5e-4' } }
    'toxcast-Tanguay' { @{ Inner = '1e-1'; Meta = '5e-4' } }
}
$trainArgs = @(
    'run.py', '--dataset', $Dataset, '--random_seed', "$Seed",
    '--n_support', '10', '--gpu', "$Gpu",
    '--mol_pretrain_load_path', './pretrained/supervised_contextpred.pth',
    '--inner_lr', $preset.Inner, '--meta_lr', $preset.Meta,
    '--dump_path', 'dump',
    '--use_fingerprint', '--use_descriptor', '--fp_type', 'mixed',
    '--fp_encoder_type', 'typewise', '--fusion_method', 'transformer',
    '--fusion_layers', '3', '--use_cross_modal_attn', '--use_l2_norm'
)
Push-Location -LiteralPath (Join-Path $PSScriptRoot '..')
try {
    & python @trainArgs @ExtraArgs
    if ($LASTEXITCODE -ne 0) {
        throw "MPGML exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
