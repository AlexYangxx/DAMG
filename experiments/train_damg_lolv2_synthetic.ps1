[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$TrainingArguments
)

$projectRoot = Split-Path -Parent $PSScriptRoot
& python (Join-Path $projectRoot 'train.py') --dataset lolv2_syn @TrainingArguments
exit $LASTEXITCODE
