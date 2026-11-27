#!/usr/bin/env pwsh
# Downloads small public text corpora into ./datasets (git-ignored).
#
# Usage: ./scripts/fetch-corpora.ps1 [-Output datasets]
#
# TinyShakespeare (~1 MB) and TinyStories valid split (~22 MB) are small enough
# to train the device example on a laptop. Project Gutenberg and WikiText are
# included but tolerate outages: failures are warnings, not errors.

param([string]$Output = "datasets")

$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path $Output | Out-Null

$targets = @(
    @{ Name = "tinyshakespeare.txt"; Url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt" },
    @{ Name = "tinystories-valid.txt"; Url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt" },
    @{ Name = "alice.txt"; Url = "https://www.gutenberg.org/cache/epub/11/pg11.txt" },
    @{ Name = "wikitext-2.zip"; Url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip" }
)

foreach ($target in $targets) {
    $destination = Join-Path $Output $target.Name
    if (Test-Path $destination) {
        Write-Host "skip $($target.Name) (already present)"
        continue
    }
    try {
        Invoke-WebRequest -Uri $target.Url -OutFile $destination -UserAgent "Mozilla/5.0" -TimeoutSec 180
        $size = (Get-Item $destination).Length
        Write-Host "downloaded $($target.Name) ($size bytes)"
    }
    catch {
        Write-Warning "failed $($target.Name): $($_.Exception.Message)"
    }
}

Write-Host "corpora in $Output; train with --corpus <path> --corpus-chars N"
