#!/usr/bin/env sh
# Downloads small public text corpora into ./datasets (git-ignored).
#
# Usage: ./scripts/fetch-corpora.sh [output-dir]
#
# TinyShakespeare (~1 MB) and TinyStories valid split (~22 MB) are small enough
# to train the device example on a laptop. Project Gutenberg and WikiText are
# included but tolerate outages: failures are warnings, not errors.

set -u
output="${1:-datasets}"
mkdir -p "$output"

fetch() {
    name="$1"
    url="$2"
    if [ -s "$output/$name" ] && [ "$(wc -c < "$output/$name")" -gt 4096 ]; then
        echo "skip $name (already present)"
        return
    fi
    if curl -L --fail --silent --show-error --max-time 300 -A "Mozilla/5.0" \
        -o "$output/$name" "$url"; then
        echo "downloaded $name"
    else
        echo "warning: failed $name" >&2
        rm -f "$output/$name"
    fi
}

fetch tinyshakespeare.txt "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
fetch tinystories-valid.txt "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
fetch alice.txt "https://www.gutenberg.org/cache/epub/11/pg11.txt"
fetch wikitext-2.zip "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-raw-v1.zip"

echo "corpora in $output; train with --corpus <path> --corpus-chars N"
