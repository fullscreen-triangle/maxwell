#!/usr/bin/env sh
# Fetch the RepoRT files used by the analysis (sparse, shallow; about 300 MB including git objects).
# RepoRT is licensed CC-BY-SA 4.0. The analysis used commit 420b550eca799be282a69c5cd7c2911a1593cbfa.
set -e
cd "$(dirname "$0")"
mkdir -p data
cd data
if [ ! -d RepoRT ]; then
  git clone --depth 50 --filter=blob:none --no-checkout --sparse https://github.com/michaelwitting/RepoRT.git
fi
cd RepoRT
git sparse-checkout set --no-cone \
  '/resources/**' \
  '/processed_data/*/*_metadata.tsv' \
  '/processed_data/*/*_rtdata_canonical_success.tsv' \
  '/processed_data/*/*_gradient.tsv' \
  '/processed_data/*/*_info.tsv'
git checkout 420b550eca799be282a69c5cd7c2911a1593cbfa 2>/dev/null || git checkout
git log -1 --format='RepoRT at %H (%ci)'
