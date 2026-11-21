#!/usr/bin/env bash
set -e
mkdir -p data/raw
if [ ! -d data/raw/SKAB ]; then
  git clone --depth 1 https://github.com/waico/SKAB.git data/raw/SKAB
fi
