#!/usr/bin/env bash
set -euo pipefail

# University of Toronto CIFAR-10 mirror, pinned to a specific repository commit.
revision=0b2714987fa478483af9968de7c934580d0bb9a2
data_dir=${1:-data}
mkdir -p "$data_dir"

curl -L --fail --retry 3 -o "$data_dir/hf-train.parquet" \
  "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/$revision/plain_text/train-00000-of-00001.parquet"
curl -L --fail --retry 3 -o "$data_dir/hf-test.parquet" \
  "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/$revision/plain_text/test-00000-of-00001.parquet"

printf '%s  %s\n' \
  '8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883' "$data_dir/hf-train.parquet" \
  '841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de' "$data_dir/hf-test.parquet" \
  | shasum -a 256 -c -
