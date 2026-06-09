#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

cargo install --path crates/shirl-cli
