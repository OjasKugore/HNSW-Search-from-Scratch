#!/bin/sh
set -eu
cd "$(dirname "$0")"
echo "Preparing Vector Search…"
make -s app
exec open VectorSearch.app
