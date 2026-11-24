#!/bin/sh
set -eu

demo=${1:-./hnsw_demo}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/hnsw-cli.XXXXXX")
trap 'rm -f "$tmp/words" "$tmp/numbers" "$tmp/bad" "$tmp/index" "$tmp/original" "$tmp/out"; rmdir "$tmp"' EXIT

cat > "$tmp/words" <<'EOF'
king 1 0
queen 0.9 0.1
1 0 1
opposite -1 0
EOF
cat > "$tmp/numbers" <<'EOF'
0 0
1 0
3 4
EOF

# Query a labeled index before and after saving. Exclude the query word.
"$demo" --vectors "$tmp/words" --query king --k 1 --save "$tmp/index" > "$tmp/out"
grep -Eq '^[[:space:]]*1\.[[:space:]]+queen[[:space:]]' "$tmp/out"
if grep -Fq 'Recall@' "$tmp/out"; then
    echo "Printed exact search metrics without --verify." >&2
    exit 1
fi
"$demo" --load "$tmp/index" --query king --k 1 --verify > "$tmp/out"
grep -Eq '^[[:space:]]*1\.[[:space:]]+queen[[:space:]]' "$tmp/out"
grep -Fq 'Recall@1: 1.0000' "$tmp/out"
printf '1\nking\n' | "$demo" --load "$tmp/index" --interactive --k 1 > "$tmp/out"
grep -Fq 'Neighbors of 1' "$tmp/out"
grep -Fq 'Neighbors of king' "$tmp/out"

# Query coordinates directly with squared Euclidean distance.
"$demo" --vectors "$tmp/numbers" --numeric --query-vector '3 4' --k 1 > "$tmp/out"
grep -Eq '^[[:space:]]*1\.[[:space:]]+vector 2[[:space:]]+0\.0+$' "$tmp/out"
"$demo" --dim 4 --limit 20 --queries 2 --k 3 > "$tmp/out"
grep -Fq 'Recall@3:' "$tmp/out"

# Reject a bad input row, an unknown word, and an incomplete query vector.
cp "$tmp/index" "$tmp/original"
printf 'good 1 2\nbad 0\n' > "$tmp/bad"
if "$demo" --vectors "$tmp/bad" --save "$tmp/index" > "$tmp/out" 2>&1; then
    echo "Accepted an invalid word-vector row." >&2
    exit 1
fi
cmp "$tmp/index" "$tmp/original"
if "$demo" --load "$tmp/index" --query missing > "$tmp/out" 2>&1; then
    echo "Accepted a missing word." >&2
    exit 1
fi
if "$demo" --load "$tmp/index" --query-vector '0' > "$tmp/out" 2>&1; then
    echo "Accepted a query with too few coordinates." >&2
    exit 1
fi
echo "CLI tests passed."
