#!/usr/bin/env sh
# Creates random secrets for the Compose stack in deploy/secrets/ (git-ignored). Existing files are kept.
set -eu
dir="$(cd "$(dirname "$0")" && pwd)/secrets"
mkdir -p "$dir"
chmod 700 "$dir"
for name in axon_api_key axon_metrics_key grafana_admin_password; do
  if [ ! -s "$dir/$name" ]; then
    umask 077
    python3 -c "import secrets; print(secrets.token_urlsafe(32))" > "$dir/$name"
    echo "created $dir/$name"
  fi
done
# containers run as their own non-root users and read the mounted files directly
chmod 644 "$dir"/*          # the directory stays 700, so other host users still cannot reach them
echo "client key:    $dir/axon_api_key"
echo "grafana admin: $dir/grafana_admin_password"
