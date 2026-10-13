#!/bin/sh
set -eu
mkdir -p /run/sshd /root/.ssh /workspace/path-b
chmod 700 /root/.ssh
if [ -n "${PUBLIC_KEY:-}" ]; then
    printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi
ssh-keygen -A
exec /usr/sbin/sshd -D -e
