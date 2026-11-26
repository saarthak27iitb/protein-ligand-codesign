"""Validate downloaded Zenodo assets against the publisher's checksums."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('manifest', type=Path)
parser.add_argument('name')
parser.add_argument('file', type=Path)
args = parser.parse_args()
entry = next(x for x in json.loads(args.manifest.read_text())['files'] if x['key'] == args.name)
if args.file.stat().st_size != entry['size']:
    raise SystemExit(f'Size mismatch for {args.file}')
algorithm, expected = entry['checksum'].split(':', 1)
h = hashlib.new(algorithm)
with args.file.open('rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        h.update(block)
if h.hexdigest() != expected:
    raise SystemExit(f'Checksum mismatch for {args.file}')
print(json.dumps({'path': str(args.file), 'bytes': entry['size'], 'checksum': entry['checksum'], 'verified': True}))
