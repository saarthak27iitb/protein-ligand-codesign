"""Fetch exact upstream commits and apply recorded local patches; no model downloads."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def run(*args, cwd=None):
    return subprocess.run(args, cwd=cwd, check=True, text=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true', help='Verify existing checkouts without cloning or patching')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    specs = json.loads((root/'provenance/upstreams.json').read_text())
    for name, spec in specs.items():
        patch = root/spec['patch']
        if hashlib.sha256(patch.read_bytes()).hexdigest() != spec['patch_sha256']:
            raise RuntimeError('Patch checksum mismatch: '+name)
        target = root/'vendor'/name
        if not target.exists():
            if args.check_only:
                raise RuntimeError('Missing upstream checkout: '+name)
            target.parent.mkdir(exist_ok=True)
            run('git', 'clone', spec['url'], str(target))
            run('git', 'checkout', '--detach', spec['commit'], cwd=target)
        head = run('git', 'rev-parse', 'HEAD', cwd=target).stdout.strip()
        if head != spec['commit']:
            raise RuntimeError('Unexpected commit; existing checkout left unchanged: '+name)
        applied = subprocess.run(['git', 'apply', '--reverse', '--check', str(patch)], cwd=target, capture_output=True)
        if applied.returncode == 0:
            print(name+': pinned commit and patch present')
            continue
        if args.check_only:
            raise RuntimeError('Expected patch absent or conflicting: '+name)
        if run('git', 'status', '--porcelain', '--untracked-files=no', cwd=target).stdout.strip():
            raise RuntimeError('Existing tracked modifications left unchanged: '+name)
        run('git', 'apply', '--check', str(patch), cwd=target)
        run('git', 'apply', str(patch), cwd=target)
        print(name+': pinned commit and patch installed')


if __name__ == '__main__':
    main()
