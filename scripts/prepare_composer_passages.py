#!/usr/bin/env python3
"""Plan or build compatible Bach/Chopin/Mozart 4/8/16-bar score bundles.

Run in the HPC embedding environment. Defaults to a read-only plan; --execute
uses the existing resumable passage pipeline, writes isolated outputs, and
refreshes the inventory even if generation fails. Source bundles are read-only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from piano_clamp.data_loading import load_pipeline_config, preflight_dataset
from piano_clamp.embedding_io import (
    EmbeddingIOError, read_embedding_bundle, sha256_path, write_embedding_bundle,
)
from piano_clamp.metadata import git_commit

COMPOSERS = ('Johann Sebastian Bach', 'Frédéric Chopin', 'Wolfgang Amadeus Mozart')


def validate_bundle(path, config, vendor_commit):
    matrix, rows, meta = read_embedding_bundle(path)
    expected = {
        'model_space': 'c2', 'id_field': 'passage_id',
        'checkpoint_hash': config['checkpoint_sha256'],
        'CLaMP_repository_commit': vendor_commit,
        'normalization': 'l2',
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise EmbeddingIOError(f'{path}: incompatible or missing {key}: {meta.get(key)!r}; expected {value!r}')
    # Legacy score-passage runs predate the source_material metadata field.
    # Only their explicit symbolic passage stage establishes the fallback;
    # never override an explicit conflicting source material.
    legacy_score = (meta.get('source_material') in {None, ''}
                    and meta.get('stage') == 'passages_symbolic')
    if meta.get('source_material') not in {'score', 'symbolic'} and not legacy_score:
        raise EmbeddingIOError(f'{path}: not a score-symbolic bundle')
    if matrix.ndim != 2 or matrix.shape[1] != 768 or not np.isfinite(matrix).all():
        raise EmbeddingIOError(f'{path}: invalid matrix')
    if meta.get('embedding_sha256') != sha256_path(Path(path) / meta['matrix_file']):
        raise EmbeddingIOError(f'{path}: matrix checksum mismatch')
    successful = [row for row in rows if row.get('embedding_row', '') != '']
    indices = [int(row['embedding_row']) for row in successful]
    if sorted(indices) != list(range(len(matrix))):
        raise EmbeddingIOError(f'{path}: invalid table/matrix alignment')
    if len(matrix) and not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
        raise EmbeddingIOError(f'{path}: vectors are not L2 normalized')
    for row in successful:
        if row.get('source_material') not in {None, '', 'score', 'symbolic'}:
            raise EmbeddingIOError(f'{path}: mixed source materials')
        if row.get('embedding_modality') not in {None, '', 'score', 'symbolic'}:
            raise EmbeddingIOError(f'{path}: incompatible row embedding modality')
    return matrix, successful, meta


def seed_cache(destination, bundles):
    """Reuse by content, never just by passage ID or bar coordinates."""
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for matrix, rows, _ in bundles:
        for row in rows:
            digest = row.get('content_hash', '')
            if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                continue  # Legacy rows without a digest must be regenerated.
            vector = matrix[int(row['embedding_row'])]
            path = destination / f'{digest}.npy'
            if path.exists():
                if not np.allclose(np.load(path, allow_pickle=False), vector, atol=1e-5):
                    raise EmbeddingIOError(f'Conflicting cached vectors for content {digest}')
            else:
                np.save(path, vector, allow_pickle=False)
                count += 1
    return count


def refresh_inventory(inventory, output):
    command = [sys.executable, str(ROOT / 'scripts/inventory_hpc_embeddings.py')]
    for root in dict.fromkeys([*inventory['embedding_roots'], str(output)]):
        command += ['--embedding-root', root]
    for root in dict.fromkeys([*inventory['store_roots'], str(output / 'embedding_store')]):
        command += ['--store-root', root]
    command += ['--output-dir', str(ROOT / 'reports/hpc/artifact_inventory')]
    subprocess.run(command, check=True, cwd=ROOT)


def run(args):
    inventory = json.loads(args.inventory.read_text())
    output = args.output_root.resolve()
    manifests = getattr(args, 'manifest', None)
    if manifests and len(manifests) != len(args.config):
        raise ValueError('Repeat --manifest once per --config, in the same order')
    configs = [load_pipeline_config(path, device=args.device, **(
        {'manifest': manifests[i]} if manifests else {})) for i, path in enumerate(args.config)]
    datasets = [preflight_dataset(config) for config in configs]
    for config, rows in zip(configs, datasets):
        print(json.dumps({'config': config.get('configuration_path'),
            'corpus_root': config['corpus_root'], 'manifest': config['corpus_manifest'],
            'composers': dict(Counter(row['composer'] for row in rows))}, indent=2), flush=True)
    vendor_commit = git_commit(configs[0]['vendor_root'])
    if vendor_commit == 'unavailable':
        raise ValueError('Cannot verify the CLaMP vendor commit')
    for config in configs:
        if (not config.get('normalize_embeddings') or
                git_commit(config['vendor_root']) != vendor_commit or
                sha256_path(config['checkpoint_path']) != config['checkpoint_sha256'] or
                config['checkpoint_sha256'] != configs[0]['checkpoint_sha256']):
            raise ValueError('Configs must use the same verified C2 checkpoint, vendor commit and L2 normalization')
    paths = args.reuse_bundle or sorted({Path(a['path']) for a in inventory['artifacts']
        if a['kind'] == 'passages_symbolic' and int(a['row_count']) > 0})
    bundles = []
    for path in paths:
        # Missing inventory paths are reported, never silently treated as reusable.
        if not path.exists():
            print(f'Unavailable inventory bundle: {path}', flush=True)
            continue
        bundles.append(validate_bundle(path, configs[0], vendor_commit))
    if args.reuse_bundle and len(bundles) != len(paths):
        raise ValueError('An explicitly requested reuse bundle is unavailable')
    jobs = []
    owners = {}
    for index, config in enumerate(configs):
        available = {row['composer'] for row in datasets[index]}
        for composer in COMPOSERS:
            if composer not in available:
                continue
            if composer in owners:
                raise ValueError(f'{composer} occurs in multiple configs; provide non-overlapping manifests')
            owners[composer] = index
            slug = ('bach', 'chopin', 'mozart')[COMPOSERS.index(composer)]
            root = output / slug
            if root.exists():
                raise ValueError(f'Refusing existing output directory: {root}; choose a new --output-root')
            jobs.append((config, composer, root))
    missing = set(COMPOSERS) - owners.keys()
    if missing:
        raise ValueError(f'Manifests do not cover composers: {sorted(missing)}')
    if (output / 'merged').exists():
        raise ValueError('Merged output already exists; choose a new --output-root')
    print(json.dumps({'jobs': [{'composer': c, 'output': str(r), 'windows': [4, 8, 16],
        'strides': [2, 4, 8]} for _, c, r in jobs], 'compatible_reuse_bundles': len(bundles),
        'execute': args.execute}, indent=2), flush=True)
    if not args.execute:
        return
    try:
        generated = []
        for config, composer, root in jobs:
            root.mkdir(parents=True)
            derived = dict(config)
            # The manifest and alignment paths remain rooted in this adapter,
            # which is outside the read-only corpus. Preserve that boundary
            # when the child process reloads the resolved configuration.
            derived.update(use_local_paths_overlay=False,
                output_root=str(root), embedding_store_root=str(output / 'embedding_store'),
                temporary_root=str(root / 'temp'), log_root=str(root / 'logs'))
            config_path = root / 'config.yaml'
            config_path.write_text(yaml.safe_dump(derived))
            seed_cache(root / 'passages/.resume/vectors', bundles)
            command = [sys.executable, str(ROOT / 'scripts/embed_passages.py'),
                '--config', str(config_path), '--device', args.device,
                '--source-material', 'score', '--composer', composer, '--overlap-half',
                '--mode', '4', '--mode', '8', '--mode', '16', '--chunk-size', str(args.chunk_size)]
            subprocess.run(command, check=True, cwd=ROOT)
            generated.append(validate_bundle(root / 'passages', configs[0], vendor_commit))
        vectors, rows, ids = [], [], set()
        coverage = Counter()
        for matrix, source_rows, _ in generated:
            for source in source_rows:
                row = dict(source)
                length = int(row['bar_end']) - int(row['bar_start']) + 1
                if length not in {4, 8, 16} or not row['passage_generation_method'].startswith('fixed_'):
                    raise ValueError('Unexpected passage window in generated output')
                if row['passage_id'] in ids:
                    raise ValueError(f"Duplicate passage ID: {row['passage_id']}")
                ids.add(row['passage_id'])
                vectors.append(matrix[int(row['embedding_row'])])
                row['embedding_row'] = len(rows)
                rows.append(row)
                coverage[f"{row['composer']} / {length}"] += 1
        for composer in COMPOSERS:
            for length in (4, 8, 16):
                if not coverage[f'{composer} / {length}']:
                    raise ValueError(f'No successful {length}-bar passages for {composer}')
        meta = {key: generated[0][2][key] for key in
            ('model_space', 'checkpoint_hash', 'CLaMP_repository_commit')}
        meta.update(source_material='score', window_bars=[4, 8, 16],
            fixed_window_overlap_half=True, coverage=dict(coverage),
            merged_sources=[str(root / 'passages') for _, _, root in jobs],
            reused_sources=[str(p) for p in paths if p.exists()])
        write_embedding_bundle(output / 'merged', np.stack(vectors), rows,
            matrix_filename='passage_embeddings.npy', table_filename='passage_embeddings.csv',
            fields=list(dict.fromkeys(k for row in rows for k in row)),
            id_field='passage_id', metadata=meta)
        print(json.dumps({'merged': str(output / 'merged'), 'coverage': dict(coverage)}, indent=2))
    finally:
        refresh_inventory(inventory, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', action='append', required=True,
        help='Repeat for non-overlapping composer manifests (e.g. Bach and Chopin/Mozart configs)')
    parser.add_argument('--manifest', action='append', help='Explicit manifest override; repeat once per config in matching order')
    parser.add_argument('--inventory', type=Path, default=ROOT / 'reports/hpc/artifact_inventory/latest.json')
    parser.add_argument('--reuse-bundle', type=Path, action='append', help='Default: all available passage bundles in inventory')
    parser.add_argument('--output-root', type=Path, required=True, help='New isolated output root')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='cuda')
    parser.add_argument('--chunk-size', type=int, default=25)
    parser.add_argument('--execute', action='store_true', help='Generate embeddings; otherwise only validate and plan')
    args = parser.parse_args()
    if args.chunk_size < 1:
        parser.error('--chunk-size must be positive')
    try:
        run(args)
    except (ValueError, OSError, EmbeddingIOError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f'error: {exc}\n')


if __name__ == '__main__':
    main()
