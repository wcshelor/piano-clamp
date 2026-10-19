import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from piano_clamp.embedding_io import EmbeddingIOError, write_embedding_bundle

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/prepare_composer_passages.py'
spec = importlib.util.spec_from_file_location('prepare_composer_passages', SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReuseTests(unittest.TestCase):
    def bundle(self, root, **changes):
        matrix = np.zeros((2, 768), dtype=np.float32)
        matrix[0, 0] = matrix[1, 1] = 1
        rows = [dict(passage_id=str(i), embedding_row=i, content_hash=str(i) * 64,
                     source_material='score') for i in range(2)]
        metadata = dict(model_space='c2', checkpoint_hash='checkpoint',
                        CLaMP_repository_commit='vendor', source_material='score')
        metadata.update(changes)
        write_embedding_bundle(root, matrix, rows, matrix_filename='passage_embeddings.npy',
            table_filename='passage_embeddings.csv', fields=list(rows[0]),
            id_field='passage_id', metadata=metadata)
        return root

    def test_reuse_preserves_content_vector_mapping(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bundle = module.validate_bundle(self.bundle(root / 'source'),
                {'checkpoint_sha256': 'checkpoint'}, 'vendor')
            cache = root / 'cache'
            self.assertEqual(module.seed_cache(cache, [bundle]), 2)
            self.assertEqual(module.seed_cache(cache, [bundle]), 0)
            np.testing.assert_array_equal(np.load(cache / ('1' * 64 + '.npy')), bundle[0][1])
            conflict = (bundle[0][::-1], bundle[1], bundle[2])
            with self.assertRaisesRegex(EmbeddingIOError, 'Conflicting'):
                module.seed_cache(cache, [conflict])

    def test_incompatible_provenance_rejected(self):
        for changes in ({'checkpoint_hash': 'other'}, {'CLaMP_repository_commit': 'other'},
                        {'model_space': 'saas'}, {'source_material': 'performance_midi'}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temp:
                path = self.bundle(Path(temp), **changes)
                with self.assertRaises(EmbeddingIOError):
                    module.validate_bundle(path, {'checkpoint_sha256': 'checkpoint'}, 'vendor')

    def test_corrupt_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.bundle(Path(temp))
            np.save(path / 'passage_embeddings.npy', np.ones((2, 768)))
            with self.assertRaisesRegex(EmbeddingIOError, 'checksum'):
                module.validate_bundle(path, {'checkpoint_sha256': 'checkpoint'}, 'vendor')

    def test_legacy_symbolic_stage_without_source_material_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            path = self.bundle(Path(temp), source_material=None, stage='passages_symbolic')
            module.validate_bundle(path, {'checkpoint_sha256': 'checkpoint'}, 'vendor')

    def test_missing_or_conflicting_legacy_provenance_is_rejected(self):
        for changes in ({'source_material': None},
                        {'source_material': None, 'stage': 'passages_audio'},
                        {'source_material': 'performance_midi', 'stage': 'passages_symbolic'}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as temp:
                path = self.bundle(Path(temp), **changes)
                with self.assertRaises(EmbeddingIOError):
                    module.validate_bundle(path, {'checkpoint_sha256': 'checkpoint'}, 'vendor')

    def test_legacy_rows_without_hash_are_not_reused(self):
        with tempfile.TemporaryDirectory() as temp:
            matrix = np.ones((1, 768))
            self.assertEqual(module.seed_cache(Path(temp), [(matrix,
                [{'embedding_row': '0', 'content_hash': ''}], {})]), 0)

    def test_workflow_generates_all_composer_windows_and_refreshes(self):
        import argparse
        import json
        from unittest.mock import patch
        import yaml

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / 'checkpoint'
            checkpoint.write_bytes(b'weights')
            config = dict(checkpoint_path=str(checkpoint),
                checkpoint_sha256=module.sha256_path(checkpoint), normalize_embeddings=True,
                vendor_root=str(root), corpus_manifest='manifest')
            inventory = root / 'inventory.json'
            inventory.write_text(json.dumps(dict(artifacts=[], embedding_roots=[], store_roots=[])))
            args = argparse.Namespace(inventory=inventory, output_root=root / 'output',
                config=['config'], device='cpu', reuse_bundle=None, execute=True, chunk_size=25)

            def embed(command, **kwargs):
                derived = yaml.safe_load(Path(command[command.index('--config') + 1]).read_text())
                composer = command[command.index('--composer') + 1]
                matrix = np.zeros((3, 768), dtype=np.float32)
                matrix[:, 0] = 1
                rows = [dict(passage_id=f'{composer}-{size}', composer=composer,
                    bar_start=1, bar_end=size, embedding_row=i,
                    passage_generation_method=f'fixed_{size}_bars_stride_{size // 2}',
                    source_material='score') for i, size in enumerate((4, 8, 16))]
                write_embedding_bundle(Path(derived['output_root']) / 'passages', matrix, rows,
                    matrix_filename='passage_embeddings.npy', table_filename='passage_embeddings.csv',
                    fields=list(rows[0]), id_field='passage_id', metadata=dict(model_space='c2',
                    checkpoint_hash=config['checkpoint_sha256'], CLaMP_repository_commit='vendor',
                    source_material='score'))

            with patch.object(module, 'load_pipeline_config', return_value=config), \
                 patch.object(module, 'git_commit', return_value='vendor'), \
                 patch.object(module, 'read_corpus_manifest', return_value=[{'composer': c} for c in module.COMPOSERS]), \
                 patch.object(module.subprocess, 'run', side_effect=embed), \
                 patch.object(module, 'refresh_inventory') as refresh:
                module.run(args)
                refresh.assert_called_once()
            matrix, rows, metadata = module.read_embedding_bundle(root / 'output/merged')
            self.assertEqual(matrix.shape, (9, 768))
            self.assertEqual(len(metadata['coverage']), 9)
            self.assertEqual([int(row['embedding_row']) for row in rows], list(range(9)))
