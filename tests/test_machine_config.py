import tempfile
import unittest
from pathlib import Path
import yaml
from piano_clamp.data_loading import load_pipeline_config, preflight_dataset, PipelineConfigurationError


class MachineConfigTests(unittest.TestCase):
    def test_profile_and_legacy_overlay_cannot_replace_dataset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / 'machine.yaml').write_text(yaml.safe_dump({'corpus_root': str(root / 'corpus')}))
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump({'machine_profile': 'machine.yaml',
                'adapter_root': 'selected', 'corpus_manifest': 'manifest.csv'}))
            (root / 'paths.yaml').write_text(yaml.safe_dump({'adapter_root': 'wrong',
                'corpus_manifest': 'wrong.csv', 'output_root': 'outputs'}))
            with self.assertWarns(UserWarning):
                resolved = load_pipeline_config(config, root=root, environ={})
            self.assertEqual(resolved['corpus_root'], str(root / 'corpus'))
            self.assertEqual(resolved['corpus_manifest'], str(root / 'selected/manifest.csv'))
            self.assertEqual(resolved['output_root'], str(root / 'outputs'))
            with self.assertWarns(UserWarning):
                override = load_pipeline_config(config, root=root, environ={}, manifest='explicit/input.csv')
            self.assertEqual(override['corpus_manifest'], str(root / 'explicit/input.csv'))

    def test_profile_rejects_dataset_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root / 'machine.yaml').write_text('adapter_root: wrong\n')
            (root / 'config.yaml').write_text('machine_profile: machine.yaml\n')
            with self.assertRaisesRegex(PipelineConfigurationError, 'non-machine'):
                load_pipeline_config(root / 'config.yaml', root=root)

    def test_missing_environment_variable_has_setup_hint(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'config.yaml'
            path.write_text('corpus_root: ${PIANO_CLAMP_CORPUS_ROOT}\n')
            with self.assertRaisesRegex(PipelineConfigurationError, 'export PIANO_CLAMP_CORPUS_ROOT='):
                load_pipeline_config(path, environ={})

    def test_preflight_reports_missing_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            config = {'corpus_root': str(root / 'missing'), 'corpus_manifest': str(root / 'manifest.csv')}
            with self.assertRaisesRegex(PipelineConfigurationError, 'corpus directory'):
                preflight_dataset(config)
            config['corpus_root'] = str(root)
            with self.assertRaisesRegex(PipelineConfigurationError, 'manifest'):
                preflight_dataset(config)
