"""Проверки изоляции нового запуска и сохранности зафиксированного эксперимента."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / 'scripts'))
from reproduce import digest, layout, prepare_workspace


class WorkspaceSafetyChecks(unittest.TestCase):
    def test_replay_does_not_inherit_saved_models_or_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = prepare_workspace(Path(directory), attach_raw=False)
            self.assertEqual(list((workspace / 'output').iterdir()), [])
            self.assertFalse((workspace / 'output/research_v2/holdout.npz').exists())
            for logical, record in layout()['sources'].items():
                self.assertEqual(digest(workspace / logical), record['sha256'])

    def test_numerical_source_change_is_rejected_but_generated_report_is_kept(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = prepare_workspace(Path(directory), attach_raw=False)
            report = workspace / 'research_v2/README.md'
            report.write_text('Новый отчёт изолированного запуска')
            prepare_workspace(workspace, attach_raw=False)
            self.assertEqual(report.read_text(), 'Новый отчёт изолированного запуска')
            source = workspace / 'research_v2/core.py'
            source.write_text(source.read_text() + '\n# changed\n')
            with self.assertRaisesRegex(ValueError, 'Изменён исходник'):
                prepare_workspace(workspace, attach_raw=False)

    def test_workspace_cannot_overwrite_inputs_or_fixed_results(self):
        for target in [ROOT, ROOT/'src', ROOT/'data/processed/replay',
                       ROOT/'results/research_v2/new_run', ROOT/'configs/replay']:
            with self.subTest(target=target), self.assertRaises(ValueError):
                prepare_workspace(target, attach_raw=False)
