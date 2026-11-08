"""Standard-library checks; no model, tensor library or training module is imported."""
import ast
import csv
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / 'provenance/restructure_manifest.json').read_text())


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


class RepositoryTests(unittest.TestCase):
    def test_all_paths_are_ascii(self):
        for path in ROOT.rglob('*'):
            if '__pycache__' not in path.parts:
                self.assertTrue(str(path.relative_to(ROOT)).isascii(), str(path))

    def test_python_syntax(self):
        for path in ROOT.rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))

    def test_scientific_definitions_preserved(self):
        for item in MANIFEST['definitions']:
            with self.subTest(file=item['file'], name=item['name']):
                tree = ast.parse((ROOT / item['file']).read_text())
                node = next(n for n in tree.body if getattr(n, 'name', None) == item['name'])
                value = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
                self.assertEqual(value, item['original_ast_sha256'])

    def test_scientific_cli_defaults_preserved(self):
        for item in MANIFEST['cli_defaults']:
            tree = ast.parse((ROOT / item['file']).read_text())
            found = {}
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == 'add_argument' and node.args
                        and isinstance(node.args[0], ast.Constant)):
                    continue
                for keyword in node.keywords:
                    if keyword.arg == 'default':
                        found[node.args[0].value] = ast.dump(keyword.value, include_attributes=False)
            for key, value in item['defaults'].items():
                self.assertEqual(found[key], value, (item['file'], key))

    def test_local_import_targets_exist(self):
        names = {'models', 'data', 'train', 'utils', 'configs'}
        for path in ROOT.rglob('*.py'):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.split('.')[0] in names:
                    module = ROOT.joinpath(*node.module.split('.'))
                    self.assertTrue(module.with_suffix('.py').exists() or (module / '__init__.py').exists(),
                                    (str(path), node.module))

    def test_csv_bytes_preserved(self):
        for item in MANIFEST['datasets']:
            self.assertEqual(hashlib.sha256((ROOT / item['file']).read_bytes()).hexdigest(), item['sha256'])

    def test_data_counts_and_test_exclusion(self):
        self.assertEqual([len(read_csv(ROOT / 'data/theory' / name))
                          for name in ['train.csv', 'val.csv', 'test.csv']], [1574, 190, 196])
        self.assertEqual(len(read_csv(ROOT / 'data/experimental/experimental_dataset.csv')), 872)
        development = read_csv(ROOT / 'data/public_tg/training_data.csv')
        test = read_csv(ROOT / 'data/public_tg/test_data.csv')
        self.assertEqual((len(development), len(test)), (683, 209))
        exclusion = json.loads((ROOT / 'provenance/public_tg/test_input_exclusion.json').read_text())
        self.assertEqual(len(exclusion['excluded_original_test_positions']), 10)
        self.assertEqual([r['ID'] for r in test], exclusion['retained_test_IDs'])
        self.assertTrue(set(exclusion['excluded_test_IDs']).isdisjoint(r['ID'] for r in test))

    def test_public_source_integrity_manifest(self):
        manifest = json.loads((ROOT / 'provenance/public_tg/package_manifest.json').read_text())
        for name, expected in manifest['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)

    def test_frozen_folds_preserved(self):
        folds = json.loads((ROOT / 'provenance/public_tg/folds.json').read_text())
        validation = []
        self.assertEqual(len(folds), 10)
        for split in folds.values():
            train, val = set(split['train']), set(split['validation'])
            self.assertFalse(train & val)
            self.assertEqual(train | val, set(range(683)))
            validation.extend(split['validation'])
        self.assertEqual(sorted(validation), list(range(683)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
