"""Desktop isolation, startup ownership, and setup input regressions."""
import importlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class DesktopPathsTests(unittest.TestCase):
    def test_cpu_bundle_keeps_torch_dependencies_when_replacing_gpu_binaries(self):
        import importlib.util
        script = Path(__file__).resolve().parents[2] / 'scripts' / 'windows' / 'build.py'
        spec = importlib.util.spec_from_file_location('desktop_build_test', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        dependencies = module.dependency_closure()
        self.assertIn('sympy', dependencies)
        self.assertIn('networkx', dependencies)
        self.assertIn('mpmath', dependencies)
        self.assertNotIn('torch', dependencies)

    def test_bundle_takes_the_installed_onnxruntime_build(self):
        import importlib.metadata as metadata
        import importlib.util
        script = Path(__file__).resolve().parents[2] / 'scripts' / 'windows' / 'build.py'
        spec = importlib.util.spec_from_file_location('desktop_build_ort_test', script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        real = metadata.distribution

        def only(installed):
            def distribution(name):
                if name in ('onnxruntime', 'onnxruntime-directml') and name != installed:
                    raise metadata.PackageNotFoundError(name)
                return real(name) if name not in ('onnxruntime', 'onnxruntime-directml') else object()
            return distribution

        for installed in ('onnxruntime-directml', 'onnxruntime'):
            with patch.object(module.metadata, 'distribution', side_effect=only(installed)):
                self.assertEqual(module.onnxruntime_distribution(), installed)

    def test_default_paths_preserve_source_installation(self):
        from core import paths
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(paths.storage_root(), paths.APP_DIR)
            self.assertEqual(paths.omnivoice_model(), paths.APP_DIR.parent / 'model_backup' / 'OmniVoice')

    def test_desktop_storage_is_separate_from_program_files(self):
        from core import paths
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'AURIS_DATA_DIR': folder}):
            self.assertEqual(paths.user_path('data', 'reader.db'), Path(folder) / 'data' / 'reader.db')
            self.assertEqual(paths.omnivoice_model(), Path(folder) / 'models' / 'OmniVoice')


class DesktopLifecycleTests(unittest.TestCase):
    def test_desktop_workers_disable_user_site_packages_and_use_utf8(self):
        from core.desktop_support import python_command
        with patch.dict(os.environ, {'AURIS_DESKTOP': '1'}):
            self.assertEqual(python_command('bundled-python'), ['bundled-python', '-s', '-X', 'utf8'])
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(python_command('source-python'), ['source-python'])

    def test_second_process_cannot_own_the_same_library(self):
        from core.desktop_support import InstanceLock
        with tempfile.TemporaryDirectory() as folder:
            first, second = InstanceLock(Path(folder)), InstanceLock(Path(folder))
            self.assertTrue(first.acquire())
            try:
                self.assertFalse(second.acquire())
            finally:
                first.close()
                second.close()
            third = InstanceLock(Path(folder))
            self.assertTrue(third.acquire())
            third.close()

    def test_worker_runtime_is_selected_before_library_imports(self):
        from core.desktop_support import activate_gpu_runtime
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ, {'AURIS_DATA_DIR': folder}):
            runtime = Path(folder) / 'runtime' / 'gpu'
            runtime.mkdir(parents=True)
            (runtime / 'ready.json').write_text('{}')
            with patch('sys.path', []):
                activate_gpu_runtime()
                import sys
                self.assertEqual(sys.path[0], str(runtime))

    def test_gpu_build_staged_while_loaded_replaces_the_old_one_at_next_start(self):
        from core.desktop_support import install_pending_gpu_runtime
        with tempfile.TemporaryDirectory() as folder:
            runtime = Path(folder) / 'runtime'
            for name, variant in (('gpu', 'cu128'), ('gpu-pending', 'cu126')):
                (runtime / name).mkdir(parents=True)
                (runtime / name / 'ready.json').write_text(f'{{"variant": "{variant}"}}')
            install_pending_gpu_runtime(Path(folder))
            self.assertIn('cu126', (runtime / 'gpu' / 'ready.json').read_text())
            self.assertEqual(sorted(p.name for p in runtime.iterdir()), ['gpu'])


class DesktopSetupTests(unittest.TestCase):
    def setUp(self):
        from core import desktop_setup
        self.setup = desktop_setup
        self.controller = desktop_setup.SetupController(Path(tempfile.gettempdir()) / 'auris-unit-test')

    def test_modern_nvidia_smi_umd_header_is_recognised(self):
        import subprocess
        outputs = [subprocess.CompletedProcess([], 0, 'CUDA UMD Version: 13.4', ''),
                   subprocess.CompletedProcess([], 0, 'NVIDIA GeForce RTX 3090\n', ''),
                   subprocess.CompletedProcess([], 0, '8.6\n', '')]
        with patch('core.desktop_setup.subprocess.run', side_effect=outputs):
            info = self.setup.hardware()
        self.assertTrue(info['gpu_supported'])
        self.assertEqual(info['nvidia'], 'NVIDIA GeForce RTX 3090')
        self.assertEqual(info['variant'], 'cu128')

    def nvidia(self, cuda, name, cap):
        import subprocess
        outputs = [subprocess.CompletedProcess([], 0, f'CUDA Version: {cuda}', ''),
                   subprocess.CompletedProcess([], 0, name + '\n', ''),
                   subprocess.CompletedProcess([], 0, cap + '\n', '')]
        with patch('core.desktop_setup.subprocess.run', side_effect=outputs):
            return self.setup.hardware()

    def test_pascal_card_gets_the_cuda_12_6_build(self):
        # Issue #13: the cu128 wheels have no sm_61 kernels for a GTX 1080 Ti.
        info = self.nvidia('12.8', 'NVIDIA GeForce GTX 1080 Ti', '6.1')
        self.assertEqual((info['variant'], info['min_cuda'], info['gpu_supported']), ('cu126', '12.6', True))
        self.assertTrue(self.nvidia('12.6', 'NVIDIA GeForce GTX 1080 Ti', '6.1')['gpu_supported'])
        self.assertFalse(self.nvidia('12.6', 'NVIDIA GeForce RTX 3090', '8.6')['gpu_supported'])

    def test_card_older_than_maxwell_is_not_offered_gpu_acceleration(self):
        info = self.nvidia('12.8', 'NVIDIA Tesla K80', '3.7')
        self.assertEqual((info['variant'], info['gpu_supported']), ('', False))

    def test_installed_build_for_newer_cards_is_reported_and_reinstalled_beside_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            gpu = root / 'runtime' / 'gpu'
            gpu.mkdir(parents=True)
            (gpu / 'ready.json').write_text('{"variant": "cu128"}')
            controller = self.setup.SetupController(root)
            controller._hardware = {'nvidia': 'GTX 1080 Ti', 'variant': 'cu126', 'min_cuda': '12.6',
                                    'gpu_supported': True}
            self.assertTrue(controller.machine()['gpu_mismatch'])
            commands = []
            with patch('core.desktop_setup.subprocess.run', side_effect=lambda cmd, **kw: commands.append(cmd)):
                controller._install_gpu('cu126')
            self.assertIn('torch==2.11.0+cu126', commands[0])
            self.assertIn('torch.ones(8,device="cuda")', commands[1][-2])
            self.assertIn('cu128', (gpu / 'ready.json').read_text())  # still loaded until restart
            self.assertIn('cu126', (root / 'runtime' / 'gpu-pending' / 'ready.json').read_text())

    def test_accented_process_path_in_nvidia_smi_output_is_tolerated(self):
        # nvidia-smi lists GPU processes in the ANSI code page; the packaged
        # runtime decodes child output as UTF-8 (-X utf8).
        import subprocess
        import sys
        real_run = subprocess.run
        ansi = (b'CUDA Version: 13.0\\n  48284  C+G  C:\\\\Users\\\\J\\xf3zsef\\\\Auris telep\\xedt\\xe9s\\\\python.exe\\n',
                b'NVIDIA GeForce RTX 3090\\n')

        def fake_nvidia_smi(args, **kwargs):
            kwargs.setdefault('encoding', 'utf-8')
            output = ansi[0] if len(args) == 1 else ansi[1]
            return real_run([sys.executable, '-c', f'import sys; sys.stdout.buffer.write(b"{output.decode()}")'],
                            **kwargs)

        with patch('core.desktop_setup.subprocess.run', side_effect=fake_nvidia_smi):
            info = self.setup.hardware()
        self.assertEqual(info['cuda'], '13.0')
        self.assertEqual(info['nvidia'], 'NVIDIA GeForce RTX 3090')

    def test_unknown_engine_is_rejected_before_network_work(self):
        with self.assertRaises(ValueError):
            self.controller.start('arbitrary-engine', False)

    def test_gpu_is_rejected_for_the_cpu_engine(self):
        with self.assertRaises(ValueError):
            self.controller.start('supertonic', True)

    def test_finished_setup_does_not_replace_existing_user_preferences(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / 'data' / 'settings.json'
            target.parent.mkdir()
            target.write_text('{"theme":"paper","desktop_setup_complete":true,"tts_engine":"omnivoice"}')
            self.assertTrue(self.setup.setup_complete(Path(folder)))
            self.assertEqual(target.read_text(), '{"theme":"paper","desktop_setup_complete":true,"tts_engine":"omnivoice"}')


if __name__ == '__main__':
    unittest.main()
