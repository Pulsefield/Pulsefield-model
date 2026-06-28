import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from pulsefield_model.training.mapper_v2_1_control_cache_overnight import parse_args as parse_cache_args
from pulsefield_model.training.mapper_v2_1_control_cache_overnight import run_supervisor as run_cache_supervisor
from pulsefield_model.training.mapper_v2_1_overnight import parse_args as parse_mapper_args
from pulsefield_model.training.mapper_v2_1_overnight import run_supervisor as run_mapper_supervisor
from pulsefield_model.training.overnight import (
    checkpoint_saved_after,
    existing_resume_checkpoint,
    hydra_override,
    next_saved_step_target,
    read_progress,
)


class OvernightWrapperTests(unittest.TestCase):
    def test_next_saved_step_target_uses_next_checkpoint_boundary(self) -> None:
        self.assertEqual(
            next_saved_step_target(completed_steps=0, max_steps=12000, save_every=250, steps_per_process=250),
            250,
        )
        self.assertEqual(
            next_saved_step_target(completed_steps=1, max_steps=12000, save_every=250, steps_per_process=250),
            500,
        )
        self.assertEqual(
            next_saved_step_target(completed_steps=11800, max_steps=12000, save_every=250, steps_per_process=250),
            12000,
        )

    def test_hydra_override_quotes_string_values(self) -> None:
        self.assertEqual(
            hydra_override("output.output_dir", Path("/tmp/pulsefield,out=a")),
            'output.output_dir="/tmp/pulsefield,out=a"',
        )
        self.assertEqual(hydra_override("output.resume_from", None), "output.resume_from=null")
        self.assertEqual(hydra_override("run.max_steps", 20), "run.max_steps=20")

    def test_existing_resume_checkpoint_prefers_latest_output_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            output_dir.mkdir()
            latest = output_dir / "checkpoint.pt"
            latest.write_bytes(b"latest")
            configured = Path(temp_dir) / "configured.pt"
            configured.write_bytes(b"configured")

            result = existing_resume_checkpoint({"resume_from": configured.as_posix()}, output_dir)

        self.assertEqual(result, latest)

    def test_read_progress_treats_completed_max_steps_as_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            report = Path(temp_dir) / "report.json"
            report.write_text(json.dumps({"completed_steps": 10, "is_complete": False}), encoding="utf-8")

            progress = read_progress(report, max_steps=10)

        self.assertEqual(progress.completed_steps, 10)
        self.assertTrue(progress.is_complete)

    def test_checkpoint_saved_after_accepts_archive_for_reported_target_step(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            checkpoint = output_dir / "checkpoint.pt"
            checkpoint.write_bytes(b"latest")
            archive = output_dir / "checkpoints" / "checkpoint_step_000250.pt"
            archive.parent.mkdir()
            archive.write_bytes(b"archive")

            saved = checkpoint_saved_after(
                output_dir=output_dir,
                checkpoint_path=checkpoint,
                completed_steps=250,
                target_step=250,
                previous_mtime_ns=checkpoint.stat().st_mtime_ns,
            )

        self.assertTrue(saved)

    def test_mapper_overnight_dry_run_launches_v2_1_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            args, trainer_args = parse_mapper_args(
                [
                    "--output-dir",
                    output_dir.as_posix(),
                    "--max-steps",
                    "20",
                    "--save-every",
                    "10",
                    "--dry-run",
                    "--uv-command",
                    "python -m pulsefield_model.training.mapper_training_hydra",
                    "output.resume_from=null",
                ]
            )
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = run_mapper_supervisor(args, trainer_args)

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("overnight_dry_run python -m pulsefield_model.training.mapper_training_hydra", output)
        self.assertIn("training/mapper=v2_1_sparse_d384_l4_phase_b", output)
        self.assertIn(f'output.output_dir="{output_dir.as_posix()}"', output)
        self.assertIn("run.max_steps=20", output)
        self.assertIn("run.save_every=10", output)
        self.assertIn("output.resume_from=null", output)

    def test_control_cache_overnight_dry_run_forces_cache_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "run"
            args, trainer_args = parse_cache_args(
                [
                    "--output-dir",
                    output_dir.as_posix(),
                    "--dry-run",
                    "--uv-command",
                    "python -m pulsefield_model.training.mapper_training_hydra",
                ]
            )
            self.assertEqual(args.max_runs, 16000)
            self.assertEqual(args.control_teacher_precompute_batch_size, 24)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = run_cache_supervisor(args, trainer_args)
        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn("overnight_cache_dry_run python -m pulsefield_model.training.mapper_training_hydra", output)
        self.assertIn("training/mapper=v2_1_sparse_d384_l4_phase_b", output)
        self.assertIn(f'output.output_dir="{output_dir.as_posix()}"', output)
        self.assertIn("data.precompute_control_teacher_cache_only=true", output)
        self.assertIn("data.control_teacher_precompute_batch_size=24", output)
        self.assertNotIn("--config", output)

    def test_overnight_wrappers_reject_legacy_trainer_config_arg_before_hydra_parse(self) -> None:
        for parse_args in (parse_mapper_args, parse_cache_args):
            with self.subTest(parse_args=parse_args.__module__):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    parse_args(["--dry-run", "--config", "configs/training/legacy.yaml"])

                self.assertEqual(raised.exception.code, 2)
                error = stderr.getvalue()
                self.assertIn("Deprecated legacy trainer argument '--config'", error)
                self.assertIn("--mapper-preset <preset>", error)
                self.assertIn("run.max_steps=20", error)
                self.assertNotIn("OverrideParseException", error)


if __name__ == "__main__":
    unittest.main()
