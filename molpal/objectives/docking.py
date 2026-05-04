import atexit
import csv
import json
import os
import shutil
import shlex
import subprocess
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem

from molpal.objectives.base import Objective

try:  # pragma: no cover
    import pyscreener as ps
except ModuleNotFoundError:  # pragma: no cover
    ps = None


class DockingObjective(Objective):
    """Docking objective with a built-in Glide backend and optional pyscreener fallback."""

    def __init__(
        self,
        objective_config: str,
        path: str = ".",
        verbose: int = 0,
        minimize: bool = True,
        **kwargs,
    ):
        self.config = parse_docking_config(objective_config)
        self.path = Path(path)
        self.verbose = verbose
        self._cleanup_callbacks = []

        screen_type = self.config["screen_type"]
        if screen_type == "glide":
            self.backend = GlideDockingBackend(self.config, self.path, verbose=verbose)
        else:
            if ps is None:
                raise ModuleNotFoundError(
                    f'pyscreener is required for screen-type="{screen_type}", but it is not installed.'
                )
            self.backend = PyscreenerBackend(objective_config, self.path)
        self.reuse_scores = self._load_reuse_scores(self.config.get("reuse_scores_csv"))
        atexit.register(self.cleanup)

        super().__init__(minimize=minimize)

    def forward(self, smis: Iterable[str], **kwargs) -> Dict[str, Optional[float]]:
        smis = list(smis)
        results: Dict[str, Optional[float]] = {}
        uncached: List[str] = []

        for smi in smis:
            if smi in self.reuse_scores:
                results[smi] = self.reuse_scores[smi]
            else:
                uncached.append(smi)

        if self.verbose > 0 and self.reuse_scores and smis:
            print(
                f"Docking cache hits: {len(smis) - len(uncached)}/{len(smis)} | "
                f"new docking required: {len(uncached)}",
                flush=True,
            )

        if uncached:
            backend_results = self.backend.run(uncached)
            results.update(backend_results)
            for smi, score in backend_results.items():
                if score is not None:
                    self.reuse_scores[smi] = score

        values = []
        for smi in smis:
            score = results.get(smi)
            values.append(None if score is None else self.c * score)
        return dict(zip(smis, values))

    def cleanup(self):
        self.backend.cleanup()

    def _load_reuse_scores(self, path: Optional[str]) -> Dict[str, float]:
        if not path:
            return {}

        cache_path = Path(path)
        if not cache_path.exists():
            raise FileNotFoundError(f"reuse_scores_csv not found: {cache_path}")

        df = pd.read_csv(cache_path)
        smiles_col = None
        score_col = None
        for candidate in ("smiles", "SMILES", "canonical_smiles"):
            if candidate in df.columns:
                smiles_col = candidate
                break
        for candidate in ("glide_score", "best_docking_score", "docking_score", "score"):
            if candidate in df.columns:
                score_col = candidate
                break
        if smiles_col is None or score_col is None:
            raise KeyError(
                f"reuse_scores_csv must contain smiles and score columns. Got columns: {list(df.columns)}"
            )

        cache: Dict[str, float] = {}
        for _, row in df.iterrows():
            smi = row[smiles_col]
            score = row[score_col]
            if not isinstance(smi, str) or not smi:
                continue
            try:
                score = float(score)
            except Exception:
                continue
            prev = cache.get(smi)
            if prev is None or score < prev:
                cache[smi] = score
        return cache


class PyscreenerBackend:
    def __init__(self, objective_config: str, path: Path):
        args = ps.args.gen_args(f"--config {objective_config}")
        metadata_template = ps.build_metadata(args.screen_type, args.metadata_template)
        self.virtual_screen = ps.virtual_screen(
            args.screen_type,
            args.receptors,
            args.center,
            args.size,
            metadata_template,
            args.pdbids,
            args.docked_ligand_file,
            args.buffer,
            args.ncpu,
            args.base_name,
            path,
            args.reduction,
            args.receptor_reduction,
            args.k,
        )

    def run(self, smis: List[str]) -> Dict[str, Optional[float]]:
        scores = self.virtual_screen(smis)
        scores = np.where(np.isnan(scores), None, scores)
        return dict(zip(smis, scores))

    def cleanup(self):
        results = self.virtual_screen.results()
        self.virtual_screen.collect_files()
        if not results:
            return
        with open(self.virtual_screen.path / "extended.csv", "w") as fid:
            writer = csv.writer(fid)
            writer.writerow(results[0].__dataclass_fields__.keys())
            for row in results:
                writer.writerow(getattr(row, field) for field in results[0].__dataclass_fields__)


class GlideDockingBackend:
    SCORE_FIELDS = (
        "r_i_docking_score",
        "r_i_glide_gscore",
        "r_glide_gscore",
    )
    TITLE_FIELDS = ("_Name", "title", "s_m_title")

    def __init__(self, config: Dict[str, object], root: Path, verbose: int = 0):
        self.config = config
        self.root = root
        self.verbose = verbose
        self.batch_index = 0
        self.base_dir = Path(config.get("working_dir") or root / "docking_batches")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.schrodinger = Path(str(config["schrodinger"]))
        self.grid_file = Path(str(config["receptors"][0]))
        self.cleanup_batches = bool(config.get("cleanup", True))
        self.keep_pose_files = bool(config.get("keep_pose_files", False))
        self.result_score_field = str(config.get("result_score_field", "r_i_docking_score"))
        self.ligprep_args = str(config.get("ligprep_args", "-s 1 -epik -ph 7.2 -pht 2.0 -WAIT")).strip()
        self.precision = str(config.get("precision", "SP"))
        self.poses_per_ligand = int(config.get("poses_per_ligand", 5))
        self.ncpu = int(config.get("ncpu", 1))
        self.slurm_enabled = bool(config.get("slurm_enabled", False))
        self.slurm_partition = str(config.get("slurm_partition", "24cQ"))
        self.slurm_cpus_per_task = int(config.get("slurm_cpus_per_task", max(self.ncpu, 1)))
        self.slurm_mem = str(config.get("slurm_mem", "64G"))
        self.slurm_time = str(config.get("slurm_time", "5-00:00:00"))
        self.slurm_module = str(config.get("slurm_module", "schrodinger/2025"))
        self.slurm_job_name = str(config.get("slurm_job_name", "molpal_glide"))
        self.slurm_extra_args = str(config.get("slurm_extra_args", "")).strip()
        self.slurm_retry_attempts = int(config.get("slurm_retry_attempts", 5))
        self.slurm_retry_wait_sec = int(config.get("slurm_retry_wait_sec", 180))

    def run(self, smis: List[str]) -> Dict[str, Optional[float]]:
        batch_dir = self.base_dir / f"batch_{self.batch_index:04d}"
        self.batch_index += 1
        batch_dir.mkdir(parents=True, exist_ok=True)

        ligand_ids = [f"lig_{i:06d}" for i in range(len(smis))]
        id_to_smiles = dict(zip(ligand_ids, smis))

        smiles_path = batch_dir / "batch.smi"
        with smiles_path.open("w", encoding="utf-8") as handle:
            for ligand_id, smi in zip(ligand_ids, smis):
                handle.write(f"{smi} {ligand_id}\n")

        prepped_path = batch_dir / "batch_prepped.maegz"
        ligprep_cmd = (
            f"{shlex.quote(str(self.schrodinger / 'ligprep'))} "
            f"-ismi {shlex.quote(str(smiles_path))} "
            f"-omae {shlex.quote(str(prepped_path))} "
            f"{self.ligprep_args}"
        )

        dock_in = batch_dir / "batch_receptor_dock.in"
        with dock_in.open("w", encoding="utf-8") as handle:
            handle.write("FORCEFIELD   OPLS3\n")
            handle.write(f"GRIDFILE   {self.grid_file}\n")
            handle.write(f"LIGANDFILE   {prepped_path}\n")
            handle.write("POSE_OUTTYPE   ligandlib_sd\n")
            handle.write(f"POSES_PER_LIG   {self.poses_per_ligand}\n")
            handle.write(f"PRECISION   {self.precision}\n")
            if self.ncpu > 1:
                handle.write(f"NCPU   {self.ncpu}\n")

        glide_cmd = f"{shlex.quote(str(self.schrodinger / 'glide'))} {shlex.quote(str(dock_in))} -OVERWRITE -WAIT"
        if self.slurm_enabled:
            self._run_slurm_batch(batch_dir, ligprep_cmd, glide_cmd)
        else:
            self._run_command(ligprep_cmd, batch_dir, "LigPrep failed")
            self._run_command(glide_cmd, batch_dir, "Glide docking failed")

        sdfgz_path = batch_dir / "batch_receptor_dock_lib.sdfgz"
        results = self._parse_results(sdfgz_path, id_to_smiles)
        summary_rows = [{"smiles": smi, "glide_score": results.get(smi)} for smi in smis]
        with (batch_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["smiles", "glide_score"])
            writer.writeheader()
            writer.writerows(summary_rows)

        if self.cleanup_batches:
            self._cleanup_batch(batch_dir)

        return results

    def cleanup(self):
        return

    def _run_command(self, cmd: str, cwd: Path, error_message: str) -> None:
        try:
            subprocess.run(cmd, shell=True, check=True, executable="/bin/bash", cwd=cwd)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"{error_message}: {exc}") from exc

    def _parse_results(self, sdfgz_path: Path, id_to_smiles: Dict[str, str]) -> Dict[str, Optional[float]]:
        results = {smi: None for smi in id_to_smiles.values()}
        if not sdfgz_path.exists():
            return results

        import gzip

        with gzip.open(sdfgz_path, "rb") as handle:
            supplier = Chem.ForwardSDMolSupplier(handle, removeHs=False)
            for mol in supplier:
                if mol is None:
                    continue
                ligand_id = self._extract_ligand_id(mol)
                if ligand_id not in id_to_smiles:
                    continue
                score = self._extract_score(mol)
                smi = id_to_smiles[ligand_id]
                if score is None:
                    continue
                if results[smi] is None or score < results[smi]:
                    results[smi] = score
        return results

    def _extract_ligand_id(self, mol: Chem.Mol) -> Optional[str]:
        for field in self.TITLE_FIELDS:
            if field == "_Name":
                value = mol.GetProp("_Name") if mol.HasProp("_Name") else None
            else:
                value = mol.GetProp(field) if mol.HasProp(field) else None
            if value:
                return value
        return None

    def _extract_score(self, mol: Chem.Mol) -> Optional[float]:
        for field in (self.result_score_field, *self.SCORE_FIELDS):
            if mol.HasProp(field):
                try:
                    return float(mol.GetProp(field))
                except ValueError:
                    continue
        return None

    def _cleanup_batch(self, batch_dir: Path) -> None:
        keep = {"scores.csv", "batch_receptor_dock.in", "batch.smi"}
        if self.keep_pose_files:
            keep.add("batch_receptor_dock_lib.sdfgz")
        for path in batch_dir.iterdir():
            if path.name in keep:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    def _run_slurm_batch(self, batch_dir: Path, ligprep_cmd: str, glide_cmd: str) -> None:
        slurm_script = batch_dir / "run_glide_batch.sbatch"
        out_log = batch_dir / "slurm.out"
        err_log = batch_dir / "slurm.err"
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={self.slurm_job_name}",
            f"#SBATCH --output={out_log}",
            f"#SBATCH --error={err_log}",
            f"#SBATCH --partition={self.slurm_partition}",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={self.slurm_cpus_per_task}",
            f"#SBATCH --mem={self.slurm_mem}",
            f"#SBATCH --time={self.slurm_time}",
        ]
        if self.slurm_extra_args:
            lines.append(self.slurm_extra_args)
        lines.extend(
            [
                "",
                "set -eo pipefail",
                f"cd {shlex.quote(str(batch_dir))}",
                "if command -v module >/dev/null 2>&1; then",
                f"  module load {shlex.quote(self.slurm_module)}",
                "fi",
                f"SCHRODINGER_ROOT={shlex.quote(str(self.schrodinger))}",
                "LICENSE_POLL_SEC=600",
                "",
                "license_available() {",
                "  local feature=\"$1\"",
                "  local output",
                "  output=\"$(${SCHRODINGER_ROOT}/utilities/licutil -available -lic \"${feature}\" -verbose 2>&1 || true)\"",
                "  [[ \"${output}\" != *\"No tokens are currently available\"* ]]",
                "}",
                "",
                "wait_for_license() {",
                "  local feature=\"$1\"",
                "  until license_available \"${feature}\"; do",
                "    echo \"[$(date)] ${feature} unavailable; sleeping ${LICENSE_POLL_SEC}s\"",
                "    sleep \"${LICENSE_POLL_SEC}\"",
                "  done",
                "  echo \"[$(date)] ${feature} appears available\"",
                "}",
                "",
                "while true; do",
                "  wait_for_license LIGPREP_MAIN",
                "  rm -f batch_prepped.maegz batch_prepped.log batch_prepped-dropped.smi \\",
                "        batch_prepped-dropped-indices.txt batch_prepped-dropped-intermediates.maegz",
                f"  if {ligprep_cmd}; then",
                "    break",
                "  fi",
                "  if grep -qi \"LIGPREP_MAIN\\\\|Failed to check out a license\\\\|Licensed number of users already reached\" batch_prepped.log slurm.out 2>/dev/null; then",
                "    echo \"[$(date)] LigPrep failed due to license race; retrying after ${LICENSE_POLL_SEC}s\"",
                "    sleep \"${LICENSE_POLL_SEC}\"",
                "    continue",
                "  fi",
                "  exit 1",
                "done",
                "",
                "while true; do",
                "  wait_for_license GLIDE_MAIN",
                "  rm -f batch_receptor_dock.log batch_receptor_dock.csv batch_receptor_dock_lib.sdfgz \\",
                "        batch_receptor_dock_pv.maegz",
                f"  if {glide_cmd}; then",
                "    break",
                "  fi",
                "  if grep -qi \"GLIDE_MAIN\\\\|Failed to check out a license\\\\|Licensed number of users already reached\" batch_receptor_dock.log slurm.out 2>/dev/null; then",
                "    echo \"[$(date)] Glide failed due to license race; retrying after ${LICENSE_POLL_SEC}s\"",
                "    sleep \"${LICENSE_POLL_SEC}\"",
                "    continue",
                "  fi",
                "  exit 1",
                "done",
                "",
            ]
        )
        slurm_script.write_text("\n".join(lines), encoding="utf-8")
        os.chmod(slurm_script, 0o755)

        submit_cmd = f"sbatch --parsable --wait {shlex.quote(str(slurm_script))}"
        attempt = 0
        while True:
            attempt += 1
            try:
                self._run_command(submit_cmd, batch_dir, "Slurm Glide batch failed")
                return
            except RuntimeError as exc:
                if attempt >= self.slurm_retry_attempts or not self._retryable_license_failure(batch_dir):
                    raise
                if self.verbose > 0:
                    print(
                        f"Glide batch retry {attempt}/{self.slurm_retry_attempts} after license exhaustion. "
                        f"Waiting {self.slurm_retry_wait_sec} seconds before resubmission...",
                        flush=True,
                    )
                time.sleep(self.slurm_retry_wait_sec)

    def _retryable_license_failure(self, batch_dir: Path) -> bool:
        patterns = (
            "insufficient licenses for feature LIGPREP_MAIN",
            "insufficient licenses for feature GLIDE_MAIN",
            "Licensed number of users already reached",
            "Unable to check out 1 LIGPREP_MAIN",
            "Unable to check out 1 GLIDE_MAIN",
            "Failed to check out a license",
        )
        for name in ("batch_prepped.log", "slurm.out", "slurm.err"):
            path = batch_dir / name
            if not path.exists():
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if any(pattern in text for pattern in patterns):
                return True
        return False


def parse_docking_config(config: str) -> Dict[str, object]:
    raw: Dict[str, str] = {}
    with open(config, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            raw[key.replace("-", "_")] = value

    metadata_template = raw.get("metadata_template", '{"software":"glide"}')
    try:
        metadata_template = json.loads(metadata_template)
    except json.JSONDecodeError:
        metadata_template = {"software": metadata_template}

    receptors = raw["receptors"].strip()
    if receptors.startswith("[") and receptors.endswith("]"):
        receptors = [part.strip() for part in receptors[1:-1].split(",") if part.strip()]
    else:
        receptors = [receptors]

    def parse_bool(value: Optional[str], default: bool = False) -> bool:
        if value is None:
            return default
        return value.lower() in {"1", "true", "yes", "y", "on"}

    return {
        "screen_type": raw["screen_type"],
        "receptors": receptors,
        "metadata_template": metadata_template,
        "reuse_scores_csv": raw.get("reuse_scores_csv"),
        "schrodinger": raw.get("schrodinger"),
        "ncpu": int(raw.get("ncpu", 1)),
        "precision": raw.get("precision", "SP"),
        "poses_per_ligand": int(raw.get("poses_per_ligand", 5)),
        "ligprep_args": raw.get("ligprep_args", "-s 1 -epik -ph 7.2 -pht 2.0 -WAIT"),
        "cleanup": parse_bool(raw.get("cleanup"), bool(metadata_template.get("cleanup", False))),
        "keep_pose_files": parse_bool(raw.get("keep_pose_files"), False),
        "result_score_field": raw.get("result_score_field", "r_i_docking_score"),
        "working_dir": raw.get("working_dir"),
        "slurm_enabled": parse_bool(raw.get("slurm_enabled"), False),
        "slurm_partition": raw.get("slurm_partition", "24cQ"),
        "slurm_cpus_per_task": int(raw.get("slurm_cpus_per_task", raw.get("ncpu", 1))),
        "slurm_mem": raw.get("slurm_mem", "64G"),
        "slurm_time": raw.get("slurm_time", "5-00:00:00"),
        "slurm_module": raw.get("slurm_module", "schrodinger/2025"),
        "slurm_job_name": raw.get("slurm_job_name", "molpal_glide"),
        "slurm_extra_args": raw.get("slurm_extra_args", ""),
        "slurm_retry_attempts": int(raw.get("slurm_retry_attempts", 5)),
        "slurm_retry_wait_sec": int(raw.get("slurm_retry_wait_sec", 180)),
    }
