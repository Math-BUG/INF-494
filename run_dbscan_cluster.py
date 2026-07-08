#!/usr/bin/env python3
"""Run the CUDA DBSCAN benchmark on a SLURM/GPU cluster.

This runner intentionally does not require Jupyter. By default it compiles the
CUDA source stored next to this script and imports the synthetic dataset helpers
from the same folder. Notebook extraction is kept as a fallback for
compatibility.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.neighbors import NearestNeighbors


DEFAULT_DATASETS = [
    "heterogeneous_blobs_2d",
    "heterogeneous_blobs_4d",
    "heterogeneous_blobs_12d",
    "heterogeneous_blobs_14d",
    "heterogeneous_blobs_16d",
    "heterogeneous_blobs_20d",
    "heterogeneous_blobs_32d",
    "dense_blobs_2d",
    "dense_blobs_12d",
    "dense_blobs_14d",
    "dense_blobs_16d", 
    "dense_blobs_32d", 
    "spiral_2d", 
    "spiral_12d", 
    "spiral_14d", 
    "spiral_16d",
    "moons_10d", 
    "moons_12d",
    "moons_14d", 
    "moons_16d", 
    "rings_10d", 
    "rings_12d", 
    "rings_14d",
    "rings_16d",
]

OUTPUT_COLUMNS = [
    "dataset",
    "D",
    "N",
    "repeticao",
    "n_eps",
    "n_minpts",
    "n_config_grid",
    "tempo_multi_eps_ms",
    "tempo_multi_minpts_ms",
    "tempo_multi_both_ms",
    "tempo_cuml_1_chamada_ms",
    "tempo_cuml_multi_eps_3_chamadas_ms",
    "tempo_cuml_multi_minpts_4_chamadas_ms",
    "tempo_cuml_multi_both_12_chamadas_ms",
    "speedup_multi_eps_vs_cuml_3_eps",
    "speedup_multi_minpts_vs_cuml_4_minpts",
    "speedup_multi_both_vs_cuml_12_combinacoes",
    "erro",
]

OUTPUT_RENAME_MAP = {
    "multi_eps_ms": "tempo_multi_eps_ms",
    "multi_minpts_ms": "tempo_multi_minpts_ms",
    "multi_both_ms": "tempo_multi_both_ms",
    "cuml_fit_ms": "tempo_cuml_1_chamada_ms",
    "cuml_1_config_ref_ms": "tempo_cuml_1_chamada_ms",
    "cuml_multi_eps_ms": "tempo_cuml_multi_eps_3_chamadas_ms",
    "cuml_multi_minpts_ms": "tempo_cuml_multi_minpts_4_chamadas_ms",
    "cuml_multi_both_ms": "tempo_cuml_multi_both_12_chamadas_ms",
    "speedup_cuml_vs_multi_eps": "speedup_multi_eps_vs_cuml_3_eps",
    "speedup_cuml_vs_multi_minpts": "speedup_multi_minpts_vs_cuml_4_minpts",
    "speedup_cuml_vs_multi_both": "speedup_multi_both_vs_cuml_12_combinacoes",
}


def shard_suffix(shard_index: int, num_shards: int) -> str:
    return f"_shard{shard_index:02d}_of_{num_shards:02d}"


def validate_shard_args(shard_index: int, num_shards: int) -> None:
    if num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")


def select_shard(combinacoes: list[tuple[str, int, int]], shard_index: int, num_shards: int) -> list[tuple[str, int, int]]:
    validate_shard_args(shard_index, num_shards)
    if num_shards == 1:
        return combinacoes
    return [combo for idx, combo in enumerate(combinacoes) if idx % num_shards == shard_index]


def parse_csv_list(value: str, cast=str) -> list[Any]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return [cast(item) for item in items]


def load_notebook(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def cell_sources(notebook: dict[str, Any]) -> list[str]:
    sources = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        sources.append(source)
    return sources


def extract_cuda_source(notebook: dict[str, Any]) -> str:
    for source in cell_sources(notebook):
        if source.lstrip().startswith("%%writefile") and "dbscan_multi_baixo_nivel.cu" in source.splitlines()[0]:
            return "\n".join(source.splitlines()[1:]) + "\n"
    raise RuntimeError("Could not find the %%writefile cell for dbscan_multi_baixo_nivel.cu")


def extract_dataset_helpers(notebook: dict[str, Any]) -> str:
    for source in cell_sources(notebook):
        if "def make_synthetic_dataset" in source and "def sugerir_parametros_dbscan" in source:
            return source
    raise RuntimeError("Could not find dataset/parameter helper functions in the notebook")


def write_cuda_source(notebook: dict[str, Any], cuda_src: Path) -> None:
    cuda_src.write_text(extract_cuda_source(notebook), encoding="utf-8")


def copy_cuda_source(cuda_source_file: Path, cuda_src: Path) -> None:
    cuda_src.write_text(cuda_source_file.read_text(encoding="utf-8"), encoding="utf-8")


def compile_cuda(cuda_src: Path, cuda_bin: Path, arch: str) -> None:
    cmd = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "-gencode",
        f"arch=compute_{arch},code=sm_{arch}",
        "-gencode",
        f"arch=compute_{arch},code=compute_{arch}",
        str(cuda_src),
        "-o",
        str(cuda_bin),
    ]
    print("Compiling CUDA:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def parse_stdout_cuda(stdout_text: str) -> dict[str, Any]:
    resultado: dict[str, Any] = {}
    for parte in stdout_text.strip().split():
        if "=" not in parte:
            continue
        chave, valor = parte.split("=", 1)
        try:
            resultado[chave] = float(valor) if "." in valor else int(valor)
        except ValueError:
            resultado[chave] = valor
    return resultado


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def save_nvprof_output(
    nvprof_dir: Path | None,
    mode: str,
    tag: str,
    argv: list[str],
    resultado: subprocess.CompletedProcess[str],
) -> None:
    if nvprof_dir is None:
        return
    nvprof_dir.mkdir(parents=True, exist_ok=True)
    output_path = nvprof_dir / f"{safe_filename(tag)}_{safe_filename(mode)}.txt"
    output_path.write_text(
        "\n".join(
            [
                "command:",
                " ".join(argv),
                "",
                f"returncode: {resultado.returncode}",
                "",
                "stdout:",
                resultado.stdout or "",
                "",
                "stderr:",
                resultado.stderr or "",
            ]
        ),
        encoding="utf-8",
    )


def load_cuml(skip_cuml: bool) -> tuple[bool, Any, Any]:
    if skip_cuml:
        return False, None, None
    try:
        import cupy
        from cuml.cluster import DBSCAN as cuDBSCAN

        print("cuML available.", flush=True)
        print("CuPy:", getattr(cupy, "__version__", "unknown"), flush=True)
        return True, cuDBSCAN, cupy
    except Exception as exc:
        print("cuML unavailable; continuing without cuML comparison.", flush=True)
        print(f"Import error: {exc}", flush=True)
        return False, None, None


def run_kernel_cuda(
    mode: str,
    x_matrix: np.ndarray,
    extra_args: list[Any],
    tag: str,
    data_dir: Path,
    cuda_bin: Path,
    timeout_s: int,
    profile_nvprof: bool,
    print_nvprof: bool,
    warmup_cuda: bool,
    nvprof_dir: Path | None,
) -> dict[str, Any]:
    n, d = x_matrix.shape
    input_path = data_dir / f"X_{tag}.bin"
    labels_path = data_dir / f"labels_{tag}.bin"
    core_path = data_dir / f"core_{tag}.bin"

    np.ascontiguousarray(x_matrix, dtype=np.float32).tofile(input_path)

    base_argv = [
        str(cuda_bin),
        mode,
        str(input_path),
        str(labels_path),
        str(core_path),
        str(n),
        str(d),
    ]
    base_argv += [str(a) for a in extra_args]

    if profile_nvprof:
        argv = ["nvprof", "--profile-from-start", "off"] + base_argv
    else:
        argv = base_argv

    env = os.environ.copy()
    env["DBSCAN_CUDA_WARMUP"] = "1" if warmup_cuda else "0"
    try:
        resultado = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s, env=env)
        if profile_nvprof:
            save_nvprof_output(nvprof_dir, mode, tag, argv, resultado)

        if profile_nvprof and print_nvprof and resultado.stderr:
            print(resultado.stderr, flush=True)

        if resultado.returncode != 0:
            detalhe = resultado.stderr.strip() or resultado.stdout.strip()
            raise RuntimeError(f"Kernel CUDA failed (mode={mode}, tag={tag}): {detalhe}")

        parsed = parse_stdout_cuda(resultado.stdout)
        if "event_total_ms" not in parsed:
            raise RuntimeError(f"Could not parse event_total_ms from CUDA output: {resultado.stdout}")
        return parsed
    finally:
        for temp_path in (input_path, labels_path, core_path):
            try:
                temp_path.unlink(missing_ok=True)
            except OSError as exc:
                print(f"Warning: could not remove temp file {temp_path}: {exc}", flush=True)


def run_cuml(cu_dbscan: Any, cupy_module: Any, x_matrix_gpu: Any, eps: float, min_samples: int) -> float:
    model_warmup = cu_dbscan(eps=float(eps), min_samples=int(min_samples))
    model_warmup.fit(x_matrix_gpu)
    cupy_module.cuda.runtime.deviceSynchronize()
    del model_warmup

    model = cu_dbscan(eps=float(eps), min_samples=int(min_samples))
    start_event = cupy_module.cuda.Event()
    end_event = cupy_module.cuda.Event()
    start_event.record()
    model.fit(x_matrix_gpu)
    end_event.record()
    end_event.synchronize()
    return float(cupy_module.cuda.get_elapsed_time(start_event, end_event))


def install_notebook_helpers(notebook: dict[str, Any], namespace: dict[str, Any]) -> None:
    source = extract_dataset_helpers(notebook)
    exec(compile(source, "<notebook_dataset_helpers>", "exec"), namespace)


def install_dataset_helpers(script_dir: Path, notebook: dict[str, Any] | None, namespace: dict[str, Any]) -> None:
    helper_file = script_dir / "datasets_sinteticos.py"
    if helper_file.exists():
        spec = importlib.util.spec_from_file_location("datasets_sinteticos", helper_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load dataset helper module: {helper_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace["make_synthetic_dataset"] = module.make_synthetic_dataset
        namespace["load_dataset_from_config"] = module.load_dataset_from_config
        namespace["sugerir_parametros_dbscan"] = module.sugerir_parametros_dbscan
        print("dataset_helpers:", helper_file, flush=True)
        return

    if notebook is None:
        raise RuntimeError("Could not find datasets_sinteticos.py and no notebook was provided for fallback")

    print("dataset_helpers: extracted from notebook helper cell", flush=True)
    install_notebook_helpers(notebook, namespace)


def execute_experiment(
    dataset_name: str,
    n_samples: int,
    repeticao: int,
    namespace: dict[str, Any],
    args: argparse.Namespace,
    data_dir: Path,
    cuda_bin: Path,
    cuml_available: bool,
    cu_dbscan: Any,
    cupy_module: Any,
) -> dict[str, Any]:
    load_dataset_from_config = namespace["load_dataset_from_config"]
    sugerir_parametros_dbscan = namespace["sugerir_parametros_dbscan"]

    config = {"dataset_name": dataset_name, "n_samples": int(n_samples)}
    loaded = load_dataset_from_config(config, seed=args.seed, verbose=False)
    x_matrix = loaded["X"]
    d_dim = int(loaded["D"])

    eps_values, minpts_values, min_pts_ref, dim_int = sugerir_parametros_dbscan(
        x_matrix, seed=args.seed, mostrar=False
    )
    eps_values = [float(v) for v in list(eps_values[: args.n_eps])]
    while len(eps_values) < args.n_eps:
        eps_values.append(float(eps_values[-1]))

    eps_ref_idx = len(eps_values) // 2
    eps_ref = float(eps_values[eps_ref_idx])

    minpts_values = [int(v) for v in list(minpts_values[: args.n_minpts])]
    try:
        minpts_ref_idx = minpts_values.index(int(min_pts_ref))
    except ValueError:
        minpts_ref_idx = len(minpts_values) // 2
        min_pts_ref = int(minpts_values[minpts_ref_idx])

    tag = f"{dataset_name}_{n_samples}_{repeticao}"

    saida_multi_eps = run_kernel_cuda(
        "multi_eps",
        x_matrix,
        [int(min_pts_ref), len(eps_values)] + eps_values,
        f"{tag}_multieps",
        data_dir,
        cuda_bin,
        args.timeout_kernel_s,
        args.profile_nvprof,
        args.print_nvprof,
        args.warmup_cuda,
        args.nvprof_dir if args.profile_nvprof else None,
    )

    saida_multi_minpts = run_kernel_cuda(
        "multi_minpts",
        x_matrix,
        [eps_ref, len(minpts_values)] + minpts_values,
        f"{tag}_multiminpts",
        data_dir,
        cuda_bin,
        args.timeout_kernel_s,
        args.profile_nvprof,
        args.print_nvprof,
        args.warmup_cuda,
        args.nvprof_dir if args.profile_nvprof else None,
    )

    saida_multi_both = run_kernel_cuda(
        "multi_both",
        x_matrix,
        [len(eps_values), len(minpts_values)] + eps_values + minpts_values,
        f"{tag}_multiboth",
        data_dir,
        cuda_bin,
        args.timeout_kernel_s,
        args.profile_nvprof,
        args.print_nvprof,
        args.warmup_cuda,
        args.nvprof_dir if args.profile_nvprof else None,
    )

    cuml_1_config_ref_ms = np.nan
    cuml_multi_eps_ms = np.nan
    cuml_multi_minpts_ms = np.nan
    cuml_multi_both_ms = np.nan

    if cuml_available:
        x_matrix_gpu = cupy_module.asarray(np.ascontiguousarray(x_matrix, dtype=np.float32))
        cupy_module.cuda.runtime.deviceSynchronize()
        cuml_grid_tempos = []
        for eps in eps_values:
            linha_tempos = []
            for minpts in minpts_values:
                linha_tempos.append(run_cuml(cu_dbscan, cupy_module, x_matrix_gpu, float(eps), int(minpts)))
            cuml_grid_tempos.append(linha_tempos)

        cuml_1_config_ref_ms = cuml_grid_tempos[eps_ref_idx][minpts_ref_idx]
        cuml_multi_eps_ms = sum(linha[minpts_ref_idx] for linha in cuml_grid_tempos)
        cuml_multi_minpts_ms = sum(cuml_grid_tempos[eps_ref_idx])
        cuml_multi_both_ms = sum(sum(linha) for linha in cuml_grid_tempos)
        del x_matrix_gpu

    multi_eps_ms = float(saida_multi_eps["event_total_ms"])
    multi_minpts_ms = float(saida_multi_minpts["event_total_ms"])
    multi_both_ms = float(saida_multi_both["event_total_ms"])

    return {
        "dataset": dataset_name,
        "D": d_dim,
        "N": int(n_samples),
        "repeticao": int(repeticao),
        "n_eps": len(eps_values),
        "n_minpts": len(minpts_values),
        "n_config_grid": len(eps_values) * len(minpts_values),
        "tempo_multi_eps_ms": multi_eps_ms,
        "tempo_multi_minpts_ms": multi_minpts_ms,
        "tempo_multi_both_ms": multi_both_ms,
        "tempo_cuml_1_chamada_ms": cuml_1_config_ref_ms,
        "tempo_cuml_multi_eps_3_chamadas_ms": cuml_multi_eps_ms,
        "tempo_cuml_multi_minpts_4_chamadas_ms": cuml_multi_minpts_ms,
        "tempo_cuml_multi_both_12_chamadas_ms": cuml_multi_both_ms,
        "speedup_multi_eps_vs_cuml_3_eps": (cuml_multi_eps_ms / multi_eps_ms) if cuml_available else np.nan,
        "speedup_multi_minpts_vs_cuml_4_minpts": (cuml_multi_minpts_ms / multi_minpts_ms) if cuml_available else np.nan,
        "speedup_multi_both_vs_cuml_12_combinacoes": (cuml_multi_both_ms / multi_both_ms) if cuml_available else np.nan,
        "erro": np.nan,
    }


def simplify_results_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for source_col, target_col in OUTPUT_RENAME_MAP.items():
        if target_col not in df.columns and source_col in df.columns:
            df[target_col] = df[source_col]
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    return df[OUTPUT_COLUMNS]


def save_checkpoint(resultados: list[dict[str, Any]], checkpoint_csv: Path) -> pd.DataFrame:
    df = simplify_results_df(pd.DataFrame(resultados))
    tmp_path = checkpoint_csv.with_suffix(checkpoint_csv.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(checkpoint_csv)
    return df


def aggregate_results(df_resultados: pd.DataFrame, output_csv: Path) -> pd.DataFrame:
    df_resultados = simplify_results_df(df_resultados)
    if "erro" in df_resultados.columns:
        df_validos = df_resultados[df_resultados["erro"].isna()].copy()
    else:
        df_validos = df_resultados.copy()

    metricas = [
        "tempo_multi_eps_ms",
        "tempo_multi_minpts_ms",
        "tempo_multi_both_ms",
        "tempo_cuml_1_chamada_ms",
        "tempo_cuml_multi_eps_3_chamadas_ms",
        "tempo_cuml_multi_minpts_4_chamadas_ms",
        "tempo_cuml_multi_both_12_chamadas_ms",
        "speedup_multi_eps_vs_cuml_3_eps",
        "speedup_multi_minpts_vs_cuml_4_minpts",
        "speedup_multi_both_vs_cuml_12_combinacoes",
    ]
    metricas = [m for m in metricas if m in df_validos.columns]
    group_cols = ["dataset", "D", "N"]
    if df_validos.empty or not metricas or any(col not in df_validos.columns for col in group_cols):
        df_agregado = pd.DataFrame()
        df_agregado.to_csv(output_csv, index=False)
        print("No valid rows to aggregate yet.", flush=True)
        return df_agregado

    df_agregado = df_validos.groupby(group_cols)[metricas].agg(["mean", "std"]).round(4)
    df_agregado.columns = [
        f"{metric}_{'media' if stat == 'mean' else 'desvio'}"
        for metric, stat in df_agregado.columns
    ]
    df_agregado = df_agregado.reset_index()
    df_agregado.to_csv(output_csv, index=False)
    return df_agregado


def merge_shard_results(output_dir: Path, num_shards: int) -> pd.DataFrame:
    validate_shard_args(0, num_shards)
    frames = []
    for shard_index in range(num_shards):
        shard_csv = output_dir / f"resultados_brutos_cuml_justo{shard_suffix(shard_index, num_shards)}.csv"
        if shard_csv.exists():
            print("Merging shard:", shard_csv, flush=True)
            frames.append(pd.read_csv(shard_csv))
        else:
            print("Missing shard CSV:", shard_csv, flush=True)

    raw_csv = output_dir / "resultados_brutos_cuml_justo.csv"
    aggregate_csv = output_dir / "resultados_agregados_cuml_justo.csv"

    if not frames:
        df_empty = simplify_results_df(pd.DataFrame())
        df_empty.to_csv(raw_csv, index=False)
        aggregate_results(df_empty, aggregate_csv)
        print("No shard CSVs found to merge.", flush=True)
        return df_empty

    df_merged = simplify_results_df(pd.concat(frames, ignore_index=True))
    df_merged = df_merged.drop_duplicates(["dataset", "N", "repeticao"], keep="last")
    df_merged = df_merged.sort_values(["N", "dataset", "repeticao"]).reset_index(drop=True)
    df_merged.to_csv(raw_csv, index=False)
    aggregate_results(df_merged, aggregate_csv)
    print("Merged raw results:", raw_csv, flush=True)
    print("Merged aggregated results:", aggregate_csv, flush=True)
    return df_merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DBSCAN CUDA multi-parameter benchmark on a GPU cluster.")
    parser.add_argument("--notebook", type=Path, default=Path("dbscan_multi_minpoints_multi_eps.ipynb"))
    parser.add_argument(
        "--cuda-source-file",
        type=Path,
        default=None,
        help="Optional .cu file. Defaults to dbscan_multi_baixo_nivel.cu next to this script when it exists.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/dbscan_cluster"))
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--n-points", default="4000,16000,256000,512000,1000000")
    parser.add_argument("--repeticoes", type=int, default=3)
    parser.add_argument("--n-eps", type=int, default=3)
    parser.add_argument("--n-minpts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-arch", default=os.environ.get("DBSCAN_CUDA_ARCH", "75"))
    parser.add_argument("--timeout-kernel-s", type=int, default=int(os.environ.get("DBSCAN_TIMEOUT_KERNEL_S", "7200")))
    parser.add_argument("--profile-nvprof", action="store_true")
    parser.add_argument("--print-nvprof", action="store_true")
    parser.add_argument("--nvprof-dir", type=Path, default=None)
    parser.add_argument("--warmup-cuda", action="store_true")
    parser.add_argument("--skip-cuml", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--shard-index", type=int, default=int(os.environ.get("DBSCAN_SHARD_INDEX", "0")))
    parser.add_argument("--num-shards", type=int, default=int(os.environ.get("DBSCAN_NUM_SHARDS", "1")))
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Optional suffix for raw/aggregate CSV names. Defaults to _shardXX_of_YY when num-shards > 1.",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge shard CSVs in output-dir and rebuild the final aggregate CSV.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validate_shard_args(args.shard_index, args.num_shards)
    script_dir = Path(__file__).resolve().parent
    project_dir = Path.cwd()
    notebook_path = args.notebook if args.notebook.is_absolute() else project_dir / args.notebook
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_dir / args.output_dir
    work_dir = args.work_dir or output_dir / "work"
    work_dir = work_dir if work_dir.is_absolute() else project_dir / work_dir
    data_dir = work_dir / "data"
    if args.nvprof_dir is None:
        args.nvprof_dir = output_dir / "nvprof"
    else:
        args.nvprof_dir = args.nvprof_dir if args.nvprof_dir.is_absolute() else project_dir / args.nvprof_dir
    output_suffix = args.output_suffix
    if output_suffix is None:
        output_suffix = shard_suffix(args.shard_index, args.num_shards) if args.num_shards > 1 else ""

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merge_only:
        df_merged = merge_shard_results(output_dir, args.num_shards)
        n_erros = int(df_merged["erro"].notna().sum()) if "erro" in df_merged.columns else 0
        print(f"Merged rows: {len(df_merged)} | rows with error: {n_erros}", flush=True)
        return 0 if len(df_merged) > 0 and n_erros == 0 else 2

    work_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_csv = output_dir / f"resultados_brutos_cuml_justo{output_suffix}.csv"
    aggregate_csv = output_dir / f"resultados_agregados_cuml_justo{output_suffix}.csv"
    cuda_src = work_dir / "dbscan_multi_baixo_nivel.cu"
    cuda_bin = work_dir / "dbscan_multi_baixo_nivel"

    print("project_dir:", project_dir, flush=True)
    print("script_dir:", script_dir, flush=True)
    print("notebook:", notebook_path if notebook_path.exists() else "not found; local files will be used", flush=True)
    print("output_dir:", output_dir, flush=True)
    print("work_dir:", work_dir, flush=True)
    print("checkpoint_csv:", checkpoint_csv, flush=True)
    if args.profile_nvprof:
        print("nvprof_dir:", args.nvprof_dir, flush=True)
    print(f"shard: {args.shard_index + 1}/{args.num_shards}", flush=True)

    notebook = load_notebook(notebook_path) if notebook_path.exists() else None
    if args.cuda_source_file is not None:
        cuda_source_file = (
            args.cuda_source_file if args.cuda_source_file.is_absolute() else project_dir / args.cuda_source_file
        )
        print("cuda_source_file:", cuda_source_file, flush=True)
        copy_cuda_source(cuda_source_file, cuda_src)
    elif (script_dir / "dbscan_multi_baixo_nivel.cu").exists():
        cuda_source_file = script_dir / "dbscan_multi_baixo_nivel.cu"
        print("cuda_source_file:", cuda_source_file, flush=True)
        copy_cuda_source(cuda_source_file, cuda_src)
    else:
        if notebook is None:
            raise RuntimeError("Could not find dbscan_multi_baixo_nivel.cu and no notebook was provided for fallback")
        print("cuda_source_file: extracted from notebook %%writefile cell", flush=True)
        write_cuda_source(notebook, cuda_src)
    if not args.no_compile:
        compile_cuda(cuda_src, cuda_bin, args.cuda_arch)

    namespace: dict[str, Any] = {
        "np": np,
        "pd": pd,
        "re": re,
        "time": time,
        "Path": Path,
        "plt": plt,
        "make_blobs": make_blobs,
        "make_moons": make_moons,
        "make_circles": make_circles,
        "NearestNeighbors": NearestNeighbors,
        "SEED": args.seed,
        "RUN_MODE": "cluster_gpu",
    }
    install_dataset_helpers(script_dir, notebook, namespace)

    cuml_available, cu_dbscan, cupy_module = load_cuml(args.skip_cuml)

    datasets = parse_csv_list(args.datasets, str)
    n_points = parse_csv_list(args.n_points, int)
    todas_combinacoes = [(dataset, n, rep) for n in n_points for dataset in datasets for rep in range(args.repeticoes)]
    combinacoes = select_shard(todas_combinacoes, args.shard_index, args.num_shards)
    total_global = len(todas_combinacoes)
    total_execucoes = len(combinacoes)
    chaves_planejadas = set(combinacoes)
    print(f"planned combinations in this shard: {total_execucoes}/{total_global}", flush=True)

    resultados: list[dict[str, Any]] = []
    chaves_ja_feitas: set[tuple[str, int, int]] = set()

    if not args.no_resume and checkpoint_csv.exists():
        df_checkpoint = pd.read_csv(checkpoint_csv)
        erros_ignorados = 0
        for _, row in df_checkpoint.iterrows():
            try:
                chave = (str(row["dataset"]), int(row["N"]), int(row["repeticao"]))
            except Exception:
                continue
            if "erro" in df_checkpoint.columns and pd.notna(row.get("erro")):
                erros_ignorados += 1
                continue
            if chave in chaves_planejadas and chave not in chaves_ja_feitas:
                resultados.append(row.to_dict())
                chaves_ja_feitas.add(chave)
        print(f"Resuming checkpoint: {len(chaves_ja_feitas)}/{total_execucoes} done.", flush=True)
        if erros_ignorados:
            print(f"Checkpoint rows with previous errors will be retried: {erros_ignorados}", flush=True)

    for contador, (dataset, n_samples, rep) in enumerate(combinacoes, start=1):
        chave = (dataset, n_samples, rep)
        if chave in chaves_ja_feitas:
            print(f"[{contador}/{total_execucoes}] skip checkpoint dataset={dataset} N={n_samples} rep={rep+1}/{args.repeticoes}", flush=True)
            continue

        print(f"[{contador}/{total_execucoes}] dataset={dataset} N={n_samples} rep={rep+1}/{args.repeticoes}", flush=True)
        try:
            linha = execute_experiment(
                dataset,
                n_samples,
                rep,
                namespace,
                args,
                data_dir,
                cuda_bin,
                cuml_available,
                cu_dbscan,
                cupy_module,
            )
            resultados.append(linha)
        except Exception as exc:
            print(f"  ERROR: {exc}", flush=True)
            resultados.append({"dataset": dataset, "N": n_samples, "repeticao": rep, "erro": str(exc)})
        chaves_ja_feitas.add(chave)
        save_checkpoint(resultados, checkpoint_csv)
        print("  checkpoint saved:", checkpoint_csv, flush=True)

    df_resultados = save_checkpoint(resultados, checkpoint_csv)
    aggregate_results(df_resultados, aggregate_csv)
    n_erros = int(df_resultados["erro"].notna().sum()) if "erro" in df_resultados.columns else 0
    print(f"Completed rows: {len(df_resultados)} | rows with error: {n_erros}", flush=True)
    print("Raw results:", checkpoint_csv, flush=True)
    print("Aggregated results:", aggregate_csv, flush=True)
    return 0 if n_erros == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
