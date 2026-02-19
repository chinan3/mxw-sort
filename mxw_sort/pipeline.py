from pathlib import Path
import time
import numpy as np

from .config import PipelineConfig
from .io_maxwell import read_maxwell, get_available_wells, get_well_duration_s
from .preprocess import unsigned_to_signed, slice_seconds, bandpass_to_frac_nyq
from .export import write_binary, write_probe_json, write_meta_json
from .ks4 import run_ks4
from .qc import write_qc


def _ks4_done(ks_dir: Path) -> bool:
    # dummy way to check if ks4 has completed its run
    return (ks_dir / "spike_times.npy").exists() and (ks_dir / "spike_clusters.npy").exists()

def process_one_well(
    h5_path: str,
    out_root: Path,
    cfg: PipelineConfig,
    well_idx: int,
    skip_existing: bool = True,
    dry_run: bool = False,
):
    """
    Modular single-well processing pipeline, wells run serially.

    But what are we looking at?

    Pipeline stages:
        1. Read the Maxwell recording and apply preprocessing
        2. Export to binary format for speed/ease of reading by KS. Alongside it, export the recording probe geometry
        3. Call/run Kilosort4
        4. Generate a handful of computationally cheap QC plots. KS4 will also create its own QC plots. (Does that make this step redundant??)

    Args:
        h5_path: Path to Maxwell .raw.h5 file
        out_root: Root dir for outputs 
        cfg: Pipeline configuration
        well_idx: Well index (0-5)
        skip_existing: Skip if KS4 outputs already exist
        dry_run: Print only actions, without executing
            
    Raises:
        FileNotFoundError: If the h5_path doesn't exist
        ValueError: If well_idx is invalid
    """
    stream = f"well{well_idx:03d}"

    well_dir = out_root / stream
    prep_dir = well_dir / "preprocessed"
    ks_dir = well_dir / "ks4"
    qc_dir = well_dir / "qc"

    bin_path = prep_dir / "traces.bin"
    probe_path = prep_dir / "ks4_probe.json"
    xy_path = prep_dir / "channel_xy.npy"
    meta_path = prep_dir / "meta.json"

    if skip_existing and _ks4_done(ks_dir):
        print(f"[SKIP] {h5_path} {stream} (ks4 outputs exist)")
        return

    print(f"[RUN] {h5_path} {stream} -> {well_dir}")

    if dry_run:
        try:
            dur = get_well_duration_s(h5_path, stream)
            print(f"  duration: {dur:.1f}s")
        except Exception:
            print("  duration: unknown")
        print("  (dry-run) would write:", bin_path)
        print("  (dry-run) would write:", probe_path)
        print("  (dry-run) would run ks4 into:", ks_dir)
        print("  (dry-run) would write qc into:", qc_dir)
        return

    t0 = time.time()

    prep_dir.mkdir(parents=True, exist_ok=True)
    ks_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Read h5, run preprocessers
    rec = read_maxwell(h5_path, stream)
    rec = unsigned_to_signed(rec)
    rec = slice_seconds(rec, cfg.start_s, cfg.dur_s)
    rec = bandpass_to_frac_nyq(rec, cfg.bp_min_hz, cfg.bp_max_frac_nyq)

    # Get sampling frequency from the actual data
    fs_hz = rec.get_sampling_frequency()

    # Export binary, export channel xy's
    write_binary(rec, bin_path)
    xy = rec.get_channel_locations()
    np.save(xy_path, xy)
    write_probe_json(xy, probe_path)

    meta = {
        "h5": h5_path,
        "stream": stream,
        "fs_hz": float(fs_hz),
        "start_s": cfg.start_s,
        "dur_s": cfg.dur_s,
        "bp_min_hz": cfg.bp_min_hz,
        "bp_max_frac_nyq": cfg.bp_max_frac_nyq,
        "n_chan": int(xy.shape[0]),
    }
    write_meta_json(meta, meta_path)

    # Run KS4 (AVOID double highpass filtering, triple check preprocess, export, and config)
    run_ks4(
        bin_file=bin_path,
        probe_path=probe_path,
        out_dir=ks_dir,
        fs_hz=float(fs_hz),
        n_chan=int(xy.shape[0]),
        batch_size=cfg.ks4_batch_size,
        highpass_cutoff_hz=cfg.ks4_highpass_cutoff_hz,
    )

    # QC
    dur_s_processed = cfg.dur_s if cfg.dur_s is not None else None # redundant?
    write_qc(ks_dir=ks_dir, qc_dir=qc_dir, fs_hz=float(fs_hz), dur_s_processed=dur_s_processed)

    elapsed = time.time() - t0
    print(f"[DONE] {stream} in {elapsed:.1f}s")

# Process multiple wells from a Maxwell H5 File
# Multiple wells could be run in parallel in the future. ProcessPoolExecutor would be added around here.
def process_h5(
    h5_path: str,
    out_root: Path,
    cfg: PipelineConfig,
    wells: tuple[int, ...] | None = None,
    skip_existing: bool = True,
    dry_run: bool = False,
    only_well: int | None = None,
):
    # Handle only_well override
    if only_well is not None:
        wells = (only_well,)
    # Auto-detect wells if not specified
    elif wells is None:
        wells = get_available_wells(h5_path)
        print(f"Auto-detected {len(wells)} wells: {wells}")

    for w in wells:
        process_one_well(
            h5_path=h5_path,
            out_root=out_root,
            cfg=cfg,
            well_idx=w,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )


# Recursively finds all .h5 files under a root directory and processes each one
# Output directories mirror the input directory structure
def process_directory(
    root_dir: Path,
    out_root: Path,
    cfg: PipelineConfig,
    wells: tuple[int, ...] | None = None,
    skip_existing: bool = True,
    dry_run: bool = False,
    only_well: int | None = None,
):
    h5_files = sorted(root_dir.rglob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found under {root_dir}")
        return

    print(f"Found {len(h5_files)} .h5 file(s) under {root_dir}:")
    for f in h5_files:
        try:
            detected = get_available_wells(str(f))
            stream = f"well{detected[0]:03d}" if detected else "well000"
            dur = get_well_duration_s(str(f), stream)
            print(f"  {f}  ({dur:.1f}s)")
        except Exception:
            print(f"  {f}")
    print()

    for h5_file in h5_files:
        file_out = out_root / h5_file.relative_to(root_dir).parent
        file_out.mkdir(parents=True, exist_ok=True)
        process_h5(
            h5_path=str(h5_file),
            out_root=file_out,
            cfg=cfg,
            wells=wells,
            skip_existing=skip_existing,
            dry_run=dry_run,
            only_well=only_well,
        )


# Flat directory mode: all h5 files are in one folder with unique names.
# Output folders are named after each file's stem (e.g. "recording_001.raw.h5" -> "recording_001.raw/")
def process_directory_flat(
    root_dir: Path,
    out_root: Path,
    cfg: PipelineConfig,
    wells: tuple[int, ...] | None = None,
    skip_existing: bool = True,
    dry_run: bool = False,
    only_well: int | None = None,
):
    h5_files = sorted(root_dir.glob("*.h5"))
    if not h5_files:
        print(f"No .h5 files found in {root_dir}")
        return

    print(f"[FLAT] Found {len(h5_files)} .h5 file(s) in {root_dir}:")
    for f in h5_files:
        try:
            detected = get_available_wells(str(f))
            stream = f"well{detected[0]:03d}" if detected else "well000"
            dur = get_well_duration_s(str(f), stream)
            print(f"  {f.name}  ({dur:.1f}s)")
        except Exception:
            print(f"  {f.name}")
    print()

    for h5_file in h5_files:
        # Use the file stem (minus .h5) as the output subdirectory
        file_out = out_root / h5_file.stem
        file_out.mkdir(parents=True, exist_ok=True)
        process_h5(
            h5_path=str(h5_file),
            out_root=file_out,
            cfg=cfg,
            wells=wells,
            skip_existing=skip_existing,
            dry_run=dry_run,
            only_well=only_well,
        )
