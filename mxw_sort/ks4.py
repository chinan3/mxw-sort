from pathlib import Path
from kilosort import run_kilosort, DEFAULT_SETTINGS

# Runs KS4
def run_ks4(
    bin_file: Path,
    probe_path: Path,
    out_dir: Path,
    fs_hz: float,
    n_chan: int,
    batch_size: int,
    highpass_cutoff_hz: float,
):
    settings = DEFAULT_SETTINGS.copy()
    settings["filename"] = str(bin_file)
    settings["probe_path"] = str(probe_path)
    settings["results_dir"] = str(out_dir)  # KS4 will create outputs here
    settings["fs"] = float(fs_hz)
    settings["n_chan_bin"] = int(n_chan)

    # your overrides
    settings["batch_size"] = int(batch_size)
    settings["highpass_cutoff"] = float(highpass_cutoff_hz)

    run_kilosort(settings=settings)
