
import os
import sys
import subprocess
from pathlib import Path
import click

DEFAULT_SCRIPTS_DIR = Path.cwd() / "scripts"

def _ensure_exists(path: Path, what: str):
    if not path.exists():
        raise click.ClickException(f"{what} not found: {path}")

def _resolve_scripts_dir(custom: Path | None) -> Path:
    scripts = custom if custom else DEFAULT_SCRIPTS_DIR
    _ensure_exists(scripts, "scripts/ directory")
    return scripts

def _resolve_root_dir(root_dir_opt: Path | None) -> Path:
    env = os.getenv("AVCV_ROOT_DIR")
    root = root_dir_opt or (Path(env) if env else None)
    if not root:
        raise click.ClickException(
            "ROOT_DIR is required. Pass --root-dir or set environment variable AVCV_ROOT_DIR."
        )
    return root

def _run_python(script: Path, env: dict | None = None, args: list[str] | None = None):
    _ensure_exists(script, "Script")
    cmd = [sys.executable, str(script)]
    if args:
        cmd.extend(args)
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        raise click.ClickException(f"Subprocess failed with exit code {e.returncode}") from e

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.version_option(package_name="avcv", prog_name="avcv")
def cli():
    """AVCV command line interface.

    Commands:
      - run: launch scripts/AVCV.py with a chosen ROOT_DIR (sets env AVCV_ROOT_DIR)
      - compare: run scripts/Comparison.py with base/secondary CSV names
      - scaffold: create a tiny starter structure (work/, images/, videos/)
      - env: create & install a local Python venv, or print conda instructions
      - validate: quick checks for expected files under ROOT_DIR
    """
    pass

@cli.command("run")
@click.option(
    "--root-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Path to your data folder (where work/ and CSVs live). If omitted, uses $AVCV_ROOT_DIR.",
)
@click.option(
    "--scripts-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Custom scripts directory (default: ./scripts).",
)
@click.option("--avcv-script", type=str, default="AVCV.py", show_default=True,
              help="Name of the viewer script inside scripts-dir.")
def run_viewer(root_dir: Path | None, scripts_dir: Path | None, avcv_script: str):
    """Launch the viewer (scripts/AVCV.py). Sets env var AVCV_ROOT_DIR for the subprocess."""
    root_dir = _resolve_root_dir(root_dir)
    scripts = _resolve_scripts_dir(scripts_dir)
    env = os.environ.copy()
    env["AVCV_ROOT_DIR"] = str(root_dir)
    avcv = scripts / avcv_script
    click.echo(f"Launching viewer: {avcv}\nAVCV_ROOT_DIR={root_dir}")
    _run_python(avcv, env=env)

@cli.command("compare")
@click.option(
    "--root-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Path to your data folder (used as AVCV_ROOT_DIR). If omitted, uses $AVCV_ROOT_DIR.",
)
@click.option("--base", "base_csv", required=True, help="Base tracks CSV (e.g., CME_tracks.csv)")
@click.option("--secondary", "sec_csv", required=True, help="Secondary tracks CSV (e.g., Dino_tracks.csv)")
@click.option(
    "--scripts-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Custom scripts directory (default: ./scripts).",
)
@click.option("--comparison-script", type=str, default="Comparison.py", show_default=True,
              help="Name of the comparison script inside scripts-dir.")
@click.option("--passthrough", "extra", help="Optional extra args passed to Comparison.py", default="")
def compare(root_dir: Path | None, base_csv: str, sec_csv: str, scripts_dir: Path | None,
            comparison_script: str, extra: str):
    """Run comparison/coverage generation via scripts/Comparison.py.
    Tries to pass --base/--secondary; if the script doesn't accept flags, retries without.
    """
    root_dir = _resolve_root_dir(root_dir)
    env = os.environ.copy()
    env["AVCV_ROOT_DIR"] = str(root_dir)
    scripts = _resolve_scripts_dir(scripts_dir)
    comp = scripts / comparison_script
    click.echo(f"Running comparison: base={base_csv}, secondary={sec_csv}\nAVCV_ROOT_DIR={root_dir}")
    args = ["--base", base_csv, "--secondary", sec_csv]
    if extra:
        args.extend(extra.split())
    try:
        _run_python(comp, env=env, args=args)
    except click.ClickException:
        click.echo("Comparison.py may not accept CLI flags; trying without flags …")
        _run_python(comp, env=env)

@cli.command("scaffold")
@click.option(
    "--into",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path.cwd(),
    show_default=True,
    help="Where to create folders (default: current working directory).",
)
@click.option("--channels", type=int, default=2, show_default=True, help="How many channel folders to create.")
def scaffold(into: Path, channels: int):
    """Create a small starter structure so new users can plug in data quickly."""
    work = into / "work"
    for c in range(1, channels + 1):
        (work / f"Channel_{c}" / "001").mkdir(parents=True, exist_ok=True)
    (into / "images").mkdir(exist_ok=True)
    (into / "videos").mkdir(exist_ok=True)
    click.echo(f"Scaffold created under: {into}")
    click.echo("Next steps:")
    click.echo("  1) Put TIFFs under work/Channel_*/001,002,…")
    click.echo("  2) Put CSVs (CME_tracks.csv, Dino_tracks.csv, detections_*.csv) in work/")
    click.echo("  3) Run: avcv run --root-dir <path-to-your-folder>")

@cli.command("env")
@click.option("--venv", is_flag=True, help="Create a Python venv in .venv and install requirements.txt here.")
@click.option("--name", default="avcv", show_default=True, help="Conda environment name (printed as instructions).")
def env_cmd(venv: bool, name: str):
    """Set up a development environment.

    - With --venv: creates a local .venv and installs requirements.txt
    - Without --venv: prints exact Conda commands to create env from environment.yml
    """
    if venv:
        venv_dir = Path.cwd() / ".venv"
        python = sys.executable
        click.echo(f"Creating venv at {venv_dir} …")
        subprocess.run([python, "-m", "venv", str(venv_dir)], check=True)
        pip_path = venv_dir / ("Scripts" if os.name == "nt" else "bin") / "pip"
        req = Path.cwd() / "requirements.txt"
        if not req.exists():
            raise click.ClickException("requirements.txt not found in current directory.")
        click.echo("Installing dependencies from requirements.txt …")
        subprocess.run([str(pip_path), "install", "-r", str(req)], check=True)
        click.echo("Done. Activate with:\n  . .venv/bin/activate    # Windows: .venv\\Scripts\\activate")
    else:
        click.echo("Conda setup (copy/paste):\n")
        click.echo(f"conda env create -f environment.yml -n {name}")
        click.echo(f"conda activate {name}")
        click.echo("# If environment exists already:")
        click.echo(f"conda env update -f environment.yml -n {name}")

@cli.command("validate")
@click.option(
    "--root-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Path to your data folder (used as AVCV_ROOT_DIR). If omitted, uses $AVCV_ROOT_DIR.",
)
def validate(root_dir: Path | None):
    """Quick checks for expected files/folders under ROOT_DIR/work."""
    root = _resolve_root_dir(root_dir)
    work = root / "work"
    ok = work.exists()
    ch1 = work / "Channel_1"
    ch2 = work / "Channel_2"
    msg = []
    msg.append(f"ROOT_DIR: {root}")
    msg.append(f"work/: {'OK' if ok else 'MISSING'}")
    msg.append(f"Channel_1: {'OK' if ch1.exists() else 'MISSING'}")
    msg.append(f"Channel_2: {'OK' if ch2.exists() else 'MISSING'}")
    for fn in ["CME_tracks.csv", "Dino_tracks.csv", "detections_CME.csv", "detections_Dino.csv"]:
        msg.append(f"{fn}: {'OK' if (work/fn).exists() else 'missing (optional)'}")
    click.echo("\n".join(msg))

if __name__ == "__main__":
    cli()
