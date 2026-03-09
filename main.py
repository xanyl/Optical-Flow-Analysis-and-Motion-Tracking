import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "video_confirm.py",
    "extract_frames.py",
    "optical_flow_analysis.py",
    "plot_motion_stats.py",
    "tracking_validation.py",
]


def run_script(script_path: Path, python_executable: str) -> int:
    print(f"\n{'=' * 80}")
    print(f"Running: {script_path.name}")
    print(f"{'=' * 80}\n")

    result = subprocess.run(
        [python_executable, str(script_path)],
        cwd=str(script_path.parent),
    )

    print(f"\nFinished: {script_path.name} | exit code = {result.returncode}")
    return result.returncode



def main():
    parser = argparse.ArgumentParser(
        description="Run the existing project scripts one by one without modifying them."
    )
    parser.add_argument(
        "--dir",
        default=".",
        help="Directory that contains the existing script files.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running the remaining scripts even if one fails.",
    )
    args = parser.parse_args()

    base_dir = Path(args.dir).resolve()
    python_executable = sys.executable

    print(f"Using Python: {python_executable}")
    print(f"Script directory: {base_dir}")

    missing = [name for name in SCRIPTS if not (base_dir / name).is_file()]
    if missing:
        print("\nError: these files were not found:")
        for name in missing:
            print(f"  - {name}")
        sys.exit(1)

    results = []
    for name in SCRIPTS:
        script_path = base_dir / name
        exit_code = run_script(script_path, python_executable)
        results.append((name, exit_code))

        if exit_code != 0 and not args.continue_on_error:
            print("\nStopped because a script failed.")
            break

    print(f"\n{'=' * 80}")
    print("Run summary")
    print(f"{'=' * 80}")
    for name, exit_code in results:
        status = "OK" if exit_code == 0 else "FAILED"
        print(f"{name:<30} {status} (exit code {exit_code})")

    if any(code != 0 for _, code in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
