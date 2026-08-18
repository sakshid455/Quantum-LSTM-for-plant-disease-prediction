"""Test environment readiness for the QuantumPlantDisease project."""
import importlib, sys
from pathlib import Path

REQUIRED_PACKAGES = ["numpy","pandas","torch","torchvision","torchaudio","cv2","sklearn","scipy","matplotlib","PIL","streamlit"]
EXPECTED_DIRS = ["data/raw","data/processed","data/meta","data/cache","src/dataset","src/preprocessing","src/models","src/training","src/evaluation","src/quantum","src/utils","notebooks","outputs","checkpoints","configs","app","docs"]

def check_packages():
    res = {}
    for pkg in REQUIRED_PACKAGES:
        try:
            m = importlib.import_module(pkg)
            res[pkg] = "OK (" + str(getattr(m, "__version__", "unknown")) + ")"
        except ImportError:
            res[pkg] = "MISSING"
    return res

def check_directories():
    return {d: Path(d).is_dir() for d in EXPECTED_DIRS}

def main():
    print("=" * 60)
    print("QuantumPlantDisease - Environment Test")
    print("=" * 60)
    print("\n[1] Python version:", sys.version.split()[0])
    print("    ->", "OK (>= 3.9)" if sys.version_info >= (3, 9) else "WARNING: < 3.9")

    print("\n[2] Required packages:")
    pkgs = check_packages()
    missing = [p for p, s in pkgs.items() if s == "MISSING"]
    for p, s in pkgs.items():
        print("    {:<20} {}".format(p, s))
    print("    " + ("MISSING: " + ", ".join(missing) if missing else "All packages available."))

    print("\n[3] Project directories:")
    dirs = check_directories()
    missing_dirs = [d for d, ok in dirs.items() if not ok]
    for d, ok in dirs.items():
        print("    {:<30} {}".format(d, "OK" if ok else "MISSING"))
    print("    " + ("MISSING: " + ", ".join(missing_dirs) if missing_dirs else "All directories present."))

    print("\n" + "=" * 60)
    if not missing and not missing_dirs and sys.version_info >= (3, 9):
        print("RESULT: Environment is READY.")
    else:
        print("RESULT: Environment has issues to resolve (see above).")
    print("=" * 60)

if __name__ == "__main__":
    main()
