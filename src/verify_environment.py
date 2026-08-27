"""Verify the Python environment required by this project."""

import importlib


REQUIRED_PACKAGES = ("pandas", "numpy", "sklearn", "streamlit")


def main() -> None:
    print("Python environment check")
    print("------------------------")
    print("Environment verification started.")

    missing_packages = []
    for package_name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(package_name)
            version = getattr(module, "__version__", "unknown")
            print(f"{package_name}: {version}")
        except ImportError:
            missing_packages.append(package_name)
            print(f"{package_name}: MISSING")

    if missing_packages:
        missing = ", ".join(missing_packages)
        raise SystemExit(
            f"Environment verification failed. Install missing packages: {missing}"
        )

    print("Environment setup successful!")


if __name__ == "__main__":
    main()
