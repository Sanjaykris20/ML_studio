import os
import zipfile
from pathlib import Path


def create_archive():
    root = Path(__file__).resolve().parent
    out_zip = root / "ML_Studio_FullProject.zip"
    if out_zip.exists():
        out_zip.unlink()

    exclude_dirs = {
        "node_modules",
        "__pycache__",
        ".pytest_cache",
        ".git",
        ".idea",
        ".vscode",
        "_legacy_archive",
    }
    exclude_files = {out_zip.name}

    count = 0
    print(f"📦 Packaging ML Studio project into {out_zip.name}...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in root.rglob("*"):
            if file_path.is_dir():
                continue
            parts = set(file_path.relative_to(root).parts)
            if any(ex in parts for ex in exclude_dirs) or file_path.name in exclude_files:
                continue
            zf.write(file_path, arcname=str(file_path.relative_to(root)))
            count += 1

    size_mb = round(out_zip.stat().st_size / (1024 * 1024), 2)
    print(f"✅ SUCCESS: Packaged {count} files into {out_zip.name} ({size_mb} MB)")
    print(f"📁 Full Path: {out_zip}")


if __name__ == "__main__":
    create_archive()
