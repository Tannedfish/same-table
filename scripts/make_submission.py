"""Assemble the Canvas submission folder in the layout the project brief asks for.

    python scripts/make_submission.py --group 01 --leader 07123456

Creates submission/Group01_07123456/ (and a .zip next to it). Put your own Project_report.pdf,
Presentation_slide.pptx and grpNN.mp4 in the marked folders before zipping for real.
Run both notebooks first so artifacts/ holds the fine-tuned model and Experimental_results.xlsx.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_FILES = ["app.py", "requirements.txt", "README.md", ".streamlit/config.toml"]
APP_DIRS = ["same_table", "data", "scripts", "tests"]


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    else:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", required=True, help="two-digit group number, e.g. 01")
    parser.add_argument("--leader", required=True, help="group leader's student ID")
    args = parser.parse_args()
    g = f"Group{args.group}"
    base = ROOT / "submission" / f"{g}_{args.leader}"
    shutil.rmtree(base, ignore_errors=True)

    missing = []

    def need(path: Path) -> Path:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
        return path

    docs = base / f"{g}_documentation"
    xlsx = need(ROOT / "artifacts" / "Experimental_results.xlsx")
    if xlsx.exists():
        copy(xlsx, docs / "Experimental_results.xlsx")
    (docs / "PUT_Project_report.pdf_HERE.txt").parent.mkdir(parents=True, exist_ok=True)
    (docs / "PUT_Project_report.pdf_HERE.txt").write_text("Project_report.pdf (under 10 pages) goes in this folder.\n")

    program = base / f"{g}_program"
    for nb in sorted((ROOT / "notebooks").glob("*.ipynb")):
        copy(nb, program / "Python_notebooks" / nb.name)
    for name in APP_FILES:
        if (ROOT / name).exists():
            copy(ROOT / name, program / "GitHub_App_Files" / name)
    for name in APP_DIRS:
        copy(ROOT / name, program / "GitHub_App_Files" / name)

    dataset = base / f"{g}_Dataset_files"
    copy(ROOT / "data", dataset / "data")
    model = need(ROOT / "artifacts" / "same-table-tagger")
    if model.exists():
        copy(model, dataset / "Fine-tuned_Model_files" / "same-table-tagger")

    pres = base / f"{g}_presentation"
    pres.mkdir(parents=True, exist_ok=True)
    (pres / "PUT_slides_and_video_HERE.txt").write_text(f"Presentation_slide.pptx and grp{args.group}.mp4 go in this folder.\n")
    (base / "Streamlit_Cloud_App_URL.txt").write_text("App URL: https://<your-app>.streamlit.app\nGitHub: https://github.com/<user>/<repo>\nModel: https://huggingface.co/<user>/same-table-tagger\n")

    archive = shutil.make_archive(str(base), "zip", root_dir=base.parent, base_dir=base.name)
    print("created", base.relative_to(ROOT), "and", Path(archive).relative_to(ROOT))
    if missing:
        print("MISSING (run the notebooks first):", ", ".join(missing))


if __name__ == "__main__":
    main()
