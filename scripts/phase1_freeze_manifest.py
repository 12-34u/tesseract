"""Build or verify the Phase 1 freeze manifest.

Phase 1 was completed and verified at commit a69054f ("Made changes and verified
Phase 1 of TESSERACT before starting phase 2"). Everything in that commit that is
not Phase 2 work product is frozen: no Phase 2 step may change it.

The manifest is a committed file so the guarantee survives across sessions and
machines; an earlier version of this check lived in a temporary directory and was
lost with it. Regenerating requires --build and is refused when any frozen file
already differs, so the manifest can never be quietly re-baselined onto drift.

    python scripts/phase1_freeze_manifest.py           # verify
    python scripts/phase1_freeze_manifest.py --build   # (re)generate
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from phase2.integrity import (
    DEFAULT_MANIFEST_PATH,
    build_phase1_manifest,
    check_phase1_frozen,
    repo_root,
    write_phase1_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="regenerate the manifest from the freeze commit")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args()

    root = repo_root()
    if args.build:
        manifest = build_phase1_manifest(root)
        write_phase1_manifest(manifest, args.manifest)
        print(f"wrote {args.manifest}: {len(manifest['files'])} frozen files "
              f"at commit {manifest['freeze_commit'][:7]}")
        return 0

    report = check_phase1_frozen(args.manifest, root)
    print(f"Phase 1 freeze commit : {report['freeze_commit'][:7]}")
    print(f"frozen files verified : {report['files_verified']}/{report['files_total']}")
    for label, paths in (("MODIFIED", report["modified"]), ("MISSING", report["missing"])):
        for path in paths:
            print(f"  {label}: {path}")
    print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
