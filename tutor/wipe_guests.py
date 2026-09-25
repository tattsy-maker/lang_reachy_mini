"""End-of-day wipe: delete every guest-tier learner, keep the family.

    python tutor/wipe_guests.py [--root learners] [--dry-run]

Part of the booth shutdown path (spec section 8: guests expire
automatically; T11 wires this into the shutdown script).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tutor.store import LearnerStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default="learners",
                    help="learner store root (default: ./learners)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be deleted, delete nothing")
    args = ap.parse_args()

    store = LearnerStore(args.root)
    learners = store.list()
    guests = [l for l in learners if l.tier == "guest"]
    family = [l for l in learners if l.tier == "family"]

    if not guests:
        print(f"no guest profiles under {store.root} "
              f"({len(family)} family kept)")
        return 0

    for guest in guests:
        if args.dry_run:
            print(f"would delete guest {guest.id} ({guest.name!r}, "
                  f"{guest.sessions} sessions)")
        else:
            store.delete(guest.id)
            print(f"deleted guest {guest.id} ({guest.name!r})")
    print(f"{'would keep' if args.dry_run else 'kept'} "
          f"{len(family)} family profile(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
