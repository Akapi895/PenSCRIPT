"""
Validates all compiled scenarios can be loaded by NASim.
"""
import os
import sys
from pathlib import Path
from nasim.scenarios import load_scenario


def validate_all(compiled_dir: str):
    """Load every compiled YAML and report results."""
    files = sorted(Path(compiled_dir).glob("*.yml"))
    print(f"Validating {len(files)} compiled scenarios...")

    ok = 0
    fail = 0
    errors = []

    for p in files:
        try:
            sc = load_scenario(str(p))
            ok += 1
        except Exception as e:
            fail += 1
            errors.append((p.name, str(e)))
            print(f"  FAIL {p.name}: {e}")

    print(f"\nResults: {ok} OK, {fail} FAIL out of {ok + fail}")

    for tier in range(1, 5):
        tier_ok = sum(1 for f in files
                      if f"_T{tier}_" in f.name
                      and (f.name, None) not in [(e[0], None) for e in errors])
        tier_total = sum(1 for f in files if f"_T{tier}_" in f.name)
        tier_fail = tier_total - tier_ok
        # recalculate properly
        tier_failed = [e[0] for e in errors if f"_T{tier}_" in e[0]]
        tier_ok2 = tier_total - len(tier_failed)
        print(f"  T{tier}: {tier_ok2}/{tier_total} OK")

    if errors:
        print(f"\nFailed files ({len(errors)}):")
        for name, err in errors[:20]:
            print(f"  {name}: {err}")

    return ok, fail


if __name__ == '__main__':
    project_root = Path(__file__).parent.parent.parent
    compiled_dir = project_root / 'data' / 'scenarios' / 'generated' / 'compiled'
    validate_all(str(compiled_dir))
