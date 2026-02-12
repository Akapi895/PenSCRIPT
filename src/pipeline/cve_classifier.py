"""
CVE Classifier — Phase 1 of the CVE Difficulty & Expansion Pipeline.

Reads CVE_dataset.csv, computes a composite difficulty score per CVE,
assigns difficulty tiers (T1-T4), and writes cve_graded.csv.

Difficulty Score Formula (from docs/cve_difficulty_and_expansion.md §3.2):
  S_diff = w1·f_prob + w2·f_AC + w3·f_PR + w4·f_UI

  where:
    f_prob = 1 - prob         (0.01, 0.20, 0.40, 0.60)
    f_AC   = LOW→0, MED→0.5, HIGH→1.0
    f_PR   = NONE→0, LOW→0.33, HIGH→0.67, empty→0
    f_UI   = NONE→0, REQUIRED→1.0

  weights: w1=0.50, w2=0.25, w3=0.15, w4=0.10

Tier Thresholds:
    T1 Easy   : [0.0,  0.15)
    T2 Medium : [0.15, 0.35)
    T3 Hard   : [0.35, 0.55)
    T4 Expert : [0.55, 1.0]

Service Abstraction (§5.7):
    webapp, iis → http
    windows → smb
    brightstor → http
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter, defaultdict


# ─── Feature Encoding Maps ──────────────────────────────────────────────────

AC_MAP = {'LOW': 0.0, 'MEDIUM': 0.5, 'HIGH': 1.0}
PR_MAP = {'': 0.0, 'NONE': 0.0, 'LOW': 0.33, 'HIGH': 0.67}
UI_MAP = {'NONE': 0.0, 'REQUIRED': 1.0, '': 0.0}

# Weights (§3.2)
W_PROB = 0.50
W_AC   = 0.25
W_PR   = 0.15
W_UI   = 0.10

# Tier thresholds — adjusted based on actual score distribution analysis:
#   prob=0.99 → score ≈ 0.005-0.06  (T1)
#   prob=0.80 → score ≈ 0.10-0.18   (T2)
#   prob=0.60 → score ≈ 0.20-0.35   (T3)
#   prob=0.40 → score ≈ 0.30-0.55   (T4, overlaps with high-AC prob=0.60)
# Using thresholds that align with natural prob clusters:
TIER_THRESHOLDS = [(0.10, 1), (0.20, 2), (0.30, 3)]  # (upper_bound, tier)
# T1: [0, 0.10) ≈ prob=0.99,  T2: [0.10, 0.20) ≈ prob=0.80
# T3: [0.20, 0.30) ≈ prob=0.60 easy,  T4: [0.30, 1.0] ≈ prob≤0.40 + hard 0.60

# PenGym-compatible services (§1.4)
PENGYM_SERVICES = {'ssh', 'ftp', 'http', 'samba', 'smtp'}

# Abstract service mapping (§5.7, Phase 1)
SERVICE_ABSTRACT_MAP = {
    'webapp':     'http',
    'iis':        'http',
    'brightstor': 'http',
    'windows':    'smb',
}

# Services to exclude (client-side, not network pentest relevant)
EXCLUDE_SERVICES = {'browser', 'fileformat'}


class CVEClassifier:
    """Classifies CVEs by computing difficulty scores and assigning tiers.

    Usage:
        classifier = CVEClassifier()
        classifier.load_csv('data/CVE/CVE_dataset.csv')
        classifier.classify()
        classifier.save_graded_csv('data/CVE/cve_graded.csv')
        report = classifier.get_distribution_report()

    With extensible registry (Phase 4):
        from .extensible_registry import ServiceRegistry, ServiceDefinition
        registry = ServiceRegistry()
        registry.register_service(ServiceDefinition('mssql', 'e_mssql', ['mssql']))
        classifier = CVEClassifier(registry=registry)
    """

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 tier_thresholds: Optional[List[Tuple[float, int]]] = None,
                 abstract_services: bool = True,
                 exclude_client_side: bool = True,
                 registry=None):
        """
        Args:
            weights: Override default {prob, ac, pr, ui} weights.
            tier_thresholds: Override default tier boundaries.
            abstract_services: Apply service abstraction mapping (webapp→http...).
            exclude_client_side: Mark browser/fileformat CVEs as excluded.
            registry: Optional ServiceRegistry for extensible keyword mapping.
                      When provided, overrides SERVICE_ABSTRACT_MAP and PENGYM_SERVICES.
        """
        self.w_prob = weights.get('prob', W_PROB) if weights else W_PROB
        self.w_ac   = weights.get('ac',   W_AC)  if weights else W_AC
        self.w_pr   = weights.get('pr',   W_PR)  if weights else W_PR
        self.w_ui   = weights.get('ui',   W_UI)  if weights else W_UI

        self.tier_thresholds = tier_thresholds or TIER_THRESHOLDS
        self.abstract_services = abstract_services
        self.exclude_client_side = exclude_client_side
        self.registry = registry  # Phase 4: extensible service registry

        # Build effective service sets from registry if provided
        if registry is not None:
            self._pengym_services = set(registry.service_names)
            # Build abstract map from registry keywords
            self._abstract_map = {}
            for svc_name in registry.service_names:
                svc = registry.get_service(svc_name)
                for kw in svc.cve_keywords:
                    if kw != svc_name:
                        self._abstract_map[kw] = svc_name
        else:
            self._pengym_services = PENGYM_SERVICES
            self._abstract_map = SERVICE_ABSTRACT_MAP

        self.rows: List[Dict] = []
        self.fieldnames: List[str] = []

    def load_csv(self, csv_path: str) -> int:
        """Load CVE_dataset.csv. Returns number of rows loaded."""
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CVE dataset not found: {csv_path}")

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.fieldnames = list(reader.fieldnames)
            self.rows = list(reader)

        return len(self.rows)

    def classify(self) -> None:
        """Compute difficulty_score, difficulty_tier, and mapped_service for each CVE."""
        for row in self.rows:
            # Compute difficulty score
            score = self._compute_difficulty(row)
            tier = self._assign_tier(score)

            row['difficulty_score'] = round(score, 4)
            row['difficulty_tier'] = tier

            # Service abstraction
            original_service = row.get('service', '').strip()
            if self.abstract_services and original_service in self._abstract_map:
                row['mapped_service'] = self._abstract_map[original_service]
            else:
                row['mapped_service'] = original_service

            # Exclusion flag
            if self.exclude_client_side and original_service in EXCLUDE_SERVICES:
                row['excluded'] = 'client_side'
            elif original_service and original_service not in self._pengym_services and \
                 original_service not in self._abstract_map:
                row['excluded'] = 'unmapped'
            else:
                row['excluded'] = ''

            # PenGym-compatible flag
            mapped = row['mapped_service']
            is_privesc = bool(row.get('process', '').strip())
            if is_privesc:
                row['pengym_compatible'] = 'privesc'
            elif mapped in self._pengym_services:
                row['pengym_compatible'] = 'yes'
            elif row['excluded']:
                row['pengym_compatible'] = 'no'
            else:
                row['pengym_compatible'] = 'no'

    def _compute_difficulty(self, row: Dict) -> float:
        """Compute composite difficulty score for a single CVE."""
        prob = float(row.get('prob', 0.6))
        f_prob = 1.0 - prob

        ac = row.get('Attack_Complexity', '').strip()
        pr = row.get('Privileges_Required', '').strip()
        ui = row.get('User_Interaction', '').strip()

        if ac:
            # Full CVSS v3 data available
            f_ac = AC_MAP.get(ac, 0.0)
            f_pr = PR_MAP.get(pr, 0.0)
            f_ui = UI_MAP.get(ui, 0.0)
            return (self.w_prob * f_prob +
                    self.w_ac * f_ac +
                    self.w_pr * f_pr +
                    self.w_ui * f_ui)
        else:
            # CVSS v2 only — scale by w_prob for consistency with composite formula.
            # Without this scaling, raw f_prob (e.g., 0.40 for prob=0.6) would put
            # CVSS v2 CVEs in much higher tiers than equivalent CVSS v3 CVEs.
            return self.w_prob * f_prob

    def _assign_tier(self, score: float) -> int:
        """Assign difficulty tier based on score thresholds."""
        for threshold, tier in self.tier_thresholds:
            if score < threshold:
                return tier
        return 4  # T4: Expert

    def save_graded_csv(self, output_path: str) -> str:
        """Save graded CVEs to CSV with additional columns."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        extra_fields = ['difficulty_score', 'difficulty_tier',
                        'mapped_service', 'excluded', 'pengym_compatible']
        out_fieldnames = self.fieldnames + extra_fields

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=out_fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

        return str(output_path)

    def get_pengym_compatible_cves(self, tier: Optional[int] = None,
                                    service: Optional[str] = None) -> List[Dict]:
        """Get CVEs that are PenGym-compatible, optionally filtered by tier/service."""
        result = []
        for row in self.rows:
            if row.get('pengym_compatible') not in ('yes', 'privesc'):
                continue
            if tier is not None and int(row.get('difficulty_tier', 0)) != tier:
                continue
            if service is not None:
                mapped = row.get('mapped_service', '')
                if mapped != service:
                    continue
            result.append(row)
        return result

    def get_privesc_cves(self, tier: Optional[int] = None) -> List[Dict]:
        """Get privilege escalation CVEs, optionally filtered by tier."""
        result = []
        for row in self.rows:
            if not row.get('process', '').strip():
                continue
            if tier is not None and int(row.get('difficulty_tier', 0)) != tier:
                continue
            result.append(row)
        return result

    def get_distribution_report(self) -> Dict:
        """Generate a comprehensive distribution report."""
        if not self.rows or 'difficulty_tier' not in self.rows[0]:
            return {'error': 'classify() must be called first'}

        report = {
            'total_cves': len(self.rows),
            'unique_cve_ids': len(set(r['CVE_ID'] for r in self.rows)),
        }

        # Tier distribution
        tier_counts = Counter(int(r['difficulty_tier']) for r in self.rows)
        report['tier_distribution'] = {
            f'T{t}': tier_counts.get(t, 0) for t in range(1, 5)
        }

        # Score statistics
        scores = [float(r['difficulty_score']) for r in self.rows]
        report['score_stats'] = {
            'min': round(min(scores), 4),
            'max': round(max(scores), 4),
            'mean': round(sum(scores) / len(scores), 4),
            'median': round(sorted(scores)[len(scores) // 2], 4),
        }

        # PenGym compatibility
        compat_counts = Counter(r.get('pengym_compatible', 'unknown') for r in self.rows)
        report['pengym_compatibility'] = dict(compat_counts)

        # Service distribution (mapped)
        svc_by_tier = defaultdict(lambda: Counter())
        for r in self.rows:
            svc = r.get('mapped_service', '') or '(privesc)'
            tier = int(r['difficulty_tier'])
            svc_by_tier[tier][svc] += 1
        report['service_by_tier'] = {
            f'T{t}': dict(svc_by_tier.get(t, {})) for t in range(1, 5)
        }

        # Exclusion stats
        excl_counts = Counter(r.get('excluded', '') for r in self.rows)
        report['excluded'] = {k: v for k, v in excl_counts.items() if k}

        return report

    def print_report(self) -> None:
        """Print a formatted distribution report."""
        report = self.get_distribution_report()
        if 'error' in report:
            print(f"Error: {report['error']}")
            return

        print("\n" + "=" * 60)
        print("CVE Difficulty Grading Report")
        print("=" * 60)

        print(f"\nTotal rows: {report['total_cves']}")
        print(f"Unique CVEs: {report['unique_cve_ids']}")

        print("\n--- Tier Distribution ---")
        for tier, count in sorted(report['tier_distribution'].items()):
            pct = count / report['total_cves'] * 100
            bar = '█' * int(pct / 2)
            print(f"  {tier}: {count:5d} ({pct:5.1f}%) {bar}")

        print(f"\n--- Score Statistics ---")
        stats = report['score_stats']
        print(f"  Min: {stats['min']:.4f}  Max: {stats['max']:.4f}")
        print(f"  Mean: {stats['mean']:.4f}  Median: {stats['median']:.4f}")

        print(f"\n--- PenGym Compatibility ---")
        for k, v in sorted(report['pengym_compatibility'].items()):
            print(f"  {k:12s}: {v:5d}")

        total_usable = sum(v for k, v in report['pengym_compatibility'].items()
                          if k in ('yes', 'privesc'))
        print(f"  {'TOTAL USABLE':12s}: {total_usable:5d} "
              f"({total_usable / report['total_cves'] * 100:.1f}%)")

        print(f"\n--- Excluded ---")
        for k, v in sorted(report.get('excluded', {}).items()):
            print(f"  {k:12s}: {v:5d}")

        print(f"\n--- Service × Tier (PenGym-compatible only) ---")
        for tier_key in ['T1', 'T2', 'T3', 'T4']:
            svcs = report['service_by_tier'].get(tier_key, {})
            pengym_svcs = {s: c for s, c in svcs.items()
                         if s in PENGYM_SERVICES or s == '(privesc)'}
            if pengym_svcs:
                items = ', '.join(f"{s}={c}" for s, c in
                                 sorted(pengym_svcs.items(), key=lambda x: -x[1]))
                print(f"  {tier_key}: {items}")

        print("=" * 60)


def run_classifier(input_csv: str, output_csv: str,
                   print_report: bool = True) -> Dict:
    """Convenience function to run the full classification pipeline.

    Args:
        input_csv: Path to CVE_dataset.csv
        output_csv: Path to write cve_graded.csv
        print_report: Whether to print the distribution report

    Returns:
        Distribution report dict
    """
    classifier = CVEClassifier()
    n = classifier.load_csv(input_csv)
    print(f"Loaded {n} CVE rows from {input_csv}")

    classifier.classify()
    out = classifier.save_graded_csv(output_csv)
    print(f"Saved graded CVEs to {out}")

    if print_report:
        classifier.print_report()

    return classifier.get_distribution_report()


if __name__ == '__main__':
    import sys
    project_root = Path(__file__).parent.parent.parent
    input_csv = project_root / 'data' / 'CVE' / 'CVE_dataset.csv'
    output_csv = project_root / 'data' / 'CVE' / 'cve_graded.csv'

    if len(sys.argv) > 1:
        input_csv = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_csv = Path(sys.argv[2])

    run_classifier(str(input_csv), str(output_csv))
