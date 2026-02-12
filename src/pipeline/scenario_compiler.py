"""
Scenario Compiler — Phase 2 of the CVE Difficulty & Expansion Pipeline.

Implements the Template + Overlay architecture described in
docs/cve_difficulty_and_expansion.md §4.

Components:
  1. ScenarioTemplate  — topology-only YAML with service/privesc slots
  2. CVEOverlay        — CVE assignments for each slot
  3. ScenarioCompiler  — merges template + overlay → full PenGym YAML
  4. generate_templates_from_existing() — converts existing YAMLs to templates
  5. CVESelector       — picks CVEs from cve_graded.csv for overlay slots
  6. OverlayGenerator  — creates overlay YAML files
"""

import os
import copy
import random
import yaml
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from collections import defaultdict
from dataclasses import dataclass, field


# ─── YAML Helper ─────────────────────────────────────────────────────────────
# NASim uses yaml.FullLoader which resolves unquoted `None` to Python None.
# We need the string "None" to survive the round-trip, so we quote it.

class _NaSimDumper(yaml.SafeDumper):
    """Custom YAML dumper that quotes the string 'None' so it survives
    yaml.FullLoader (which otherwise converts it to Python None)."""
    pass

def _none_str_representer(dumper, data):
    if data == 'None':
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

_NaSimDumper.add_representer(str, _none_str_representer)


def _nasim_yaml_dump(data, stream, **kwargs):
    """Dump YAML using our custom dumper that quotes 'None' strings."""
    kwargs.setdefault('default_flow_style', False)
    kwargs.setdefault('sort_keys', False)
    kwargs.setdefault('allow_unicode', True)
    return yaml.dump(data, stream, Dumper=_NaSimDumper, **kwargs)


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class ServiceSlot:
    """A slot in a template where a service + exploit can be placed."""
    slot_id: str
    host: Tuple[int, int]       # (subnet, host_idx)
    role: str                   # 'entry_point', 'pivot', 'target'
    allowed_services: List[str]

@dataclass
class PrivescSlot:
    """A slot for privilege escalation."""
    slot_id: str
    host: Tuple[int, int]
    allowed_processes: List[str]  # ['any'] means all

@dataclass
class SlotAssignment:
    """CVE assignment for a single slot."""
    service: Optional[str] = None
    process: Optional[str] = None
    exploit_name: str = ''
    cve_id: str = ''
    prob: float = 0.6
    cost: int = 3
    access: str = 'user'
    os: str = 'linux'


# ─── Template Converter ─────────────────────────────────────────────────────

def generate_template_from_yaml(yaml_path: str,
                                 template_name: Optional[str] = None) -> Dict:
    """Convert an existing PenGym scenario YAML to a template with slots.

    Extracts the topology, creates service/privesc slots from host_configurations,
    and strips out the specific exploit/privesc definitions.

    Args:
        yaml_path: Path to existing PenGym scenario YAML
        template_name: Name for the template (defaults to filename stem)

    Returns:
        Template dict ready to save as YAML
    """
    with open(yaml_path, 'r') as f:
        scenario = yaml.safe_load(f)

    if template_name is None:
        template_name = Path(yaml_path).stem

    # All known PenGym services
    all_services = list(scenario.get('services', []))
    all_processes = list(scenario.get('processes', []))

    template = {
        'meta': {
            'name': template_name,
            'description': f'Auto-generated from {Path(yaml_path).name}',
            'host_count': sum(scenario.get('subnets', [])),
            'subnet_count': len(scenario.get('subnets', [])),
        },
        'subnets': scenario['subnets'],
        'topology': scenario['topology'],
        'sensitive_hosts': scenario.get('sensitive_hosts', {}),
        'service_scan_cost': scenario.get('service_scan_cost', 1),
        'os_scan_cost': scenario.get('os_scan_cost', 1),
        'subnet_scan_cost': scenario.get('subnet_scan_cost', 1),
        'process_scan_cost': scenario.get('process_scan_cost', 1),
        'step_limit': scenario.get('step_limit', 1000),
    }

    # Create service and privesc slots from host_configurations
    service_slots = []
    privesc_slots = []
    host_configs = scenario.get('host_configurations', {})

    slot_idx = 0
    for host_key, config in sorted(host_configs.items(), key=str):
        # Parse host key — can be string "(1, 0)" or tuple
        if isinstance(host_key, str):
            parts = host_key.strip('()').split(',')
            host_addr = (int(parts[0].strip()), int(parts[1].strip()))
        else:
            host_addr = tuple(host_key)

        host_services = config.get('services', [])
        host_processes = config.get('processes', [])
        host_os = config.get('os', 'linux')

        # Determine role based on sensitive_hosts
        sensitive = scenario.get('sensitive_hosts', {})
        if isinstance(sensitive, dict):
            is_sensitive = str(host_addr) in str(sensitive) or host_addr in sensitive
        else:
            is_sensitive = False

        # Simple heuristic: first subnet hosts = entry_point, sensitive = target
        if host_addr[0] == 1:
            role = 'entry_point'
        elif is_sensitive:
            role = 'target'
        else:
            role = 'pivot'

        # Create service slots for each service on this host
        for svc in host_services:
            slot_idx += 1
            service_slots.append({
                'slot_id': f'S{slot_idx}',
                'host': list(host_addr),
                'role': role,
                'allowed_services': [svc],  # Keep original, overlay can change
                'default_service': svc,
                'default_os': host_os,
            })

        # Create privesc slots
        for proc in host_processes:
            slot_idx += 1
            privesc_slots.append({
                'slot_id': f'P{slot_idx}',
                'host': list(host_addr),
                'allowed_processes': [proc, 'any'],
                'default_process': proc,
                'default_os': host_os,
            })

    template['service_slots'] = service_slots
    template['privesc_slots'] = privesc_slots

    # Firewall rules — keep as-is
    template['firewall'] = scenario.get('firewall', {})

    # Store original host_configurations for reference
    template['_original_host_configurations'] = host_configs

    return template


def save_template(template: Dict, output_path: str) -> str:
    """Save template dict to YAML file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        _nasim_yaml_dump(template, f)
    return str(output_path)


# ─── CVE Selector ────────────────────────────────────────────────────────────

class CVESelector:
    """Selects CVEs from cve_graded.csv matching template slot constraints.

    Implements Module 2 from docs/cve_difficulty_and_expansion.md §5.3.
    """

    def __init__(self, graded_csv_path: str):
        """Load graded CVEs from CSV."""
        self.cves: List[Dict] = []
        with open(graded_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            self.cves = list(reader)

        # Build indexes
        self._by_tier: Dict[int, List[Dict]] = defaultdict(list)
        self._by_service: Dict[str, List[Dict]] = defaultdict(list)
        self._privesc: List[Dict] = []

        for cve in self.cves:
            if cve.get('pengym_compatible') not in ('yes', 'privesc'):
                continue
            tier = int(cve.get('difficulty_tier', 1))
            self._by_tier[tier].append(cve)

            mapped_svc = cve.get('mapped_service', '')
            if mapped_svc:
                self._by_service[mapped_svc].append(cve)

            if cve.get('process', '').strip():
                self._privesc.append(cve)

    def select_for_template(self, template: Dict, tier: int,
                            seed: Optional[int] = None,
                            allow_tier_relaxation: bool = True) -> Dict[str, Dict]:
        """Select CVEs for each slot in a template.

        Args:
            template: Parsed template dict
            tier: Target difficulty tier (1-4)
            seed: Random seed for reproducibility
            allow_tier_relaxation: If no CVEs found in target tier, try ±1

        Returns:
            Dict mapping slot_id → selected CVE row
        """
        rng = random.Random(seed)
        assignments = {}
        used_cves: Set[str] = set()

        # Assign service slots
        for slot in template.get('service_slots', []):
            slot_id = slot['slot_id']
            allowed = slot.get('allowed_services', [])
            default_svc = slot.get('default_service', '')

            cve = self._find_cve_for_service(
                allowed + [default_svc], tier, used_cves, rng,
                allow_tier_relaxation)

            if cve:
                assignments[slot_id] = cve
                used_cves.add(cve['CVE_ID'])
            else:
                # Fallback: use default service with generic exploit params
                assignments[slot_id] = self._make_fallback(
                    default_svc, tier, slot.get('default_os', 'linux'))

        # Assign privesc slots
        for slot in template.get('privesc_slots', []):
            slot_id = slot['slot_id']
            cve = self._find_privesc(tier, used_cves, rng,
                                     allow_tier_relaxation)
            if cve:
                assignments[slot_id] = cve
                used_cves.add(cve['CVE_ID'])
            else:
                # Fallback privesc
                assignments[slot_id] = self._make_fallback_privesc(
                    slot.get('default_process', 'tomcat'),
                    slot.get('default_os', 'linux'))

        return assignments

    def _find_cve_for_service(self, services: List[str], tier: int,
                               used: Set[str], rng: random.Random,
                               allow_relax: bool) -> Optional[Dict]:
        """Find a CVE matching any of the given services at the target tier."""
        services = [s for s in services if s]  # Remove empty strings

        for try_tier in self._tier_search_order(tier, allow_relax):
            candidates = []
            for svc in services:
                for cve in self._by_service.get(svc, []):
                    if (int(cve.get('difficulty_tier', 0)) == try_tier and
                            cve['CVE_ID'] not in used):
                        candidates.append(cve)

            if candidates:
                return rng.choice(candidates)

        return None

    def _find_privesc(self, tier: int, used: Set[str],
                       rng: random.Random,
                       allow_relax: bool) -> Optional[Dict]:
        """Find a privesc CVE at the target tier."""
        for try_tier in self._tier_search_order(tier, allow_relax):
            candidates = [c for c in self._privesc
                         if int(c.get('difficulty_tier', 0)) == try_tier
                         and c['CVE_ID'] not in used]
            if candidates:
                return rng.choice(candidates)
        return None

    def _tier_search_order(self, tier: int, allow_relax: bool) -> List[int]:
        """Generate tier search order: target first, then ±1, ±2."""
        order = [tier]
        if allow_relax:
            for delta in [1, -1, 2, -2]:
                t = tier + delta
                if 1 <= t <= 4 and t not in order:
                    order.append(t)
        return order

    def _make_fallback(self, service: str, tier: int, os_name: str) -> Dict:
        """Create a fallback CVE entry when no real CVE matches."""
        # Use tier-appropriate prob
        tier_prob = {1: 0.99, 2: 0.8, 3: 0.6, 4: 0.4}
        tier_cost = {1: 1, 2: 2, 3: 3, 4: 4}
        return {
            'CVE_ID': f'FALLBACK-{service.upper()}',
            'service': service,
            'mapped_service': service,
            'process': '',
            'prob': str(tier_prob.get(tier, 0.6)),
            'cost': str(tier_cost.get(tier, 3)),
            'access': 'user',
            'os': os_name,
            'difficulty_tier': str(tier),
            'difficulty_score': '0.0',
            '_is_fallback': True,
        }

    def _make_fallback_privesc(self, process: str, os_name: str) -> Dict:
        """Create fallback privesc entry."""
        return {
            'CVE_ID': f'FALLBACK-PE-{process.upper()}',
            'service': '',
            'mapped_service': '',
            'process': process,
            'prob': '0.8',
            'cost': '2',
            'access': 'root',
            'os': os_name,
            'difficulty_tier': '1',
            'difficulty_score': '0.0',
            '_is_fallback': True,
        }

    def get_tier_stats(self) -> Dict:
        """Return counts per tier and service."""
        stats = {}
        for tier in range(1, 5):
            tier_cves = self._by_tier[tier]
            svc_counts = defaultdict(int)
            for c in tier_cves:
                svc = c.get('mapped_service', '') or '(privesc)'
                svc_counts[svc] += 1
            stats[f'T{tier}'] = {
                'total': len(tier_cves),
                'services': dict(svc_counts),
            }
        stats['total_privesc'] = len(self._privesc)
        return stats


# ─── Overlay Generator ───────────────────────────────────────────────────────

class OverlayGenerator:
    """Generates overlay YAML files from CVE selections.

    Implements Module 3 from docs/cve_difficulty_and_expansion.md §5.4.
    """

    @staticmethod
    def _sanitize_access(access_val: str) -> str:
        """Normalize access value to NASim-compatible 'user' or 'root'."""
        access_val = str(access_val).strip().lower()
        if access_val in ('root', 'admin', 'administrator', 'system'):
            return 'root'
        return 'user'

    @staticmethod
    def generate(assignments: Dict[str, Dict], overlay_id: str,
                 tier: int) -> Dict:
        """Generate an overlay dict from slot assignments.

        Args:
            assignments: slot_id → CVE row dict
            overlay_id: Unique overlay identifier
            tier: Difficulty tier

        Returns:
            Overlay dict ready to save as YAML
        """
        scores = []
        cve_list = []
        slot_assignments = {}
        os_assignments = {}

        for slot_id, cve in sorted(assignments.items()):
            score = float(cve.get('difficulty_score', 0))
            scores.append(score)
            cve_list.append(cve.get('CVE_ID', 'unknown'))

            service = cve.get('mapped_service', cve.get('service', ''))
            process = cve.get('process', '')
            os_name = cve.get('os', 'linux')

            if process and not service:
                # Privesc slot — use full process name for consistency
                pe_name = f"pe_{process}"
                slot_assignments[slot_id] = {
                    'process': process,
                    'privesc': {
                        'name': pe_name,
                        'cve_id': cve['CVE_ID'],
                        'prob': float(cve.get('prob', 0.8)),
                        'cost': int(cve.get('cost', 2)),
                        'access': OverlayGenerator._sanitize_access(
                            cve.get('access', 'root')),
                    }
                }
            else:
                # Service exploit slot
                exp_name = f"e_{service}"
                slot_assignments[slot_id] = {
                    'service': service,
                    'exploit': {
                        'name': exp_name,
                        'cve_id': cve['CVE_ID'],
                        'prob': float(cve.get('prob', 0.6)),
                        'cost': int(cve.get('cost', 3)),
                        'access': OverlayGenerator._sanitize_access(
                            cve.get('access', 'user')),
                    }
                }

            os_assignments[slot_id] = os_name

        overlay = {
            'meta': {
                'overlay_id': overlay_id,
                'difficulty_tier': tier,
                'avg_difficulty_score': round(
                    sum(scores) / max(len(scores), 1), 4),
                'cve_list': cve_list,
            },
            'slot_assignments': slot_assignments,
            'os_assignments': os_assignments,
        }
        return overlay

    @staticmethod
    def save(overlay: Dict, output_path: str) -> str:
        """Save overlay dict to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            _nasim_yaml_dump(overlay, f)
        return str(output_path)


# ─── Scenario Compiler ──────────────────────────────────────────────────────

class ScenarioCompiler:
    """Merges a template + overlay → full PenGym-compatible scenario YAML.

    Implements Module 4 from docs/cve_difficulty_and_expansion.md §5.5.
    """

    @staticmethod
    def _sanitize_os(os_val: str) -> str:
        """Normalize OS to NASim-compatible value."""
        os_val = str(os_val).strip().lower()
        if os_val in ('windows', 'win', 'win32', 'win64'):
            return 'windows'
        if os_val in ('none', '', 'any', 'null'):
            return 'linux'  # default fallback
        return os_val  # linux, freebsd, etc.

    @staticmethod
    def compile(template: Dict, overlay: Dict,
                standard_lists: Optional[Dict] = None) -> Dict:
        """Compile template + overlay into a full PenGym scenario.

        Args:
            template: Template dict with topology + slots
            overlay: Overlay dict with CVE assignments
            standard_lists: Optional dict with 'os', 'services', 'processes'
                to standardize dimensions across scenarios from the same template.
                If provided, all scenarios will use these lists regardless of
                which CVEs are assigned. This ensures consistent obs/action dims.

        Returns:
            Full PenGym scenario dict (compatible with NASim load_scenario)
        """
        slot_assignments = overlay.get('slot_assignments', {})
        os_assignments = overlay.get('os_assignments', {})

        # Collect all services, processes, exploits, privesc from overlay
        services_set = set()
        processes_set = set()
        exploits = {}
        privilege_escalation = {}

        for slot_id, assignment in slot_assignments.items():
            if 'service' in assignment and assignment['service']:
                svc = assignment['service']
                services_set.add(svc)
                exp = assignment['exploit']
                exp_name = exp['name']
                if exp_name not in exploits:
                    exploits[exp_name] = {
                        'service': svc,
                        'os': 'None',  # NASim expects str; converts "None"→null
                        'prob': exp['prob'],
                        'cost': exp['cost'],
                        'access': exp['access'],
                    }

            if 'process' in assignment and assignment['process']:
                proc = assignment['process']
                processes_set.add(proc)
                pe = assignment['privesc']
                pe_name = pe['name']
                if pe_name not in privilege_escalation:
                    privilege_escalation[pe_name] = {
                        'process': proc,
                        'os': 'None',  # NASim expects str; converts "None"→null
                        'prob': pe['prob'],
                        'cost': pe['cost'],
                        'access': pe['access'],
                    }

        # Build host_configurations by mapping slots back to hosts
        host_configs = {}
        slot_to_host = {}

        for slot in template.get('service_slots', []):
            host = tuple(slot['host'])
            slot_to_host[slot['slot_id']] = host

        for slot in template.get('privesc_slots', []):
            host = tuple(slot['host'])
            slot_to_host[slot['slot_id']] = host

        # Group assignments by host
        host_services = defaultdict(list)
        host_processes = defaultdict(list)
        host_os = {}

        for slot_id, assignment in slot_assignments.items():
            host = slot_to_host.get(slot_id)
            if host is None:
                continue

            os_name = ScenarioCompiler._sanitize_os(
                os_assignments.get(slot_id, 'linux'))
            host_os[host] = os_name

            if 'service' in assignment and assignment['service']:
                svc = assignment['service']
                if svc not in host_services[host]:
                    host_services[host].append(svc)

            if 'process' in assignment and assignment['process']:
                proc = assignment['process']
                if proc not in host_processes[host]:
                    host_processes[host].append(proc)

        # Build host_configurations
        original_configs = template.get('_original_host_configurations', {})
        for host_key, original in original_configs.items():
            if isinstance(host_key, str):
                parts = host_key.strip('()').split(',')
                host = (int(parts[0].strip()), int(parts[1].strip()))
            else:
                host = tuple(host_key)

            config = {
                'os': ScenarioCompiler._sanitize_os(
                    host_os.get(host, original.get('os', 'linux'))),
                'services': host_services.get(host, original.get('services', [])),
                'processes': host_processes.get(host, original.get('processes', [])),
            }

            # Preserve original firewall if present
            if 'firewall' in original:
                config['firewall'] = original['firewall']

            host_configs[host_key] = config

        # Assemble the full scenario
        # If standard_lists provided, use canonical sets to ensure
        # consistent obs/action dimensions across all overlays
        if standard_lists:
            final_os = standard_lists.get('os', sorted(set(host_os.values())) or ['linux'])
            final_services = standard_lists.get('services', sorted(services_set))
            final_processes = standard_lists.get('processes', sorted(processes_set))
        else:
            final_os = sorted(set(host_os.values())) or ['linux']
            final_services = sorted(services_set)
            final_processes = sorted(processes_set)

        # Ensure all used services/processes are in the final lists
        for svc in services_set:
            if svc not in final_services:
                final_services.append(svc)
        for proc in processes_set:
            if proc not in final_processes:
                final_processes.append(proc)
        final_services = sorted(set(final_services))
        final_processes = sorted(set(final_processes))

        # Add dummy exploits/privesc for standard services not in overlay
        for svc in final_services:
            exp_name = f"e_{svc}"
            if exp_name not in exploits:
                exploits[exp_name] = {
                    'service': svc,
                    'os': 'None',
                    'prob': 0.0,  # Impossible exploit (placeholder)
                    'cost': 100,
                    'access': 'user',
                }
        for proc in final_processes:
            pe_name = f"pe_{proc}"
            if pe_name not in privilege_escalation:
                privilege_escalation[pe_name] = {
                    'process': proc,
                    'os': 'None',
                    'prob': 0.0,
                    'cost': 100,
                    'access': 'root',
                }

        scenario = {
            'subnets': template['subnets'],
            'topology': template['topology'],
            'sensitive_hosts': template['sensitive_hosts'],
            'os': final_os,
            'services': final_services,
            'processes': final_processes,
            'exploits': exploits,
            'privilege_escalation': privilege_escalation,
            'service_scan_cost': template.get('service_scan_cost', 1),
            'os_scan_cost': template.get('os_scan_cost', 1),
            'subnet_scan_cost': template.get('subnet_scan_cost', 1),
            'process_scan_cost': template.get('process_scan_cost', 1),
            'host_configurations': host_configs,
            'firewall': template.get('firewall', {}),
            'step_limit': template.get('step_limit', 1000),
        }

        return scenario

    @staticmethod
    def save_scenario(scenario: Dict, output_path: str) -> str:
        """Save compiled scenario to YAML."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            _nasim_yaml_dump(scenario, f)
        return str(output_path)

    @staticmethod
    def validate_scenario(scenario: Dict) -> List[str]:
        """Validate a compiled scenario for common errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Required keys
        required = ['subnets', 'topology', 'sensitive_hosts', 'services',
                     'exploits', 'host_configurations']
        for key in required:
            if key not in scenario:
                errors.append(f"Missing required key: {key}")

        # Check host configs reference valid services/processes
        for host, config in scenario.get('host_configurations', {}).items():
            for svc in config.get('services', []):
                if svc not in scenario.get('services', []):
                    errors.append(f"Host {host}: service '{svc}' not in services list")
            for proc in config.get('processes', []):
                if proc not in scenario.get('processes', []):
                    errors.append(f"Host {host}: process '{proc}' not in processes list")

        # Check exploits reference valid services
        for exp_name, exp in scenario.get('exploits', {}).items():
            if exp.get('service') not in scenario.get('services', []):
                errors.append(f"Exploit {exp_name}: service '{exp.get('service')}'"
                             f" not in services list")

        # Check privesc reference valid processes
        for pe_name, pe in scenario.get('privilege_escalation', {}).items():
            if pe.get('process') not in scenario.get('processes', []):
                errors.append(f"Privesc {pe_name}: process '{pe.get('process')}'"
                             f" not in processes list")

        # Check subnet count matches topology
        n_subnets = len(scenario.get('subnets', []))
        topology = scenario.get('topology', [])
        if topology and len(topology) != n_subnets + 1:
            errors.append(f"Topology size ({len(topology)}) != subnets+1 "
                         f"({n_subnets + 1})")

        return errors


# ─── Batch Pipeline ──────────────────────────────────────────────────────────

class ScenarioPipeline:
    """End-to-end pipeline: template + graded CVEs → batch of PenGym scenarios.

    Usage:
        pipeline = ScenarioPipeline(
            graded_csv='data/CVE/cve_graded.csv',
            template_dir='data/scenarios/templates',
            output_dir='data/scenarios/generated',
        )
        pipeline.generate_templates_from_existing(['tiny.yml', 'medium.yml'])
        pipeline.generate_overlays(n_per_tier=10)
        pipeline.compile_all()
    """

    def __init__(self, graded_csv: str, template_dir: str, output_dir: str,
                 scenario_dir: Optional[str] = None):
        self.graded_csv = graded_csv
        self.template_dir = Path(template_dir)
        self.output_dir = Path(output_dir)
        self.scenario_dir = Path(scenario_dir) if scenario_dir else self.template_dir.parent

        self.template_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.selector = CVESelector(graded_csv)
        self.compiler = ScenarioCompiler()

        self.templates: Dict[str, Dict] = {}
        self.overlays: Dict[str, Dict] = {}

    def generate_templates_from_existing(self, yaml_files: List[str]) -> int:
        """Convert existing scenario YAMLs to templates.

        Args:
            yaml_files: List of YAML filenames in scenario_dir

        Returns:
            Number of templates created
        """
        count = 0
        for fname in yaml_files:
            yaml_path = self.scenario_dir / fname
            if not yaml_path.exists():
                print(f"  WARNING: {yaml_path} not found, skipping")
                continue

            name = Path(fname).stem
            template = generate_template_from_yaml(str(yaml_path), name)
            template_path = self.template_dir / f"{name}.template.yml"
            save_template(template, str(template_path))
            self.templates[name] = template
            count += 1
            n_svc = len(template.get('service_slots', []))
            n_pe = len(template.get('privesc_slots', []))
            print(f"  Template '{name}': {n_svc} service slots, "
                  f"{n_pe} privesc slots → {template_path}")

        return count

    def generate_overlays(self, n_per_tier: int = 10,
                          template_name: Optional[str] = None,
                          seed_base: int = 42) -> int:
        """Generate overlay files for each tier × template combination.

        Args:
            n_per_tier: Number of overlays per (template, tier) combination
            template_name: If specified, only generate for this template
            seed_base: Base random seed

        Returns:
            Total number of overlays generated
        """
        overlay_dir = self.output_dir / 'overlays'
        count = 0

        templates = self.templates
        if template_name:
            templates = {template_name: templates[template_name]}

        for tname, template in templates.items():
            for tier in range(1, 5):
                tier_dir = overlay_dir / f'tier{tier}'
                tier_dir.mkdir(parents=True, exist_ok=True)

                for i in range(n_per_tier):
                    seed = seed_base + tier * 1000 + i
                    assignments = self.selector.select_for_template(
                        template, tier, seed=seed)

                    overlay_id = f"{tname}_T{tier}_{i:03d}"
                    overlay = OverlayGenerator.generate(
                        assignments, overlay_id, tier)

                    overlay_path = tier_dir / f"{overlay_id}.overlay.yml"
                    OverlayGenerator.save(overlay, str(overlay_path))
                    self.overlays[overlay_id] = overlay
                    count += 1

        print(f"  Generated {count} overlays across 4 tiers")
        return count

    def compile_all(self, standardize_dims: bool = True) -> Tuple[int, List[str]]:
        """Compile all template × overlay combinations.

        Args:
            standardize_dims: If True, compute a canonical service/process/OS
                list for each template to ensure all overlays produce the same
                obs/action dimensions. Required for RL training consistency.

        Returns:
            (number_compiled, list_of_error_messages)
        """
        gen_dir = self.output_dir / 'compiled'
        gen_dir.mkdir(parents=True, exist_ok=True)

        compiled = 0
        all_errors = []

        # Build standard lists per template by collecting all services/processes
        # from all overlays for that template
        standard_lists_per_template = {}
        if standardize_dims:
            for overlay_id, overlay in self.overlays.items():
                parts = overlay_id.split('_T')
                tname = parts[0] if parts else list(self.templates.keys())[0]

                if tname not in standard_lists_per_template:
                    standard_lists_per_template[tname] = {
                        'services': set(), 'processes': set(), 'os': set()
                    }

                for slot_id, assignment in overlay.get('slot_assignments', {}).items():
                    if 'service' in assignment and assignment['service']:
                        standard_lists_per_template[tname]['services'].add(
                            assignment['service'])
                    if 'process' in assignment and assignment['process']:
                        standard_lists_per_template[tname]['processes'].add(
                            assignment['process'])

                for slot_id, os_val in overlay.get('os_assignments', {}).items():
                    standard_lists_per_template[tname]['os'].add(
                        ScenarioCompiler._sanitize_os(os_val))

            # Convert sets to sorted lists
            for tname, lists in standard_lists_per_template.items():
                standard_lists_per_template[tname] = {
                    'services': sorted(lists['services']),
                    'processes': sorted(lists['processes']),
                    'os': sorted(lists['os']) or ['linux'],
                }

        for overlay_id, overlay in self.overlays.items():
            # Extract template name from overlay_id
            parts = overlay_id.split('_T')
            tname = parts[0] if parts else list(self.templates.keys())[0]

            if tname not in self.templates:
                all_errors.append(f"Template '{tname}' not found for overlay {overlay_id}")
                continue

            template = self.templates[tname]
            std_lists = standard_lists_per_template.get(tname) if standardize_dims else None
            scenario = self.compiler.compile(template, overlay, std_lists)

            # Validate
            errors = self.compiler.validate_scenario(scenario)
            if errors:
                for e in errors:
                    all_errors.append(f"[{overlay_id}] {e}")
                # Still save but note it's invalid
                scenario['_validation_errors'] = errors

            output_path = gen_dir / f"{overlay_id}.yml"
            self.compiler.save_scenario(scenario, str(output_path))
            compiled += 1

        valid = compiled - len([e for e in all_errors if '[' in e])
        print(f"  Compiled {compiled} scenarios ({valid} valid, "
              f"{len(all_errors)} errors)")

        return compiled, all_errors

    def get_scenario_path(self, template_name: str, tier: int,
                          index: int = 0) -> Optional[str]:
        """Get path to a compiled scenario by template, tier, and index."""
        overlay_id = f"{template_name}_T{tier}_{index:03d}"
        path = self.output_dir / 'compiled' / f"{overlay_id}.yml"
        if path.exists():
            return str(path)
        return None

    def list_scenarios_by_tier(self, tier: int) -> List[str]:
        """List all compiled scenario paths for a given tier."""
        compiled_dir = self.output_dir / 'compiled'
        if not compiled_dir.exists():
            return []
        pattern = f"*_T{tier}_*.yml"
        return sorted(str(p) for p in compiled_dir.glob(pattern))


if __name__ == '__main__':
    import sys

    project_root = Path(__file__).parent.parent.parent
    graded_csv = project_root / 'data' / 'CVE' / 'cve_graded.csv'
    scenario_dir = project_root / 'data' / 'scenarios'
    template_dir = scenario_dir / 'templates'
    output_dir = scenario_dir / 'generated'

    print("=" * 60)
    print("Scenario Pipeline — Template + Overlay Generation")
    print("=" * 60)

    pipeline = ScenarioPipeline(
        graded_csv=str(graded_csv),
        template_dir=str(template_dir),
        output_dir=str(output_dir),
        scenario_dir=str(scenario_dir),
    )

    # Step 1: Convert existing scenarios to templates
    print("\n[1/3] Generating templates from existing scenarios...")
    yaml_files = ['tiny.yml', 'tiny-small.yml', 'small-linear.yml', 'medium.yml']
    n_templates = pipeline.generate_templates_from_existing(yaml_files)
    print(f"  Created {n_templates} templates")

    # Step 2: Generate overlays
    print("\n[2/3] Generating overlays...")
    n_overlays = pipeline.generate_overlays(n_per_tier=5, seed_base=42)
    print(f"  Created {n_overlays} overlays")

    # Step 3: Compile
    print("\n[3/3] Compiling scenarios...")
    n_compiled, errors = pipeline.compile_all()
    if errors:
        print(f"\n  Validation errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Pipeline complete:")
    print(f"  Templates:  {n_templates}")
    print(f"  Overlays:   {n_overlays}")
    print(f"  Compiled:   {n_compiled}")
    print(f"  Errors:     {len(errors)}")

    # Show tier stats
    for tier in range(1, 5):
        scenarios = pipeline.list_scenarios_by_tier(tier)
        print(f"  T{tier} scenarios: {len(scenarios)}")

    print(f"{'=' * 60}")
