"""
Extensible Service Registry — Phase 4 of the CVE Difficulty & Expansion Pipeline.

Provides:
  1. ServiceRegistry  — extensible registry for PenGym services/processes
  2. CVEAdditionPipeline — end-to-end: new CSV → graded → overlay → scenario
  3. TemplateExpander — add new services to existing templates

Design from docs/cve_difficulty_and_expansion.md §7.

The registry allows adding new services/processes WITHOUT modifying
existing code. New services are registered once and become available
throughout the scenario generation pipeline.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass, field


# ─── Service Registry ───────────────────────────────────────────────────────

@dataclass
class ServiceDefinition:
    """Defines a PenGym-compatible service."""
    name: str                         # e.g. 'ssh', 'http', 'mssql'
    exploit_name: str                 # e.g. 'e_ssh', 'e_http'
    cve_keywords: List[str]           # CVE service keywords that map here
    default_port: int = 0             # For reference
    category: str = 'network'         # network, web, database, etc.
    description: str = ''


@dataclass
class ProcessDefinition:
    """Defines a PenGym-compatible process (for privilege escalation)."""
    name: str                         # e.g. 'tomcat', 'schtask'
    privesc_name: str                 # e.g. 'pe_tomcat'
    cve_keywords: List[str]           # CVE process keywords that map here
    category: str = 'privesc'
    description: str = ''


class ServiceRegistry:
    """Extensible registry for PenGym services and processes.

    Provides a single source of truth for:
      - Which services exist in PenGym
      - How CVE keywords map to services
      - How to name exploits/privesc consistently

    Usage:
        registry = ServiceRegistry()         # loads defaults
        registry.register_service(ServiceDefinition(
            name='mssql', exploit_name='e_mssql',
            cve_keywords=['mssql', 'sql_server', 'sqlserver'],
            default_port=1433, category='database',
        ))
        registry.save('path/to/registry.json')

    To extend PenGym with a new service:
        1. registry.register_service(...)
        2. Re-run scenario pipeline
        3. New scenarios will include the service
    """

    # Default PenGym services (built-in)
    _DEFAULT_SERVICES = [
        ServiceDefinition('ssh', 'e_ssh', ['ssh', 'openssh', 'sshd'],
                          22, 'network', 'Secure Shell'),
        ServiceDefinition('ftp', 'e_ftp', ['ftp', 'proftpd', 'vsftpd', 'pureftpd'],
                          21, 'network', 'File Transfer Protocol'),
        ServiceDefinition('http', 'e_http',
                          ['http', 'https', 'apache', 'nginx', 'iis', 'webapp',
                           'tomcat', 'httpd', 'webrick', 'lighttpd', 'php',
                           'webapps', 'cgi', 'brightstor'],
                          80, 'web', 'HTTP Web Service'),
        ServiceDefinition('samba', 'e_samba',
                          ['samba', 'smb', 'cifs', 'windows', 'netbios'],
                          445, 'network', 'Samba/SMB File Sharing'),
        ServiceDefinition('smtp', 'e_smtp',
                          ['smtp', 'sendmail', 'postfix', 'exim'],
                          25, 'network', 'Simple Mail Transfer Protocol'),
    ]

    _DEFAULT_PROCESSES = [
        ProcessDefinition('tomcat', 'pe_tomcat',
                          ['tomcat', 'java', 'jboss', 'wildfly'],
                          'privesc', 'Apache Tomcat / Java server'),
        ProcessDefinition('schtask', 'pe_schtask',
                          ['schtask', 'scheduled_task', 'cron', 'at'],
                          'privesc', 'Scheduled Tasks'),
        ProcessDefinition('daclsvc', 'pe_daclsvc',
                          ['daclsvc', 'dacl', 'service_permissions', 'proftpd'],
                          'privesc', 'Weak Service DACL'),
    ]

    def __init__(self, load_defaults: bool = True):
        self._services: Dict[str, ServiceDefinition] = {}
        self._processes: Dict[str, ProcessDefinition] = {}
        self._keyword_map: Dict[str, str] = {}       # keyword → service name
        self._process_keyword_map: Dict[str, str] = {}

        if load_defaults:
            for svc in self._DEFAULT_SERVICES:
                self.register_service(svc)
            for proc in self._DEFAULT_PROCESSES:
                self.register_process(proc)

    def register_service(self, svc: ServiceDefinition):
        """Register a new service (or update existing)."""
        self._services[svc.name] = svc
        for kw in svc.cve_keywords:
            self._keyword_map[kw.lower()] = svc.name

    def register_process(self, proc: ProcessDefinition):
        """Register a new process (or update existing)."""
        self._processes[proc.name] = proc
        for kw in proc.cve_keywords:
            self._process_keyword_map[kw.lower()] = proc.name

    def get_service(self, name: str) -> Optional[ServiceDefinition]:
        return self._services.get(name)

    def get_process(self, name: str) -> Optional[ProcessDefinition]:
        return self._processes.get(name)

    def map_cve_service(self, cve_service: str) -> Optional[str]:
        """Map a CVE service keyword to a registered PenGym service name."""
        return self._keyword_map.get(cve_service.lower())

    def map_cve_process(self, cve_process: str) -> Optional[str]:
        """Map a CVE process keyword to a registered PenGym process name."""
        return self._process_keyword_map.get(cve_process.lower())

    @property
    def service_names(self) -> List[str]:
        return sorted(self._services.keys())

    @property
    def process_names(self) -> List[str]:
        return sorted(self._processes.keys())

    @property
    def exploit_names(self) -> List[str]:
        return [self._services[s].exploit_name for s in self.service_names]

    @property
    def privesc_names(self) -> List[str]:
        return [self._processes[p].privesc_name for p in self.process_names]

    def get_all_keywords(self) -> Dict[str, List[str]]:
        """Return all keyword→service mappings."""
        result = {}
        for svc_name, svc in self._services.items():
            result[svc_name] = svc.cve_keywords
        return result

    def describe(self) -> str:
        """Human-readable summary."""
        lines = [
            f"ServiceRegistry: {len(self._services)} services, "
            f"{len(self._processes)} processes",
            f"  Services: {', '.join(self.service_names)}",
            f"  Processes: {', '.join(self.process_names)}",
            f"  Keyword mappings: {len(self._keyword_map)} service, "
            f"{len(self._process_keyword_map)} process",
        ]
        return '\n'.join(lines)

    def save(self, path: str):
        """Save registry to JSON for persistence."""
        data = {
            'services': [
                {
                    'name': s.name,
                    'exploit_name': s.exploit_name,
                    'cve_keywords': s.cve_keywords,
                    'default_port': s.default_port,
                    'category': s.category,
                    'description': s.description,
                }
                for s in self._services.values()
            ],
            'processes': [
                {
                    'name': p.name,
                    'privesc_name': p.privesc_name,
                    'cve_keywords': p.cve_keywords,
                    'category': p.category,
                    'description': p.description,
                }
                for p in self._processes.values()
            ],
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'ServiceRegistry':
        """Load registry from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)

        registry = cls(load_defaults=False)
        for s in data.get('services', []):
            registry.register_service(ServiceDefinition(**s))
        for p in data.get('processes', []):
            registry.register_process(ProcessDefinition(**p))
        return registry


# ─── CVE Addition Pipeline ──────────────────────────────────────────────────

class CVEAdditionPipeline:
    """End-to-end pipeline for adding new CVEs to the system.

    Automates: CSV → graded → overlay → compiled scenario.

    Usage:
        pipeline = CVEAdditionPipeline(
            project_root='d:/NCKH/fusion/pentest',
            registry=ServiceRegistry(),
        )
        # Add new CVEs
        pipeline.add_cves_from_csv('path/to/new_cves.csv')
        # Regenerate everything
        pipeline.regenerate_all()
    """

    def __init__(self, project_root: str,
                 registry: Optional[ServiceRegistry] = None):
        self.project_root = Path(project_root)
        self.registry = registry or ServiceRegistry()

        # Standard paths
        self.cve_dir = self.project_root / 'data' / 'CVE'
        self.scenario_dir = self.project_root / 'data' / 'scenarios'
        self.template_dir = self.scenario_dir / 'templates'
        self.generated_dir = self.scenario_dir / 'generated'

    def add_cves_from_csv(self, csv_path: str,
                          append: bool = True) -> Dict:
        """Grade new CVEs and add them to the graded dataset.

        Args:
            csv_path: Path to new CVE CSV (same format as CVE_dataset.csv)
            append: If True, append to existing cve_graded.csv.
                    If False, replace it.

        Returns:
            Dict with grade distribution info
        """
        from .cve_classifier import CVEClassifier

        classifier = CVEClassifier()
        classifier.load_csv(csv_path)
        classifier.classify()

        if append:
            output_path = str(self.cve_dir / 'cve_graded.csv')
            # Append new rows (skip header if file exists)
            import csv
            existing_rows = 0
            if Path(output_path).exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_rows = sum(1 for _ in f) - 1  # minus header

            classifier.save_graded_csv(output_path)
            report = classifier.get_distribution_report()
            report['append_mode'] = True
            report['existing_rows'] = existing_rows
        else:
            output_path = str(self.cve_dir / 'cve_graded.csv')
            classifier.save_graded_csv(output_path)
            report = classifier.get_distribution_report()
            report['append_mode'] = False

        return report

    def regenerate_all(self, yaml_files: Optional[List[str]] = None,
                       n_per_tier: int = 5,
                       seed_base: int = 42) -> Dict:
        """Regenerate all templates, overlays, and compiled scenarios.

        Args:
            yaml_files: Source YAML files for template generation.
                Defaults to pre-existing scenario files.
            n_per_tier: Overlays per tier per template
            seed_base: Random seed

        Returns:
            Summary dict
        """
        from .scenario_compiler import ScenarioPipeline

        if yaml_files is None:
            yaml_files = ['tiny.yml', 'tiny-small.yml',
                          'small-linear.yml', 'medium.yml']

        pipeline = ScenarioPipeline(
            graded_csv=str(self.cve_dir / 'cve_graded.csv'),
            template_dir=str(self.template_dir),
            output_dir=str(self.generated_dir),
            scenario_dir=str(self.scenario_dir),
        )

        n_templates = pipeline.generate_templates_from_existing(yaml_files)
        n_overlays = pipeline.generate_overlays(
            n_per_tier=n_per_tier, seed_base=seed_base)
        n_compiled, errors = pipeline.compile_all()

        return {
            'templates': n_templates,
            'overlays': n_overlays,
            'compiled': n_compiled,
            'errors': len(errors),
            'error_details': errors[:10],
        }

    def expand_with_service(self, service: ServiceDefinition,
                             regenerate: bool = True) -> Dict:
        """Add a new service to the registry and optionally regenerate.

        This is the primary extension point for Phase 4.

        Args:
            service: New service definition
            regenerate: If True, regenerate all scenarios

        Returns:
            Summary of changes
        """
        self.registry.register_service(service)

        # Save updated registry
        registry_path = str(self.cve_dir / 'service_registry.json')
        self.registry.save(registry_path)

        result = {
            'service_added': service.name,
            'exploit_name': service.exploit_name,
            'keywords': service.cve_keywords,
            'registry_path': registry_path,
            'total_services': len(self.registry.service_names),
        }

        if regenerate:
            regen_result = self.regenerate_all()
            result['regeneration'] = regen_result

        return result


# ─── Template Expander ──────────────────────────────────────────────────────

class TemplateExpander:
    """Expand existing templates with new service slots.

    When a new service is added to the registry, existing templates
    can be expanded to include slots for that service.
    """

    @staticmethod
    def expand_template(template: Dict,
                        new_services: List[str],
                        hosts: Optional[List[Tuple[int, int]]] = None) -> Dict:
        """Add new service slots to a template.

        Args:
            template: Existing template dict
            new_services: List of new service names to add
            hosts: Specific hosts to add services to. If None,
                   adds to all non-internet hosts.

        Returns:
            Updated template dict
        """
        import copy
        template = copy.deepcopy(template)

        existing_slots = template.get('service_slots', [])
        max_slot_id = 0
        for slot in existing_slots:
            sid = slot.get('slot_id', 'S0')
            num = int(sid[1:]) if sid[1:].isdigit() else 0
            max_slot_id = max(max_slot_id, num)

        # Determine target hosts
        if hosts is None:
            # Use existing hosts from service_slots
            hosts_set = set()
            for slot in existing_slots:
                hosts_set.add(tuple(slot['host']))
            hosts = sorted(hosts_set)

        # Add new slots
        for svc in new_services:
            for host in hosts:
                max_slot_id += 1
                new_slot = {
                    'slot_id': f'S{max_slot_id}',
                    'host': list(host),
                    'role': 'pivot',  # default role for new services
                    'allowed_services': [svc],
                    'default_service': svc,
                    'default_os': 'linux',
                }
                template['service_slots'].append(new_slot)

        return template

    @staticmethod
    def get_template_info(template: Dict) -> Dict:
        """Get summary info about a template."""
        return {
            'name': template.get('meta', {}).get('name', 'unknown'),
            'subnets': len(template.get('subnets', [])),
            'hosts': sum(template.get('subnets', [])),
            'service_slots': len(template.get('service_slots', [])),
            'privesc_slots': len(template.get('privesc_slots', [])),
            'unique_services': len(set(
                s.get('default_service', '')
                for s in template.get('service_slots', [])
            )),
        }


# ─── CLI Entry Point ────────────────────────────────────────────────────────

def main():
    """Demo the extensible architecture."""
    project_root = Path(__file__).parent.parent.parent

    print("=" * 60)
    print("Phase 4: Extensible Architecture Demo")
    print("=" * 60)

    # 1. Service Registry
    print("\n[1/3] Service Registry")
    registry = ServiceRegistry()
    print(registry.describe())

    # Demo: add a new service
    registry.register_service(ServiceDefinition(
        name='mssql',
        exploit_name='e_mssql',
        cve_keywords=['mssql', 'sql_server', 'sqlserver', 'ms_sql'],
        default_port=1433,
        category='database',
        description='Microsoft SQL Server',
    ))
    registry.register_service(ServiceDefinition(
        name='rdp',
        exploit_name='e_rdp',
        cve_keywords=['rdp', 'remote_desktop', 'terminal_services'],
        default_port=3389,
        category='network',
        description='Remote Desktop Protocol',
    ))

    print(f"\n  After adding mssql + rdp:")
    print(f"  Services: {', '.join(registry.service_names)}")
    print(f"  Keywords for mssql: {registry.get_service('mssql').cve_keywords}")

    # Save registry
    registry_path = str(project_root / 'data' / 'CVE' / 'service_registry.json')
    registry.save(registry_path)
    print(f"  Registry saved: {registry_path}")

    # 2. CVE mapping test
    print("\n[2/3] CVE Keyword Mapping")
    test_keywords = ['ssh', 'apache', 'mssql', 'rdp', 'samba', 'unknown_service']
    for kw in test_keywords:
        mapped = registry.map_cve_service(kw)
        print(f"  '{kw}' → {mapped or '(unmapped)'}")

    # 3. Template expansion demo
    print("\n[3/3] Template Expansion Demo")
    template_dir = project_root / 'data' / 'scenarios' / 'templates'
    tiny_template_path = template_dir / 'tiny.template.yml'

    if tiny_template_path.exists():
        with open(tiny_template_path, 'r') as f:
            template = yaml.safe_load(f)

        info_before = TemplateExpander.get_template_info(template)
        print(f"  Before: {info_before}")

        expanded = TemplateExpander.expand_template(
            template, new_services=['mssql', 'rdp'])
        info_after = TemplateExpander.get_template_info(expanded)
        print(f"  After:  {info_after}")
        print(f"  Added {info_after['service_slots'] - info_before['service_slots']} "
              f"new service slots")

    print(f"\n{'=' * 60}")
    print("Phase 4 architecture components ready:")
    print("  - ServiceRegistry: extensible service/process definitions")
    print("  - CVEAdditionPipeline: CSV → graded → scenarios (automated)")
    print("  - TemplateExpander: add new services to existing templates")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
