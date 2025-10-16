"""Template management for the export service."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

DEFAULT_TEMPLATES: Dict[str, str] = {
    "markdown_default": """# {{ session_name }} - Export Report

Generated on: {{ export_date }}
Total Items: {{ total_items }}
Total Clusters: {{ total_clusters }}

## Summary
{{ summary }}

{% for cluster in clusters %}
## Cluster {{ cluster.id }}: {{ cluster.label }}

**Size:** {{ cluster.size }} items
**Coherence Score:** {{ cluster.coherence_score }}

### Items in this cluster:
{% for item in cluster['items'] %}
- **{{ item.title }}** ({{ item.domain }})
  - URL: {{ item.url }}
  - Content Preview: {{ item.content[:200] }}...
  
{% endfor %}
{% endfor %}
""",
    "obsidian_default": """---
tags: [export, clustering, {{ session_id }}]
created: {{ export_date }}
total_items: {{ total_items }}
total_clusters: {{ total_clusters }}
---

# {{ session_name }} - Clustering Results

## Overview
- **Session ID:** {{ session_id }}
- **Export Date:** {{ export_date }}
- **Total Items:** {{ total_items }}
- **Total Clusters:** {{ total_clusters }}

## Clusters

{% for cluster in clusters %}
### [[Cluster {{ cluster.id }}]] - {{ cluster.label }}

**Metrics:**
- Size: {{ cluster.size }}
- Coherence: {{ cluster.coherence_score }}

**Items:**
{% for item in cluster['items'] %}
- [[{{ item.title }}]] - {{ item.domain }}
{% endfor %}

{% endfor %}

## Tags
{% for tag in all_tags %}
#{{ tag }} {% endfor %}
""",
    "json_default": "{{ data | tojson(indent=2) }}",
}


def ensure_default_templates(templates_dir: Path) -> None:
    """Write the built-in templates into the configured directory.

    Historically the service overwrote the on-disk templates during start-up to
    guarantee they contained the most up-to-date definitions. We keep the same
    behaviour to preserve compatibility with existing deployments and tests.
    """
    templates_dir.mkdir(parents=True, exist_ok=True)

    for template_name, content in DEFAULT_TEMPLATES.items():
        template_path = templates_dir / f"{template_name}.j2"
        template_path.write_text(content)


__all__ = ["DEFAULT_TEMPLATES", "ensure_default_templates"]
