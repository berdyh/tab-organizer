"""Export functionality for sessions and clusters."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..sessions.manager import Session


class Exporter:
    """Export sessions to various formats."""
    
    def __init__(self, templates_dir: Optional[str] = None):
        self.templates_dir = templates_dir or str(
            Path(__file__).parent.parent.parent.parent.parent / "templates"
        )
        self._env: Optional[Environment] = None
    
    @property
    def env(self) -> Environment:
        """Lazy-load Jinja2 environment."""
        if self._env is None:
            self._env = Environment(
                loader=FileSystemLoader(self.templates_dir),
                autoescape=select_autoescape(["html", "xml"]),
            )
        return self._env
    
    def export_markdown(self, session: Session) -> str:
        """Export session to Markdown format."""
        lines = [
            f"# {session.name}",
            "",
            f"*Exported: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*",
            "",
            f"**Total URLs:** {session.url_store.count()}",
            "",
        ]
        
        # Export clusters if available
        if session.clusters:
            lines.append("## Clusters")
            lines.append("")
            
            for i, cluster in enumerate(session.clusters, 1):
                cluster_name = cluster.get("name", f"Cluster {i}")
                lines.append(f"### {cluster_name}")
                lines.append("")
                
                for url_data in cluster.get("urls", []):
                    url = url_data if isinstance(url_data, str) else url_data.get("url", "")
                    title = url_data.get("title", url) if isinstance(url_data, dict) else url
                    lines.append(f"- [{title}]({url})")
                
                lines.append("")
        else:
            # Export all URLs without clustering
            lines.append("## URLs")
            lines.append("")
            
            for record in session.url_store.get_all():
                title = record.metadata.get("title", record.original)
                lines.append(f"- [{title}]({record.original})")
        
        return "\n".join(lines)
    
    def export_json(self, session: Session) -> str:
        """Export session to JSON format."""
        data = {
            "session": {
                "id": session.id,
                "name": session.name,
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
            },
            "urls": [
                {
                    "original": r.original,
                    "normalized": r.normalized,
                    "status": r.status,
                    "metadata": r.metadata,
                }
                for r in session.url_store.get_all()
            ],
            "clusters": session.clusters,
        }
        return json.dumps(data, indent=2)
    
    def export_obsidian(self, session: Session) -> str:
        """Export session to Obsidian-compatible Markdown."""
        lines = [
            "---",
            f"title: {session.name}",
            f"date: {datetime.utcnow().strftime('%Y-%m-%d')}",
            "tags: [tab-organizer, research]",
            "---",
            "",
            f"# {session.name}",
            "",
        ]
        
        if session.clusters:
            for cluster in session.clusters:
                cluster_name = cluster.get("name", "Unnamed Cluster")
                # Create wiki-link compatible name
                safe_name = cluster_name.replace(" ", "-").lower()
                lines.append(f"## [[{safe_name}|{cluster_name}]]")
                lines.append("")
                
                for url_data in cluster.get("urls", []):
                    url = url_data if isinstance(url_data, str) else url_data.get("url", "")
                    title = url_data.get("title", url) if isinstance(url_data, dict) else url
                    lines.append(f"- [{title}]({url})")
                
                lines.append("")
        else:
            for record in session.url_store.get_all():
                title = record.metadata.get("title", record.original)
                lines.append(f"- [{title}]({record.original})")
        
        return "\n".join(lines)
    
    def export_notion(self, session: Session) -> dict:
        """Export session to Notion-compatible format (blocks)."""
        blocks = [
            {
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"type": "text", "text": {"content": session.name}}]
                }
            }
        ]
        
        if session.clusters:
            for cluster in session.clusters:
                cluster_name = cluster.get("name", "Unnamed Cluster")
                
                # Add cluster heading
                blocks.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": cluster_name}}]
                    }
                })
                
                # Add URLs as bookmarks
                for url_data in cluster.get("urls", []):
                    url = url_data if isinstance(url_data, str) else url_data.get("url", "")
                    blocks.append({
                        "object": "block",
                        "type": "bookmark",
                        "bookmark": {"url": url}
                    })
        else:
            for record in session.url_store.get_all():
                blocks.append({
                    "object": "block",
                    "type": "bookmark",
                    "bookmark": {"url": record.original}
                })
        
        return {"blocks": blocks}
    
    def export_html(self, session: Session) -> str:
        """Export session to HTML format."""
        try:
            template = self.env.get_template("export.html.j2")
            return template.render(session=session, export_time=datetime.utcnow())
        except Exception:
            # Fallback to basic HTML if template not found
            return self._generate_basic_html(session)
    
    def _generate_basic_html(self, session: Session) -> str:
        """Generate basic HTML without template."""
        html = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            f"<title>{session.name}</title>",
            "<style>",
            "body { font-family: system-ui, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }",
            "h1 { color: #333; }",
            "h2 { color: #666; margin-top: 30px; }",
            "ul { list-style-type: none; padding: 0; }",
            "li { margin: 10px 0; }",
            "a { color: #0066cc; text-decoration: none; }",
            "a:hover { text-decoration: underline; }",
            "</style>",
            "</head>",
            "<body>",
            f"<h1>{session.name}</h1>",
        ]
        
        if session.clusters:
            for cluster in session.clusters:
                cluster_name = cluster.get("name", "Unnamed Cluster")
                html.append(f"<h2>{cluster_name}</h2>")
                html.append("<ul>")
                
                for url_data in cluster.get("urls", []):
                    url = url_data if isinstance(url_data, str) else url_data.get("url", "")
                    title = url_data.get("title", url) if isinstance(url_data, dict) else url
                    html.append(f'<li><a href="{url}">{title}</a></li>')
                
                html.append("</ul>")
        else:
            html.append("<ul>")
            for record in session.url_store.get_all():
                title = record.metadata.get("title", record.original)
                html.append(f'<li><a href="{record.original}">{title}</a></li>')
            html.append("</ul>")
        
        html.extend(["</body>", "</html>"])
        return "\n".join(html)
    
    def export(self, session: Session, format: str) -> str:
        """Export session to specified format."""
        exporters = {
            "markdown": self.export_markdown,
            "md": self.export_markdown,
            "json": self.export_json,
            "obsidian": self.export_obsidian,
            "html": self.export_html,
        }
        
        exporter = exporters.get(format.lower())
        if not exporter:
            raise ValueError(f"Unsupported export format: {format}")
        
        result = exporter(session)
        if isinstance(result, dict):
            return json.dumps(result, indent=2)
        return result
