"""
Auto-generated Documentation Generator
Extracts architecture information from code and configuration
"""

import os
import yaml
import json
from typing import Dict, List, Any
from pathlib import Path

class DocumentationGenerator:
    """Generate architecture documentation from code and configuration"""
    
    def __init__(self, project_root: str = "../.."):
        self.project_root = Path(project_root)
        self.services_dir = self.project_root / "services"
        
    def extract_service_info(self) -> List[Dict[str, Any]]:
        """Extract service information from directory structure"""
        services = []
        
        if not self.services_dir.exists():
            return services
            
        for service_dir in self.services_dir.iterdir():
            if service_dir.is_dir() and not service_dir.name.startswith('.'):
                service_info = {
                    "name": service_dir.name,
                    "path": str(service_dir.relative_to(self.project_root)),
                    "has_dockerfile": (service_dir / "Dockerfile").exists(),
                    "has_requirements": (service_dir / "requirements.txt").exists(),
                    "has_tests": any(service_dir.glob("test_*.py")),
                }
                
                # Extract port from main.py if exists
                main_file = service_dir / "main.py"
                if main_file.exists():
                    port = self._extract_port_from_main(main_file)
                    if port:
                        service_info["port"] = port
                
                services.append(service_info)
        
        return services
    
    def _extract_port_from_main(self, main_file: Path) -> int:
        """Extract port number from main.py file"""
        try:
            with open(main_file, 'r') as f:
                content = f.read()
                # Look for common port patterns
                import re
                patterns = [
                    r'port["\s]*[:=]["\s]*(\d+)',
                    r'--port["\s]+(\d+)',
                    r'PORT["\s]*=["\s]*(\d+)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        return int(match.group(1))
        except Exception:
            pass
        return None
    
    def extract_docker_compose_info(self) -> Dict[str, Any]:
        """Extract information from docker-compose.yml"""
        compose_file = self.project_root / "docker-compose.yml"
        
        if not compose_file.exists():
            return {}
        
        try:
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
                return compose_data
        except Exception as e:
            print(f"Error reading docker-compose.yml: {e}")
            return {}
    
    def generate_service_matrix(self) -> str:
        """Generate markdown table of services"""
        services = self.extract_service_info()
        
        markdown = "## Service Matrix\n\n"
        markdown += "| Service | Port | Dockerfile | Tests | Requirements |\n"
        markdown += "|---------|------|------------|-------|-------------|\n"
        
        for service in sorted(services, key=lambda x: x['name']):
            name = service['name']
            port = service.get('port', 'N/A')
            dockerfile = "✅" if service['has_dockerfile'] else "❌"
            tests = "✅" if service['has_tests'] else "❌"
            requirements = "✅" if service['has_requirements'] else "❌"
            
            markdown += f"| {name} | {port} | {dockerfile} | {tests} | {requirements} |\n"
        
        return markdown
    
    def generate_architecture_overview(self) -> str:
        """Generate architecture overview documentation"""
        services = self.extract_service_info()
        compose_info = self.extract_docker_compose_info()
        
        markdown = "# System Architecture Overview\n\n"
        markdown += f"**Generated:** {self._get_timestamp()}\n\n"
        
        markdown += "## Services\n\n"
        markdown += f"Total Services: **{len(services)}**\n\n"
        
        markdown += self.generate_service_matrix()
        
        markdown += "\n## Docker Compose Configuration\n\n"
        if compose_info:
            if 'services' in compose_info:
                markdown += f"Configured Services: **{len(compose_info['services'])}**\n\n"
            if 'volumes' in compose_info:
                markdown += f"Volumes: **{len(compose_info['volumes'])}**\n\n"
            if 'networks' in compose_info:
                markdown += f"Networks: **{len(compose_info['networks'])}**\n\n"
        
        return markdown
    
    def generate_api_endpoints_doc(self) -> str:
        """Generate API endpoints documentation"""
        markdown = "# API Endpoints Reference\n\n"
        
        # This would ideally parse FastAPI routes from each service
        # For now, we'll provide a template
        
        markdown += "## Service Endpoints\n\n"
        markdown += "### API Gateway (Port 8080)\n"
        markdown += "- `GET /health` - Health check\n"
        markdown += "- `POST /api/*` - Route to backend services\n\n"
        
        markdown += "### URL Input Service (Port 8081)\n"
        markdown += "- `POST /urls/validate` - Validate URLs\n"
        markdown += "- `POST /urls/batch` - Batch URL processing\n\n"
        
        markdown += "### Visualization Service (Port 8090)\n"
        markdown += "- `GET /dashboard` - Interactive dashboard\n"
        markdown += "- `GET /architecture/diagram` - Architecture data\n"
        markdown += "- `GET /pipeline/status` - Pipeline metrics\n"
        markdown += "- `GET /diagrams/mermaid/{type}` - Mermaid diagrams\n\n"
        
        return markdown
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    def generate_all_docs(self, output_dir: str = "docs"):
        """Generate all documentation files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Generate architecture overview
        arch_doc = self.generate_architecture_overview()
        with open(output_path / "ARCHITECTURE_GENERATED.md", 'w') as f:
            f.write(arch_doc)
        
        # Generate API endpoints
        api_doc = self.generate_api_endpoints_doc()
        with open(output_path / "API_ENDPOINTS.md", 'w') as f:
            f.write(api_doc)
        
        print(f"✅ Documentation generated in {output_dir}/")
        print(f"   - ARCHITECTURE_GENERATED.md")
        print(f"   - API_ENDPOINTS.md")

if __name__ == "__main__":
    generator = DocumentationGenerator()
    generator.generate_all_docs()
