"""
Tests for Documentation Generator
"""

import pytest
from pathlib import Path
from doc_generator import DocumentationGenerator

def test_documentation_generator_init():
    """Test documentation generator initialization"""
    generator = DocumentationGenerator()
    assert generator.project_root is not None
    assert generator.services_dir is not None

def test_extract_service_info():
    """Test service information extraction"""
    generator = DocumentationGenerator()
    services = generator.extract_service_info()
    
    # Should return a list
    assert isinstance(services, list)
    
    # Each service should have required fields
    for service in services:
        assert "name" in service
        assert "path" in service
        assert "has_dockerfile" in service
        assert "has_requirements" in service
        assert "has_tests" in service

def test_generate_service_matrix():
    """Test service matrix generation"""
    generator = DocumentationGenerator()
    matrix = generator.generate_service_matrix()
    
    assert isinstance(matrix, str)
    assert "## Service Matrix" in matrix
    assert "| Service | Port |" in matrix
    assert "Dockerfile" in matrix
    assert "Tests" in matrix

def test_generate_architecture_overview():
    """Test architecture overview generation"""
    generator = DocumentationGenerator()
    overview = generator.generate_architecture_overview()
    
    assert isinstance(overview, str)
    assert "# System Architecture Overview" in overview
    assert "## Services" in overview
    assert "Total Services:" in overview

def test_generate_api_endpoints_doc():
    """Test API endpoints documentation generation"""
    generator = DocumentationGenerator()
    api_doc = generator.generate_api_endpoints_doc()
    
    assert isinstance(api_doc, str)
    assert "# API Endpoints Reference" in api_doc
    assert "API Gateway" in api_doc
    assert "Visualization Service" in api_doc

def test_extract_docker_compose_info():
    """Test Docker Compose information extraction"""
    generator = DocumentationGenerator()
    compose_info = generator.extract_docker_compose_info()
    
    # Should return a dict (empty if file doesn't exist)
    assert isinstance(compose_info, dict)

def test_generate_all_docs(tmp_path):
    """Test generating all documentation files"""
    generator = DocumentationGenerator()
    output_dir = tmp_path / "docs"
    
    generator.generate_all_docs(str(output_dir))
    
    # Check that files were created
    assert (output_dir / "ARCHITECTURE_GENERATED.md").exists()
    assert (output_dir / "API_ENDPOINTS.md").exists()
    
    # Check content
    with open(output_dir / "ARCHITECTURE_GENERATED.md", 'r') as f:
        content = f.read()
        assert "System Architecture Overview" in content

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
