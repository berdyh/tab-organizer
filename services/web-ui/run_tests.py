#!/usr/bin/env python3
"""
Web UI Test Runner

This script runs containerized tests for the Web UI service.
It includes unit tests, integration tests, and UI tests.
"""

import subprocess
import sys
import os
import time
import json
from pathlib import Path

def run_command(cmd, cwd=None, capture_output=False):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        if capture_output:
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, cwd=cwd, check=True)
            return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if capture_output and e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def check_docker():
    """Check if Docker is available."""
    try:
        run_command(['docker', '--version'], capture_output=True)
        run_command(['docker', 'compose', 'version'], capture_output=True)
        return True
    except:
        print("Docker or docker compose not found. Please install Docker.")
        return False

def build_test_image():
    """Build the test Docker image."""
    print("Building test image...")
    dockerfile_content = """
FROM node:18-alpine

WORKDIR /app

# Install curl for health checks
RUN apk add --no-cache curl

# Copy package files
COPY package*.json ./

# Install dependencies including dev dependencies
RUN npm install --silent

# Copy source code
COPY . .

# Run tests
CMD ["npm", "run", "test:ci"]
"""
    
    # Write test Dockerfile
    with open('Dockerfile.test', 'w') as f:
        f.write(dockerfile_content)
    
    return run_command(['docker', 'build', '--no-cache', '-f', 'Dockerfile.test', '-t', 'web-ui-test', '.'])

def run_unit_tests():
    """Run unit tests in Docker container."""
    print("\n=== Running Unit Tests ===")
    
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{os.getcwd()}:/app',
        '-v', '/app/node_modules',
        'web-ui-test'
    ]
    
    return run_command(cmd)

def run_integration_tests():
    """Run integration tests with API gateway."""
    print("\n=== Running Integration Tests ===")
    
    # Check if API gateway is running
    api_available = False
    try:
        result = run_command(['curl', '-f', 'http://localhost:8080/health'], capture_output=True)
        if result:
            print("API Gateway is running, proceeding with integration tests...")
            api_available = True
    except:
        pass
    
    if not api_available:
        print("API Gateway not available. Starting minimal test environment...")
        
        # Create a minimal test compose file
        test_compose = """
version: '3.8'
services:
  web-ui-integration-test:
    build:
      context: .
      dockerfile: Dockerfile.test
    environment:
      - REACT_APP_API_URL=http://mock-api:8080
      - CI=true
    depends_on:
      - mock-api
    networks:
      - test-network
      
  mock-api:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./test-fixtures/mock-api.conf:/etc/nginx/nginx.conf
    networks:
      - test-network

networks:
  test-network:
    driver: bridge
"""
        
        with open('docker-compose.test.yml', 'w') as f:
            f.write(test_compose)
        
        # Create mock API configuration
        os.makedirs('test-fixtures', exist_ok=True)
        mock_api_conf = """
events {}
http {
    server {
        listen 80;
        location /health {
            return 200 '{"status": "ok"}';
            add_header Content-Type application/json;
        }
        location /api/ {
            return 200 '{"data": []}';
            add_header Content-Type application/json;
        }
    }
}
"""
        with open('test-fixtures/mock-api.conf', 'w') as f:
            f.write(mock_api_conf)
        
        # Run integration tests with mock API
        cmd = [
            'docker', 'compose', '-f', 'docker-compose.test.yml',
            'run', '--rm', 'web-ui-integration-test'
        ]
        
        result = run_command(cmd)
        
        # Cleanup
        run_command(['docker', 'compose', '-f', 'docker-compose.test.yml', 'down', '-v'])
    else:
        # Run tests against real API
        cmd = [
            'docker', 'run', '--rm',
            '-v', f'{os.getcwd()}:/app',
            '-v', '/app/node_modules',
            '--network', 'host',
            '-e', 'REACT_APP_API_URL=http://localhost:8080',
            'web-ui-test'
        ]
        
        result = run_command(cmd)
    
    return result

def run_lint_tests():
    """Run linting tests."""
    print("\n=== Running Lint Tests ===")
    
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{os.getcwd()}:/app',
        '-v', '/app/node_modules',
        'web-ui-test',
        'npm', 'run', 'lint'
    ]
    
    return run_command(cmd)

def run_build_test():
    """Test if the application builds successfully."""
    print("\n=== Running Build Test ===")
    
    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{os.getcwd()}:/app',
        '-v', '/app/node_modules',
        'web-ui-test',
        'npm', 'run', 'build'
    ]
    
    return run_command(cmd)

def generate_test_report():
    """Generate test coverage report."""
    print("\n=== Generating Test Report ===")
    
    # Check if coverage directory exists
    if os.path.exists('coverage'):
        print("Test coverage report generated in ./coverage directory")
        
        # Try to read coverage summary
        try:
            with open('coverage/coverage-summary.json', 'r') as f:
                coverage_data = json.load(f)
                
            print("\nCoverage Summary:")
            for file_path, metrics in coverage_data.items():
                if file_path != 'total':
                    continue
                    
                print(f"Lines: {metrics['lines']['pct']}%")
                print(f"Functions: {metrics['functions']['pct']}%")
                print(f"Branches: {metrics['branches']['pct']}%")
                print(f"Statements: {metrics['statements']['pct']}%")
                
        except FileNotFoundError:
            print("Coverage summary not found")
    else:
        print("No coverage report generated")

def cleanup():
    """Clean up test artifacts."""
    print("\n=== Cleaning Up ===")
    
    # Remove test files
    test_files = ['Dockerfile.test', 'docker-compose.test.yml']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")
    
    # Remove test fixtures
    if os.path.exists('test-fixtures'):
        import shutil
        shutil.rmtree('test-fixtures')
        print("Removed test-fixtures directory")

def main():
    """Main test runner function."""
    print("Web UI Test Runner")
    print("==================")
    
    # Check prerequisites
    if not check_docker():
        sys.exit(1)
    
    # Change to the web-ui directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Track test results
    test_results = {}
    
    try:
        # Build test image
        if not build_test_image():
            print("Failed to build test image")
            sys.exit(1)
        
        # Run tests
        test_results['unit_tests'] = run_unit_tests()
        test_results['integration_tests'] = run_integration_tests()
        test_results['lint_tests'] = run_lint_tests()
        test_results['build_test'] = run_build_test()
        
        # Generate report
        generate_test_report()
        
        # Print summary
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        
        all_passed = True
        for test_name, result in test_results.items():
            status = "PASS" if result else "FAIL"
            print(f"{test_name.replace('_', ' ').title()}: {status}")
            if not result:
                all_passed = False
        
        if all_passed:
            print("\n✅ All tests passed!")
            exit_code = 0
        else:
            print("\n❌ Some tests failed!")
            exit_code = 1
            
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        exit_code = 1
    finally:
        cleanup()
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()