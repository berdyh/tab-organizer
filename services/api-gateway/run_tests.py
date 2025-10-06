#!/usr/bin/env python3
"""Test runner for API Gateway service."""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

# Add the service directory to Python path
service_dir = Path(__file__).parent
sys.path.insert(0, str(service_dir))

def run_tests():
    """Run all tests for the API Gateway service."""
    print("🚀 Running API Gateway Integration Tests...")
    print("=" * 60)
    
    # Set environment variables for testing
    os.environ["TESTING"] = "true"
    os.environ["LOG_LEVEL"] = "DEBUG"
    
    # Test files to run
    test_files = [
        "test_api_gateway_integration.py"
    ]
    
    success = True
    
    for test_file in test_files:
        print(f"\n📋 Running {test_file}...")
        print("-" * 40)
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                test_file,
                "-v",
                "--tb=short",
                "--disable-warnings",
                f"--cov={service_dir}",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov"
            ], cwd=service_dir, capture_output=False)
            
            if result.returncode != 0:
                print(f"❌ {test_file} failed!")
                success = False
            else:
                print(f"✅ {test_file} passed!")
                
        except Exception as e:
            print(f"❌ Error running {test_file}: {e}")
            success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All API Gateway tests passed!")
        return 0
    else:
        print("💥 Some API Gateway tests failed!")
        return 1


def run_integration_tests():
    """Run integration tests with mock services."""
    print("\n🔗 Running Integration Tests with Mock Services...")
    print("=" * 60)
    
    # Test service communication
    test_scenarios = [
        "test_service_discovery",
        "test_service_proxy", 
        "test_health_monitoring",
        "test_rate_limiting",
        "test_authentication"
    ]
    
    for scenario in test_scenarios:
        print(f"\n🧪 Testing {scenario}...")
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest",
                "test_api_gateway_integration.py",
                "-k", scenario,
                "-v"
            ], cwd=service_dir)
            
            if result.returncode == 0:
                print(f"✅ {scenario} integration test passed!")
            else:
                print(f"❌ {scenario} integration test failed!")
                
        except Exception as e:
            print(f"❌ Error in {scenario}: {e}")


def validate_implementation():
    """Validate the API Gateway implementation."""
    print("\n🔍 Validating API Gateway Implementation...")
    print("=" * 60)
    
    validation_checks = []
    
    # Check required files exist
    required_files = [
        "main.py",
        "config.py", 
        "logging_config.py",
        "health_checker.py",
        "service_registry.py",
        "rate_limiter.py",
        "auth_middleware.py",
        "requirements.txt",
        "Dockerfile"
    ]
    
    for file in required_files:
        if (service_dir / file).exists():
            validation_checks.append(f"✅ {file} exists")
        else:
            validation_checks.append(f"❌ {file} missing")
    
    # Check for key functionality
    try:
        from main import app
        validation_checks.append("✅ FastAPI app imports successfully")
        
        from rate_limiter import RateLimiter
        validation_checks.append("✅ RateLimiter imports successfully")
        
        from auth_middleware import AuthMiddleware
        validation_checks.append("✅ AuthMiddleware imports successfully")
        
        from service_registry import ServiceRegistry
        validation_checks.append("✅ ServiceRegistry imports successfully")
        
        from health_checker import HealthChecker
        validation_checks.append("✅ HealthChecker imports successfully")
        
    except ImportError as e:
        validation_checks.append(f"❌ Import error: {e}")
    
    # Print validation results
    for check in validation_checks:
        print(check)
    
    # Check if all validations passed
    failed_checks = [check for check in validation_checks if check.startswith("❌")]
    if failed_checks:
        print(f"\n💥 {len(failed_checks)} validation checks failed!")
        return False
    else:
        print(f"\n🎉 All {len(validation_checks)} validation checks passed!")
        return True


def main():
    """Main test runner."""
    print("🏗️  API Gateway Service Test Suite")
    print("=" * 60)
    
    # Validate implementation first
    if not validate_implementation():
        print("\n❌ Implementation validation failed!")
        return 1
    
    # Run unit and integration tests
    test_result = run_tests()
    
    # Run integration tests
    run_integration_tests()
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    print(f"   Implementation Validation: ✅ Passed")
    print(f"   Unit & Integration Tests: {'✅ Passed' if test_result == 0 else '❌ Failed'}")
    print(f"   Service Communication: 🔗 Tested")
    
    return test_result


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)