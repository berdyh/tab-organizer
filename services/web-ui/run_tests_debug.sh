#!/bin/bash
# Debug test runner - captures full output

cd /media/shoh/Shared/Shox/Projects/tab_organizer/services/web-ui

echo "Building Docker image..."
docker build --no-cache -f Dockerfile.test -t web-ui-test . > /dev/null 2>&1

echo "Running tests and capturing output..."
docker run --rm -v $(pwd):/app -v /app/node_modules web-ui-test 2>&1 | tee full_test_output.txt

echo ""
echo "Test output saved to full_test_output.txt"
echo ""
echo "=== FAILED TESTS ==="
grep -A 10 "FAIL\|●" full_test_output.txt | head -100
