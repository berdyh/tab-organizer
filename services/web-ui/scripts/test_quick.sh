#!/bin/bash
# Quick test script to verify fixes

echo "Running tests..."
CI=true npm run test:ci --passWithNoTests 2>&1 | tee test_output.log

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ Tests passed!"
    exit 0
else
    echo "❌ Tests failed!"
    tail -100 test_output.log
    exit 1
fi
