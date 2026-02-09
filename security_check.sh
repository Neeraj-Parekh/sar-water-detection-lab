#!/bin/bash
# SAR Water Detection Lab - Pre-Release Security Check
# Run this before pushing to GitHub

set -e

echo "🔍 SAR Water Detection Lab - Security Scan"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check 1: Scan for hardcoded credentials
echo "1️⃣  Checking for hardcoded credentials..."
if grep -r "password\|secret\|api_key\|token\|credential" . --exclude-dir=".git" --exclude="security_check.sh" --exclude="*.md" --exclude="*.sh" --exclude="*.txt" | grep -v "# " | grep -v "your-" | grep -v "change-me" | grep -v "password@" | grep -v "PASSWORD" | grep -v "Token" | grep -v "_key" | grep -v "SECRET"; then
    echo -e "${RED}❌ FAIL: Found potential credentials in code${NC}"
    exit 1
else
    echo -e "${GREEN}✅ PASS: No hardcoded credentials found${NC}"
fi
echo ""

# Check 2: Scan for personal information
echo "2️⃣  Checking for personal information..."
if grep -r "100\.84\.105\|mitaoe\|neeraj-parekh" . --exclude-dir=".git" --exclude="security_check.sh" --exclude="*.md" | grep -v "example"; then
    echo -e "${RED}❌ FAIL: Found personal information${NC}"
    exit 1
else
    echo -e "${GREEN}✅ PASS: No personal information found${NC}"
fi
echo ""

# Check 3: Verify .env is gitignored
echo "3️⃣  Checking .gitignore..."
if ! grep -q "^\.env$" .gitignore; then
    echo -e "${RED}❌ FAIL: .env not in .gitignore${NC}"
    exit 1
else
    echo -e "${GREEN}✅ PASS: .env properly gitignored${NC}"
fi
echo ""

# Check 4: Verify no .env file exists
echo "4️⃣  Checking for .env file..."
if [ -f .env ]; then
    echo -e "${YELLOW}⚠️  WARNING: .env file exists (will not be committed)${NC}"
else
    echo -e "${GREEN}✅ PASS: No .env file present${NC}"
fi
echo ""

# Check 5: Verify config.py exists
echo "5️⃣  Checking for config.py..."
if [ ! -f config.py ]; then
    echo -e "${RED}❌ FAIL: config.py not found${NC}"
    exit 1
else
    echo -e "${GREEN}✅ PASS: config.py exists${NC}"
fi
echo ""

# Check 6: Verify README exists
echo "6️⃣  Checking for README.md..."
if [ ! -f README.md ]; then
    echo -e "${RED}❌ FAIL: README.md not found${NC}"
    exit 1
else
    echo -e "${GREEN}✅ PASS: README.md exists${NC}"
fi
echo ""

# Check 7: Verify LICENSE exists
echo "7️⃣  Checking for LICENSE..."
if [ ! -f LICENSE ]; then
    echo -e "${RED}❌ FAIL: LICENSE not found${NC}"
    exit 1
else
    echo -e "${GREEN}✅ PASS: LICENSE exists${NC}"
fi
echo ""

# Check 8: Verify Dockerfile exists
echo "8️⃣  Checking for Dockerfile..."
if [ ! -f Dockerfile ]; then
    echo -e "${RED}❌ FAIL: Dockerfile not found${NC}"
    exit 1
else
    echo -e "${GREEN}✅ PASS: Dockerfile exists${NC}"
fi
echo ""

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}🎉 All security checks passed!${NC}"
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Test Docker: docker build -t sar-lab-test ."
echo "3. Initialize git: git init"
echo "4. Commit: git add . && git commit -m 'Initial commit'"
echo "5. Push to GitHub"
echo ""
echo "See OPEN_SOURCE_SUMMARY.md for detailed release instructions"
echo "=========================================="
