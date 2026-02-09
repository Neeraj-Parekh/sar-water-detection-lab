#!/bin/bash
# SAR Water Detection Lab - GitHub Deployment Script
# ====================================================
# This script will push your project to GitHub

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SAR Water Detection Lab - GitHub Push${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Verify we're in the right directory
echo -e "${YELLOW}Step 1: Verifying directory...${NC}"
if [ ! -f "app.py" ] || [ ! -f "config.py" ]; then
    echo -e "${RED}❌ Error: Not in the correct directory!${NC}"
    echo "Please run this script from: chips/gui/"
    exit 1
fi
echo -e "${GREEN}✅ Correct directory${NC}"
echo ""

# Step 2: Check if git is initialized
echo -e "${YELLOW}Step 2: Checking git status...${NC}"
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
    echo -e "${GREEN}✅ Git initialized${NC}"
else
    echo -e "${GREEN}✅ Git already initialized${NC}"
fi
echo ""

# Step 3: Configure git (if needed)
echo -e "${YELLOW}Step 3: Configuring git...${NC}"
git config user.name "Neeraj Parekh" || true
git config user.email "neeraj@example.com" || true
echo -e "${GREEN}✅ Git configured${NC}"
echo ""

# Step 4: Add all files
echo -e "${YELLOW}Step 4: Adding files to git...${NC}"
git add .
echo -e "${GREEN}✅ Files staged${NC}"
echo ""

# Step 5: Create initial commit
echo -e "${YELLOW}Step 5: Creating initial commit...${NC}"
if git rev-parse HEAD >/dev/null 2>&1; then
    echo "Repository already has commits"
else
    git commit -m "Initial commit: SAR Water Detection Lab v1.0.0

Features:
- 47+ water detection algorithms (classical + deep learning)
- Interactive Streamlit interface
- Real-time parameter tuning
- Docker deployment
- Ensemble fusion methods
- Custom equation engine
- QA/audit system
- Production-ready configuration

Tech Stack: Python, Streamlit, PyTorch, scikit-learn, GDAL, Docker"
    echo -e "${GREEN}✅ Initial commit created${NC}"
fi
echo ""

# Step 6: Set main branch
echo -e "${YELLOW}Step 6: Setting main branch...${NC}"
git branch -M main
echo -e "${GREEN}✅ Branch set to 'main'${NC}"
echo ""

# Step 7: Add remote (if not exists)
echo -e "${YELLOW}Step 7: Adding GitHub remote...${NC}"
REPO_URL="https://github.com/Neeraj-Parekh/sar-water-detection-lab.git"

if git remote get-url origin >/dev/null 2>&1; then
    echo "Remote 'origin' already exists"
    git remote set-url origin "$REPO_URL"
    echo -e "${GREEN}✅ Remote URL updated${NC}"
else
    git remote add origin "$REPO_URL"
    echo -e "${GREEN}✅ Remote added${NC}"
fi
echo ""

# Step 8: Push to GitHub
echo -e "${YELLOW}Step 8: Pushing to GitHub...${NC}"
echo -e "${BLUE}This will push to: $REPO_URL${NC}"
echo ""
echo -e "${YELLOW}⚠️  Make sure you've created the repository on GitHub first!${NC}"
echo -e "${YELLOW}   Go to: https://github.com/new${NC}"
echo -e "${YELLOW}   Repository name: sar-water-detection-lab${NC}"
echo -e "${YELLOW}   Visibility: Public${NC}"
echo ""
read -p "Press Enter to continue (or Ctrl+C to cancel)..."

git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}✅ SUCCESS! Project pushed to GitHub!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Your repository:${NC}"
    echo "https://github.com/Neeraj-Parekh/sar-water-detection-lab"
    echo ""
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. Go to your GitHub repository"
    echo "2. Add a description"
    echo "3. Add topics: sar, remote-sensing, python, streamlit, docker, water-detection"
    echo "4. Create a release (v1.0.0)"
    echo "5. Post on LinkedIn (see LINKEDIN_POST.md)"
    echo ""
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}❌ Error pushing to GitHub${NC}"
    echo -e "${RED}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Common issues:${NC}"
    echo "1. Repository doesn't exist on GitHub - create it first"
    echo "2. Authentication failed - set up GitHub token"
    echo "3. Network issues - check internet connection"
    echo ""
    echo -e "${YELLOW}Manual push command:${NC}"
    echo "git push -u origin main"
    exit 1
fi
