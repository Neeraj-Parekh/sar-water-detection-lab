#!/bin/bash
# ================================================================================
# DEPLOY TO SERVER - Sync new modules to remote server
# ================================================================================
#
# This script syncs the new modules to the remote server for training.
#
# Usage:
#   ./deploy_to_server.sh          # Sync all new files
#   ./deploy_to_server.sh --dry-run # Show what would be synced
#
# Author: SAR Water Detection Project
# Date: 2026-01-26
# ================================================================================

set -e

# Configuration
SERVER_USER="mit-aoe"
SERVER_HOST="100.84.105.5"
SERVER_DIR="/home/mit-aoe/sar_water_detection"
LOCAL_DIR="/media/neeraj-parekh/Data1/sar soil system/chips/gui"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Files to sync
NEW_FILES=(
    "cldice_loss.py"
    "tta_module.py"
    "learnable_ensemble.py"
    "comprehensive_loss.py"
    "train_with_cldice_v10.py"
    "evaluate_tta.py"
    "benchmark_per_class.py"
)

echo "==============================================="
echo "  SAR Water Detection - Deploy to Server"
echo "==============================================="
echo ""

# Check if dry run
DRY_RUN=""
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN="--dry-run"
    echo -e "${YELLOW}DRY RUN MODE - No files will be transferred${NC}"
    echo ""
fi

# Check if sshpass is available
if ! command -v sshpass &> /dev/null; then
    echo -e "${YELLOW}sshpass not installed. You'll be prompted for password.${NC}"
    echo "Install with: sudo apt install sshpass"
    echo ""
    SSH_CMD="ssh"
    SCP_CMD="scp"
else
    echo -e "${GREEN}Using sshpass for authentication${NC}"
    SSH_CMD="sshpass -p 'mitaoe' ssh"
    SCP_CMD="sshpass -p 'mitaoe' scp"
fi

# Verify local files exist
echo "Checking local files..."
for file in "${NEW_FILES[@]}"; do
    if [[ -f "$LOCAL_DIR/$file" ]]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file (not found)"
    fi
done
echo ""

# Test server connection
echo "Testing server connection..."
if sshpass -p 'mitaoe' ssh -o ConnectTimeout=5 -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_HOST} "echo 'Connection successful'" 2>/dev/null; then
    echo -e "${GREEN}Server connection OK${NC}"
else
    echo -e "${RED}Cannot connect to server. Check if server is running and accessible.${NC}"
    echo "Try: sshpass -p 'mitaoe' ssh ${SERVER_USER}@${SERVER_HOST}"
    exit 1
fi
echo ""

# Create target directory on server
echo "Ensuring target directory exists..."
sshpass -p 'mitaoe' ssh ${SERVER_USER}@${SERVER_HOST} "mkdir -p ${SERVER_DIR}" 2>/dev/null
echo ""

# Sync files
echo "Syncing files..."
for file in "${NEW_FILES[@]}"; do
    if [[ -f "$LOCAL_DIR/$file" ]]; then
        echo -n "  Syncing $file... "
        if [[ -z "$DRY_RUN" ]]; then
            sshpass -p 'mitaoe' scp "$LOCAL_DIR/$file" "${SERVER_USER}@${SERVER_HOST}:${SERVER_DIR}/" 2>/dev/null
            echo -e "${GREEN}done${NC}"
        else
            echo -e "${YELLOW}(dry run)${NC}"
        fi
    fi
done
echo ""

# Verify files on server
if [[ -z "$DRY_RUN" ]]; then
    echo "Verifying files on server..."
    for file in "${NEW_FILES[@]}"; do
        if sshpass -p 'mitaoe' ssh ${SERVER_USER}@${SERVER_HOST} "test -f ${SERVER_DIR}/$file" 2>/dev/null; then
            size=$(sshpass -p 'mitaoe' ssh ${SERVER_USER}@${SERVER_HOST} "stat -c%s ${SERVER_DIR}/$file" 2>/dev/null)
            echo -e "  ${GREEN}✓${NC} $file (${size} bytes)"
        else
            echo -e "  ${RED}✗${NC} $file (missing)"
        fi
    done
    echo ""
fi

# Print next steps
echo "==============================================="
echo "  NEXT STEPS"
echo "==============================================="
echo ""
echo "1. SSH to server:"
echo "   sshpass -p 'mitaoe' ssh ${SERVER_USER}@${SERVER_HOST}"
echo ""
echo "2. Activate environment:"
echo "   source ~/anaconda3/bin/activate"
echo "   cd ${SERVER_DIR}"
echo ""
echo "3. Test modules:"
echo "   python cldice_loss.py"
echo "   python tta_module.py"
echo ""
echo "4. Train with clDice:"
echo "   python train_with_cldice_v10.py"
echo ""
echo "5. Evaluate TTA:"
echo "   python evaluate_tta.py --model_type pytorch"
echo ""
echo "6. Benchmark per-class:"
echo "   python benchmark_per_class.py --model_type lightgbm"
echo ""
echo "==============================================="
echo -e "  ${GREEN}Deployment complete!${NC}"
echo "==============================================="
