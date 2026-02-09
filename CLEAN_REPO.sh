#!/bin/bash
# Remove unnecessary files from git

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Cleaning up repository...${NC}"
echo ""

# Remove old version files
echo "Removing old version files..."
git rm -rf retrain_v*.py attention_unet_v*.py ensemble_v*.py unet_v*.py 2>/dev/null || true
git rm -rf final_ensemble_v*.py train_with_cldice_v*.py master_training_pipeline*.py 2>/dev/null || true
git rm -rf equation_search_v*.py gpu_equation_search*.py 2>/dev/null || true
git rm -rf unified_vision_physics_v*.py physics_vision_hybrid_v*.py 2>/dev/null || true

# Remove training/development scripts
echo "Removing training scripts..."
git rm -rf train_physics_unet.py train_chip_selector.py physics_unet.py 2>/dev/null || true
git rm -rf learnable_ensemble.py conditional_ensemble_v*.py adaptive_*.py 2>/dev/null || true
git rm -rf pysr_water_equations.py sar_water_detector.py ensemble_water_detector.py 2>/dev/null || true

# Remove test files (except test_edge_cases.py)
echo "Removing old test files..."
git rm -rf test_ensemble.py test_gpu_pipeline.py 2>/dev/null || true

# Remove evaluation scripts
echo "Removing evaluation scripts..."
git rm -rf benchmark_per_class.py evaluate_tta.py comprehensive_evaluation.py 2>/dev/null || true
git rm -rf false_positive_analysis.py boundary_refinement.py full_ensemble_eval.py 2>/dev/null || true
git rm -rf full_pipeline_v*.py edge_case_handlers.py 2>/dev/null || true

# Remove loss functions
echo "Removing internal loss functions..."
git rm -rf tversky_centerline_loss.py cldice_loss.py comprehensive_loss.py tta_module.py 2>/dev/null || true

# Remove SOTA modules
echo "Removing research modules..."
git rm -rf sota_*.py 2>/dev/null || true

# Remove data processing
echo "Removing data processing scripts..."
git rm -rf convert_tif_to_npy*.py merge_features_labels.py 2>/dev/null || true

# Remove GEE scripts
echo "Removing GEE data collection scripts..."
git rm -rf gee_*.js gee_*.py india_chip_*.json india_chip_*.py 2>/dev/null || true

# Remove validators
echo "Removing development validators..."
git rm -rf lobo_validator.py data_validator*.py verify_smart_paste.py final_math_audit.py 2>/dev/null || true

# Remove server deployment with credentials
echo "Removing server deployment scripts..."
git rm -rf deploy_to_server.sh launch_gpu_pipeline.sh run_gpu_pipeline.py 2>/dev/null || true

# Remove old documentation
echo "Removing duplicate documentation..."
git rm -rf README_OLD.md SAR_WATER_DETECTION_*.md 2>/dev/null || true
git rm -rf COMPLETE_SYSTEM_DOCUMENTATION_*.md COMPREHENSIVE_LAB_GUIDE.md STAGE_V*.md 2>/dev/null || true

# Remove audit files
echo "Removing audit/notes files..."
git rm -rf "AUDIT LIST USER LOGIC.txt" "features audit list final.txt" "my talks.txt" 2>/dev/null || true
git rm -rf "SAR Software Feature Audit.md" 2>/dev/null || true

# Remove docs folder
echo "Removing docs folder..."
git rm -rf docs/ 2>/dev/null || true

# Remove tests folder
echo "Removing tests folder..."  
git rm -rf tests/ 2>/dev/null || true

# Remove download script
echo "Removing download script..."
git rm -rf download_from_gdrive.py 2>/dev/null || true

# Remove internal files
echo "Removing internal files..."
git rm -rf .cleanup_list RELEASE_SUMMARY.txt READY_TO_DEPLOY.md 2>/dev/null || true

echo ""
echo -e "${GREEN}✅ Cleanup complete!${NC}"
echo ""
echo "Files removed. Now commit the changes:"
echo ""
echo "  git commit -m \"chore: remove development and duplicate files\""
echo "  git push origin main"
echo ""
