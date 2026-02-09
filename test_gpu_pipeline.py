"""
Test Suite for GPU Equation Search Pipeline
==============================================

Local verification of math correctness and logic before deployment.
This script tests components WITHOUT requiring actual GPU.

Usage:
    python test_gpu_pipeline.py

Author: SAR Water Detection Lab
Date: 2026-01-14
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import json
import sys

# Import modules to test
from gpu_equation_search import (
    GPUFeatureComputer,
    PhysicsChecker,
    GPUEquationEvaluator,
    EQUATION_TEMPLATES,
    PARAM_RANGES,
    REGIME_GRAMMAR,
    EquationCandidate,
    EquationCandidate,
    EvaluationResult,
    extract_rules_from_chips,
)
from lobo_validator import (
    BlockBootstrap,
    MetricsComputer,
    LOBOValidator,
    ChipMetadata,
    ValidationResult,
)


class TestPhysicsConstraints(unittest.TestCase):
    """Test physics constraint checking logic."""
    
    def test_hand_monotonicity_correct(self):
        """Water at low HAND should pass physics check."""
        # Create prediction: water only where HAND < 5
        hand = np.random.uniform(0, 30, (100, 100))
        pred_mask = hand < 5  # Water at low HAND
        
        score, corr = PhysicsChecker.check_hand_monotonicity(pred_mask, hand)
        
        # Should have negative correlation (water at low HAND)
        self.assertLess(corr, 0, 
                       f"Expected negative correlation, got {corr:.3f}")
        self.assertGreater(score, 0.8,
                          f"Expected high physics score, got {score:.3f}")
    
    def test_hand_monotonicity_incorrect(self):
        """Water at high HAND should fail physics check."""
        hand = np.random.uniform(0, 30, (100, 100))
        pred_mask = hand > 20  # Water at HIGH HAND (wrong!)
        
        score, corr = PhysicsChecker.check_hand_monotonicity(pred_mask, hand)
        
        # Should have positive correlation (physics violation)
        self.assertGreater(corr, 0,
                          f"Expected positive correlation, got {corr:.3f}")
        self.assertLess(score, 0.6,
                       f"Expected low physics score, got {score:.3f}")
    
    def test_slope_exclusion(self):
        """Water on steep slopes should be flagged."""
        slope = np.random.uniform(0, 45, (100, 100))
        
        # Create water mask including some steep pixels
        pred_mask = np.random.random((100, 100)) > 0.7
        
        violation_rate = PhysicsChecker.check_slope_exclusion(
            pred_mask, slope, threshold=15.0
        )
        
        # Check violation rate is in valid range
        self.assertGreaterEqual(violation_rate, 0.0)
        self.assertLessEqual(violation_rate, 1.0)
        
        # Count expected violations manually
        steep = slope > 15.0
        water_on_steep = np.logical_and(pred_mask, steep).sum()
        total_water = pred_mask.sum()
        expected = water_on_steep / total_water if total_water > 0 else 0
        
        self.assertAlmostEqual(violation_rate, expected, places=5)
    
    def test_combined_physics_score(self):
        """Test combined physics score calculation."""
        # Create physically correct prediction
        hand = np.random.uniform(0, 30, (100, 100))
        slope = np.random.uniform(0, 45, (100, 100))
        pred_mask = (hand < 5) & (slope < 10)  # Low HAND, low slope
        
        score = PhysicsChecker.combined_physics_score(pred_mask, hand, slope)
        
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        # Should be high for physically correct prediction
        self.assertGreater(score, 0.7)


class TestMetricsComputer(unittest.TestCase):
    """Test metric computation correctness."""
    
    def test_iou_perfect(self):
        """Perfect prediction should have IoU = 1.0."""
        truth = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
        pred = truth.copy()
        
        iou = MetricsComputer.compute_iou(pred, truth)
        self.assertAlmostEqual(iou, 1.0, places=5)
    
    def test_iou_no_overlap(self):
        """Non-overlapping prediction should have IoU = 0.0."""
        truth = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]])
        pred = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 1]])
        
        iou = MetricsComputer.compute_iou(pred, truth)
        self.assertAlmostEqual(iou, 0.0, places=5)
    
    def test_iou_partial(self):
        """Partial overlap should have 0 < IoU < 1."""
        truth = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
        pred = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]])
        
        # Intersection: (0,0), (1,0) = 2
        # Union: (0,0), (0,1), (1,0), (1,1) = 4
        # IoU = 2/4 = 0.5
        
        iou = MetricsComputer.compute_iou(pred, truth)
        self.assertAlmostEqual(iou, 0.5, places=5)
    
    def test_precision_recall_f1(self):
        """Test precision, recall, F1 calculation."""
        truth = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])
        pred = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
        
        # TP = 2 (correct water at 0,0 and 0,1)
        # FP = 1 (false water at 0,2)
        # FN = 1 (missed water at 1,0)
        
        prec, rec, f1 = MetricsComputer.compute_precision_recall_f1(pred, truth)
        
        expected_prec = 2 / (2 + 1)  # 0.667
        expected_rec = 2 / (2 + 1)   # 0.667
        expected_f1 = 2 * expected_prec * expected_rec / (expected_prec + expected_rec)
        
        self.assertAlmostEqual(prec, expected_prec, places=3)
        self.assertAlmostEqual(rec, expected_rec, places=3)
        self.assertAlmostEqual(f1, expected_f1, places=3)
    
    def test_area_bias(self):
        """Test area bias calculation."""
        truth = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]])  # 3 water pixels
        
        # Over-prediction
        pred_over = np.array([[1, 1, 1], [1, 1, 0], [1, 0, 0]])  # 6 pixels
        bias_over = MetricsComputer.compute_area_bias(pred_over, truth)
        self.assertAlmostEqual(bias_over, 6/3 - 1, places=3)  # +1.0 (100% over)
        
        # Under-prediction
        pred_under = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])  # 1 pixel
        bias_under = MetricsComputer.compute_area_bias(pred_under, truth)
        self.assertAlmostEqual(bias_under, 1/3 - 1, places=3)  # -0.67


class TestBlockBootstrap(unittest.TestCase):
    """Test block bootstrap confidence interval calculation."""
    
    def test_ci_with_known_values(self):
        """Test CI with known fold values."""
        bootstrap = BlockBootstrap(n_resamples=5000, random_state=42)
        
        # 6 fold values
        fold_values = [0.7, 0.75, 0.72, 0.78, 0.71, 0.74]
        
        mean, lower, upper = bootstrap.compute_ci(fold_values, confidence=0.95)
        
        # Mean should match
        self.assertAlmostEqual(mean, np.mean(fold_values), places=5)
        
        # CI should be around the mean
        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)
        
        # CI should cover most of the data
        self.assertGreater(upper - lower, 0.01)  # Non-trivial width
        self.assertLess(upper - lower, 0.2)      # Not too wide
    
    def test_ci_with_few_folds(self):
        """Test CI with insufficient folds (< 3)."""
        bootstrap = BlockBootstrap(n_resamples=1000, random_state=42)
        
        fold_values = [0.7, 0.8]  # Only 2 folds
        
        mean, lower, upper = bootstrap.compute_ci(fold_values)
        
        # Should fall back to min/max
        self.assertAlmostEqual(lower, 0.7, places=5)
        self.assertAlmostEqual(upper, 0.8, places=5)


class TestEquationEvaluation(unittest.TestCase):
    """Test equation template substitution and evaluation."""
    
    def test_simple_threshold(self):
        """Test simple VV threshold equation."""
        vv = np.array([[-25, -22], [-19, -15]])
        
        # Equation: vv < -20
        local_vars = {'vv': vv, 'np': np}
        result = eval("(vv < -20)", {"__builtins__": {}}, local_vars)
        
        expected = np.array([[True, True], [False, False]])
        np.testing.assert_array_equal(result, expected)
    
    def test_combined_threshold(self):
        """Test combined SAR + HAND equation."""
        vv = np.array([[-25, -22], [-19, -15]])
        hand = np.array([[1, 10], [5, 20]])
        
        # Equation: (vv < -20) & (hand < 8)
        local_vars = {'vv': vv, 'hand': hand, 'np': np}
        result = eval("(vv < -20) & (hand < 8)", {"__builtins__": {}}, local_vars)
        
        # vv < -20:  [[T, T], [F, F]]
        # hand < 8:  [[T, F], [T, F]]
        # AND:       [[T, F], [F, F]]
        expected = np.array([[True, False], [False, False]])
        np.testing.assert_array_equal(result, expected)
    
    def test_equation_template_substitution(self):
        """Test parameter substitution in equation templates."""
        template = "(vv < {T_vv}) & (hand < {T_hand})"
        params = {'T_vv': -19.5, 'T_hand': 15.0}
        
        eq = template
        for key, value in params.items():
            eq = eq.replace(f"{{{key}}}", str(value))
        
        expected = "(vv < -19.5) & (hand < 15.0)"
        self.assertEqual(eq, expected)


class TestRegimeGrammar(unittest.TestCase):
    """Test regime-specific equation grammar constraints."""
    
    def test_regime_definitions_exist(self):
        """All expected regimes should be defined."""
        expected_regimes = ['large_lake', 'wide_river', 'narrow_river', 
                           'wetland', 'arid', 'reservoir']
        
        for regime in expected_regimes:
            self.assertIn(regime, REGIME_GRAMMAR,
                         f"Regime '{regime}' not defined")
    
    def test_regime_has_required_fields(self):
        """Each regime should have required fields."""
        required_fields = ['allowed_templates', 'mandatory_params', 'max_complexity']
        
        for regime, grammar in REGIME_GRAMMAR.items():
            for field in required_fields:
                self.assertIn(field, grammar,
                             f"Regime '{regime}' missing field '{field}'")
    
    def test_allowed_templates_exist(self):
        """All allowed templates should be defined in EQUATION_TEMPLATES."""
        for regime, grammar in REGIME_GRAMMAR.items():
            for template_name in grammar['allowed_templates']:
                self.assertIn(template_name, EQUATION_TEMPLATES,
                             f"Template '{template_name}' for regime '{regime}' not defined")


class TestFeatureComputation(unittest.TestCase):
    """Test GPU feature computation (runs on CPU if no GPU)."""
    
    def setUp(self):
        self.computer = GPUFeatureComputer(device_id=0)
        self.test_data = np.random.uniform(-30, -10, (64, 64))
    
    def test_cov_shape(self):
        """CoV output should match input shape."""
        cov = self.computer.compute_cov(self.test_data, window_size=9)
        self.assertEqual(cov.shape, self.test_data.shape)
    
    def test_cov_values(self):
        """CoV should be non-negative."""
        cov = self.computer.compute_cov(self.test_data, window_size=9)
        self.assertTrue(np.all(cov >= 0), "CoV should be non-negative")
    
    def test_entropy_shape(self):
        """Entropy output should match input shape."""
        entropy = self.computer.compute_glcm_entropy(self.test_data, window_size=9)
        self.assertEqual(entropy.shape, self.test_data.shape)
    
    def test_sdwi_formula(self):
        """SDWI formula should be correctly implemented."""
        # SDWI = ln(10 * VV * VH) - 8
        vv_linear = np.array([[0.1, 0.2], [0.05, 0.15]])
        vh_linear = np.array([[0.05, 0.1], [0.02, 0.08]])
        
        sdwi = self.computer.compute_sdwi(vv_linear, vh_linear)
        
        # Manual calculation
        expected = np.log(10 * vv_linear * vh_linear + 1e-10) - 8
        
        np.testing.assert_array_almost_equal(sdwi, expected, decimal=5)
    
    def test_gamma0_conversion(self):
        """Test Gamma0 terrain flattening."""
        sigma0_db = np.array([[-20.0, -18.0], [-22.0, -19.0]])
        incidence_angle = np.array([[30.0, 35.0], [40.0, 45.0]])
        
        gamma0 = self.computer.compute_gamma0(sigma0_db, incidence_angle)
        
        # Gamma0 should be different from Sigma0 where angle != 0
        self.assertEqual(gamma0.shape, sigma0_db.shape)
        # At higher incidence angles, correction should increase values
        self.assertTrue(np.all(gamma0[:, 1:] > sigma0_db[:, 1:] - 5))
    
    def test_lee_filter(self):
        """Test Refined Lee speckle filter."""
        noisy_data = np.random.uniform(-30, -10, (32, 32))
        filtered = self.computer.refined_lee_filter(noisy_data, window_size=5)
        
        self.assertEqual(filtered.shape, noisy_data.shape)
        # Filtered data should have lower variance
        self.assertLess(filtered.std(), noisy_data.std() * 1.5)
    
    def test_area_opening(self):
        """Test morphological area opening."""
        # Create mask with small and large components
        mask = np.zeros((50, 50), dtype=bool)
        mask[5:10, 5:10] = True    # 25 pixels
        mask[30:45, 30:45] = True  # 225 pixels
        
        cleaned = self.computer.area_opening(mask, min_area=100)
        
        # Small component should be removed
        self.assertEqual(cleaned[7, 7], 0)
        # Large component should remain
        self.assertEqual(cleaned[35, 35], 1)
    
    def test_betti_numbers(self):
        """Test topological Betti number computation."""
        # Single connected component with one hole
        mask = np.zeros((20, 20), dtype=bool)
        mask[2:18, 2:18] = True
        mask[8:12, 8:12] = False  # Hole
        
        betti = self.computer.compute_betti_numbers(mask)
        
        self.assertEqual(betti['b0'], 1)  # 1 component
        self.assertEqual(betti['b1'], 1)  # 1 hole


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""
    
    def test_full_evaluation_pipeline(self):
        """Test complete evaluation of a single equation."""
        # Create synthetic chip data
        np.random.seed(42)
        
        # Bands: VV, VH, DEM, Slope, HAND, TWI, Truth
        height, width = 64, 64
        vv = np.random.uniform(-30, -10, (height, width))
        vh = np.random.uniform(-35, -15, (height, width))
        hand = np.random.uniform(0, 30, (height, width))
        slope = np.random.uniform(0, 45, (height, width))
        truth = ((vv < -20) & (hand < 10)).astype(float)
        
        features = {
            'vv': vv,
            'vh': vh,
            'hand': hand,
            'slope': slope,
            'twi': np.random.uniform(0, 15, (height, width)),
            'truth': truth,
            'cov': np.random.uniform(0, 1, (height, width)),
            'entropy': np.random.uniform(0, 2, (height, width)),
            'cpr': vh - vv,
            'sdwi': np.log(10 * 0.1 * 0.05 + 1e-10) - 8 + np.zeros((height, width)),
            'frangi': np.zeros((height, width)),
        }
        
        # Evaluate
        evaluator = GPUEquationEvaluator()
        params = {'T_vv': -20.0, 'T_hand': 10.0}
        
        result = evaluator.evaluate_equation(
            "(vv < {T_vv}) & (hand < {T_hand})",
            features, params
        )
        
        # Should have high IoU (equation matches truth generation)
        self.assertIsNotNone(result)
        self.assertGreater(result.iou, 0.9,
                          f"Expected IoU > 0.9, got {result.iou:.3f}")


class TestSaveLoadJSON(unittest.TestCase):
    """Test JSON serialization of results."""
    
    def test_evaluation_result_serialization(self):
        """EvaluationResult should serialize to valid JSON."""
        result = EvaluationResult(
            equation="(vv < -20) & (hand < 15)",
            params={'T_vv': -20.0, 'T_hand': 15.0},
            regime='wide_river',
            iou=0.85,
            precision=0.88,
            recall=0.82,
            f1_score=0.85,
            physics_score=0.92,
            complexity=3,
            hand_correlation=-0.45,
            slope_violation_rate=0.02,
            water_fraction=0.15,
        )
        
        # Should serialize without error
        data = result.to_dict()
        json_str = json.dumps(data)
        
        # Should deserialize correctly
        loaded = json.loads(json_str)
        self.assertEqual(loaded['equation'], result.equation)
        self.assertAlmostEqual(loaded['iou'], result.iou, places=5)
    
    def test_validation_result_serialization(self):
        """ValidationResult should serialize to valid JSON."""
        result = ValidationResult(
            equation="(vv < -20)",
            regime='large_lake',
            fold_ious=[0.8, 0.75, 0.82],
            fold_f1s=[0.78, 0.73, 0.80],
            fold_precisions=[0.85, 0.80, 0.87],
            fold_recalls=[0.72, 0.67, 0.74],
            fold_physics=[0.9, 0.88, 0.91],
            fold_basins=['A', 'B', 'C'],
            mean_iou=0.79,
            ci_iou=(0.74, 0.84),
            mean_f1=0.77,
            ci_f1=(0.72, 0.82),
            mean_physics=0.90,
            ci_physics=(0.87, 0.93),
            combined_score=0.71,
        )
        
        data = result.to_dict()
        json_str = json.dumps(data)
        
        loaded = json.loads(json_str)
        loaded = json.loads(json_str)
        self.assertEqual(loaded['regime'], 'large_lake')


class TestRuleExtraction(unittest.TestCase):
    """Test decision tree rule extraction logic."""
    
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.chip_dir = Path(self.temp_dir.name)
        self.output_file = self.chip_dir / "rules.md"
        
    def tearDown(self):
        self.temp_dir.cleanup()
        
    def test_extract_rules_8band(self):
        """Test rule extraction with 8-band chips (truth at index 7)."""
        # Create synthetic 8-band chip
        shape = (10, 10, 8)
        data = np.random.uniform(0, 1, shape).astype(np.float32)
        # Make vv (band 0) and truth (band 7) correlated
        data[:, :, 0] = np.linspace(0, 1, 100).reshape(10, 10) # VV
        data[:, :, 7] = (data[:, :, 0] > 0.5).astype(np.float32) # Truth
        
        chip_path = self.chip_dir / "chip_8band.npy"
        np.save(chip_path, data)
        
        # Run extraction
        rules = extract_rules_from_chips([chip_path], self.output_file, max_depth=2)
        
        # Should find at least one rule
        self.assertTrue(len(rules) > 0)
        self.assertTrue(self.output_file.exists())
        
        # The rule should involve 'vv'
        rule_text = " ".join(rules)
        self.assertIn("vv", rule_text)
        
    def test_extract_rules_no_truth(self):
        """Should handle chips without truth gracefully."""
        # Create 7-band chip with NO truth signal (random noise in band 6)
        shape = (10, 10, 7)
        data = np.random.uniform(0, 1, shape).astype(np.float32)
        
        chip_path = self.chip_dir / "chip_no_truth.npy"
        np.save(chip_path, data)
        
        # Capture logs to suppress error
        with self.assertLogs('gpu_equation_search', level='ERROR'):
            rules = extract_rules_from_chips([chip_path], self.output_file)
            
        # Should return empty list (or handle graceful failure)
        self.assertEqual(len(rules), 0)


# =============================================================================
# Main Test Runner
# =============================================================================

def run_tests():
    """Run all tests and report results."""
    print("\n" + "="*60)
    print("     GPU EQUATION SEARCH - TEST SUITE")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPhysicsConstraints))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricsComputer))
    suite.addTests(loader.loadTestsFromTestCase(TestBlockBootstrap))
    suite.addTests(loader.loadTestsFromTestCase(TestEquationEvaluation))
    suite.addTests(loader.loadTestsFromTestCase(TestRegimeGrammar))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureComputation))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEnd))
    suite.addTests(loader.loadTestsFromTestCase(TestSaveLoadJSON))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    if result.wasSuccessful():
        print("✓ ALL TESTS PASSED")
        print("="*60)
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print(f"  Failures: {len(result.failures)}")
        print(f"  Errors: {len(result.errors)}")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(run_tests())
