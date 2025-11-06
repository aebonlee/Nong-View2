"""
Unit tests for Nong-View2 pipeline
"""
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.config import Config
from core.utils import FileUtils, GeoUtils, ValidationUtils
from pods.pod1_ingestion import IngestionEngine, MetadataExtractor, FileConverter


class TestCoreModules(unittest.TestCase):
    """Test core utility modules"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        config = Config()
        self.assertIsNotNone(config._config)
        self.assertTrue(config.input_dir.exists())
        self.assertTrue(config.output_dir.exists())
    
    def test_file_utils(self):
        """Test file utilities"""
        # Test directory creation
        test_dir = Path("test_temp")
        created_dir = FileUtils.ensure_dir(test_dir)
        self.assertTrue(created_dir.exists())
        
        # Clean up
        test_dir.rmdir()
    
    def test_validation_utils(self):
        """Test validation utilities"""
        # Test PNU validation
        valid_pnu = "4511010100102080000"  # 19 digits
        invalid_pnu = "12345"
        
        self.assertTrue(ValidationUtils.validate_pnu(valid_pnu))
        self.assertFalse(ValidationUtils.validate_pnu(invalid_pnu))


class TestPOD1(unittest.TestCase):
    """Test POD1 Ingestion module"""
    
    def test_metadata_extractor(self):
        """Test metadata extraction"""
        extractor = MetadataExtractor()
        
        # Test with a dummy file path (would need real file for actual test)
        # metadata = extractor.extract_image_metadata("test.tif")
        
        self.assertIsNotNone(extractor)
    
    def test_file_converter(self):
        """Test file converter initialization"""
        converter = FileConverter()
        self.assertIsNotNone(converter)
    
    def test_ingestion_engine_init(self):
        """Test ingestion engine initialization"""
        engine = IngestionEngine()
        
        self.assertIsNotNone(engine)
        self.assertTrue(engine.output_dir.exists())
        self.assertIsInstance(engine.registry, dict)


class TestPOD2(unittest.TestCase):
    """Test POD2 Cropping module"""
    
    def test_cropping_engine_init(self):
        """Test cropping engine initialization"""
        from pods.pod2_cropping import CroppingEngine
        
        engine = CroppingEngine()
        
        self.assertIsNotNone(engine)
        self.assertTrue(engine.output_dir.exists())
        self.assertEqual(engine.use_convex_hull, True)


class TestPOD3(unittest.TestCase):
    """Test POD3 Tiling module"""
    
    def test_tiling_engine_init(self):
        """Test tiling engine initialization"""
        from pods.pod3_tiling import TilingEngine
        
        engine = TilingEngine()
        
        self.assertIsNotNone(engine)
        self.assertTrue(engine.output_dir.exists())
        self.assertEqual(engine.tile_size, 1024)
        self.assertEqual(engine.overlap, 0.2)


class TestPOD4(unittest.TestCase):
    """Test POD4 AI Analysis module"""
    
    def test_analysis_engine_init(self):
        """Test analysis engine initialization"""
        from pods.pod4_ai_analysis import AnalysisEngine
        
        engine = AnalysisEngine()
        
        self.assertIsNotNone(engine)
        self.assertTrue(engine.output_dir.exists())
        self.assertIsNotNone(engine.model_manager)
        self.assertEqual(len(engine.classes), 7)
    
    def test_model_manager_init(self):
        """Test model manager initialization"""
        from pods.pod4_ai_analysis import ModelManager
        
        manager = ModelManager()
        
        self.assertIsNotNone(manager)
        self.assertTrue(manager.model_dir.exists())


class TestPOD5(unittest.TestCase):
    """Test POD5 Merging module"""
    
    def test_merging_engine_init(self):
        """Test merging engine initialization"""
        from pods.pod5_merging import MergingEngine
        
        engine = MergingEngine()
        
        self.assertIsNotNone(engine)
        self.assertTrue(engine.output_dir.exists())
        self.assertEqual(engine.merge_strategy, 'nms')


class TestPOD6(unittest.TestCase):
    """Test POD6 GPKG Export module"""
    
    def test_gpkg_exporter_init(self):
        """Test GPKG exporter initialization"""
        from pods.pod6_gpkg_export import GPKGExporter
        
        exporter = GPKGExporter()
        
        self.assertIsNotNone(exporter)
        self.assertTrue(exporter.output_dir.exists())
        self.assertEqual(exporter.output_format, 'gpkg')
        self.assertEqual(exporter.coordinate_system, 'EPSG:5186')


class TestMainPipeline(unittest.TestCase):
    """Test main pipeline"""
    
    def test_pipeline_init(self):
        """Test pipeline initialization"""
        from main import NongViewPipeline
        
        pipeline = NongViewPipeline()
        
        self.assertIsNotNone(pipeline)
        self.assertIsNotNone(pipeline.config)
    
    def test_pod_determination(self):
        """Test POD selection logic"""
        from main import NongViewPipeline
        
        pipeline = NongViewPipeline()
        
        # Test skip pods
        pods = pipeline._determine_pods_to_run(skip_pods=[1, 2], only_pods=None)
        self.assertEqual(pods, [3, 4, 5, 6])
        
        # Test only pods
        pods = pipeline._determine_pods_to_run(skip_pods=None, only_pods=[3, 4])
        self.assertEqual(pods, [3, 4])
        
        # Test all pods
        pods = pipeline._determine_pods_to_run(skip_pods=None, only_pods=None)
        self.assertEqual(pods, [1, 2, 3, 4, 5, 6])


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestCoreModules))
    suite.addTests(loader.loadTestsFromTestCase(TestPOD1))
    suite.addTests(loader.loadTestsFromTestCase(TestPOD2))
    suite.addTests(loader.loadTestsFromTestCase(TestPOD3))
    suite.addTests(loader.loadTestsFromTestCase(TestPOD4))
    suite.addTests(loader.loadTestsFromTestCase(TestPOD5))
    suite.addTests(loader.loadTestsFromTestCase(TestPOD6))
    suite.addTests(loader.loadTestsFromTestCase(TestMainPipeline))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)