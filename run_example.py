"""
Example script for running Nong-View2 pipeline
"""
from pathlib import Path
from main import NongViewPipeline
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def run_full_pipeline_example():
    """Run complete pipeline with example data"""
    
    # Initialize pipeline
    pipeline = NongViewPipeline()
    
    # Example input paths (replace with actual data paths)
    input_image = "data/input/orthophoto.tif"  # or .ecw
    input_shapefile = "data/input/parcels.shp"
    input_excel = "data/input/pnu_data.xlsx"
    
    # Run full pipeline
    results = pipeline.run(
        input_image=input_image,
        input_shapefile=input_shapefile,
        input_excel=input_excel
    )
    
    # Check results
    if results.get('pod6', {}).get('gpkg_path'):
        print(f"\nSuccess! GPKG output: {results['pod6']['gpkg_path']}")
    else:
        print("\nPipeline completed with issues. Check logs for details.")
    
    return results


def run_partial_pipeline_example():
    """Run only specific PODs"""
    
    pipeline = NongViewPipeline()
    
    # Example: Run only tiling, AI analysis, and merging (PODs 3, 4, 5)
    # Assuming PODs 1 and 2 were already run
    results = pipeline.run(
        only_pods=[3, 4, 5]
    )
    
    return results


def run_ai_analysis_only():
    """Run only AI analysis on existing tiles"""
    
    pipeline = NongViewPipeline()
    
    # Run only POD4 (AI Analysis)
    results = pipeline.run(
        only_pods=[4]
    )
    
    return results


def test_individual_pod():
    """Test individual POD module"""
    
    from pods.pod1_ingestion import IngestionEngine
    
    # Test POD1
    pod1 = IngestionEngine()
    
    # Process single image
    results = pod1.process(
        image_path="data/input/test_image.tif"
    )
    
    print(f"POD1 Results: {results['status']}")
    if results.get('images'):
        print(f"Processed {len(results['images'])} images")
    
    return results


if __name__ == '__main__':
    print("=" * 60)
    print("Nong-View2 Pipeline Example")
    print("=" * 60)
    
    # Choose which example to run
    print("\nSelect example to run:")
    print("1. Full pipeline")
    print("2. Partial pipeline (PODs 3-5)")
    print("3. AI analysis only (POD 4)")
    print("4. Test individual POD")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == '1':
        results = run_full_pipeline_example()
    elif choice == '2':
        results = run_partial_pipeline_example()
    elif choice == '3':
        results = run_ai_analysis_only()
    elif choice == '4':
        results = test_individual_pod()
    else:
        print("Invalid choice")
        results = None
    
    if results:
        print("\n" + "=" * 60)
        print("Pipeline execution completed!")
        print("Check 'data/output' directory for results")
        print("=" * 60)