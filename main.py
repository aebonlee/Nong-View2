"""
Nong-View2 Main Pipeline
지리정보 기반 AI 농업 분석 파이프라인 메인 스크립트
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.config import get_config, Config
from core.logger import get_logger

# Import POD modules
from pods.pod1_ingestion import IngestionEngine
from pods.pod2_cropping import CroppingEngine
from pods.pod3_tiling import TilingEngine
from pods.pod4_ai_analysis import AnalysisEngine
from pods.pod5_merging import MergingEngine
from pods.pod6_gpkg_export import GPKGExporter

logger = get_logger(__name__)


class NongViewPipeline:
    """Main pipeline orchestrator for Nong-View2"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.config = Config(config_path)
        self.results = {}
        self.start_time = None
        self.end_time = None
        
        # Initialize POD engines
        self.pod1_engine = None
        self.pod2_engine = None
        self.pod3_engine = None
        self.pod4_engine = None
        self.pod5_engine = None
        self.pod6_engine = None
        
        logger.info("Nong-View2 Pipeline initialized")
    
    def run(self,
            input_image: Optional[str] = None,
            input_shapefile: Optional[str] = None,
            input_excel: Optional[str] = None,
            skip_pods: Optional[list] = None,
            only_pods: Optional[list] = None) -> Dict[str, Any]:
        """Run the complete pipeline
        
        Args:
            input_image: Path to input orthophoto
            input_shapefile: Path to shapefile with parcels
            input_excel: Path to Excel with PNU data
            skip_pods: List of POD numbers to skip (e.g., [1, 2])
            only_pods: List of POD numbers to run exclusively
            
        Returns:
            Dictionary with pipeline results
        """
        self.start_time = datetime.now()
        logger.info(f"Starting Nong-View2 Pipeline at {self.start_time}")
        
        try:
            # Determine which PODs to run
            pods_to_run = self._determine_pods_to_run(skip_pods, only_pods)
            
            # POD1: Data Ingestion
            if 1 in pods_to_run:
                logger.info("=" * 50)
                logger.info("Running POD1: Data Ingestion")
                logger.info("=" * 50)
                
                self.pod1_engine = IngestionEngine()
                pod1_results = self.pod1_engine.process(
                    image_path=input_image,
                    shapefile_path=input_shapefile,
                    excel_path=input_excel
                )
                
                self.results['pod1'] = pod1_results
                
                if pod1_results.get('status') != 'completed':
                    logger.error("POD1 failed, stopping pipeline")
                    return self.results
            
            # POD2: Cropping
            if 2 in pods_to_run:
                logger.info("=" * 50)
                logger.info("Running POD2: Cropping")
                logger.info("=" * 50)
                
                self.pod2_engine = CroppingEngine()
                pod2_results = self.pod2_engine.process()
                
                self.results['pod2'] = pod2_results
                
                if pod2_results.get('status') != 'completed':
                    logger.error("POD2 failed, stopping pipeline")
                    return self.results
            
            # POD3: Tiling
            if 3 in pods_to_run:
                logger.info("=" * 50)
                logger.info("Running POD3: Tiling")
                logger.info("=" * 50)
                
                self.pod3_engine = TilingEngine()
                pod3_results = self.pod3_engine.process()
                
                self.results['pod3'] = pod3_results
                
                if pod3_results.get('status') != 'completed':
                    logger.error("POD3 failed, stopping pipeline")
                    return self.results
            
            # POD4: AI Analysis
            if 4 in pods_to_run:
                logger.info("=" * 50)
                logger.info("Running POD4: AI Analysis")
                logger.info("=" * 50)
                
                self.pod4_engine = AnalysisEngine()
                pod4_results = self.pod4_engine.process()
                
                self.results['pod4'] = pod4_results
                
                if pod4_results.get('status') != 'completed':
                    logger.error("POD4 failed, stopping pipeline")
                    return self.results
            
            # POD5: Merging
            if 5 in pods_to_run:
                logger.info("=" * 50)
                logger.info("Running POD5: Merging")
                logger.info("=" * 50)
                
                self.pod5_engine = MergingEngine()
                pod5_results = self.pod5_engine.process()
                
                self.results['pod5'] = pod5_results
                
                if pod5_results.get('status') != 'completed':
                    logger.error("POD5 failed, stopping pipeline")
                    return self.results
            
            # POD6: GPKG Export
            if 6 in pods_to_run:
                logger.info("=" * 50)
                logger.info("Running POD6: GPKG Export")
                logger.info("=" * 50)
                
                self.pod6_engine = GPKGExporter()
                pod6_results = self.pod6_engine.process()
                
                self.results['pod6'] = pod6_results
                
                if pod6_results.get('status') != 'completed':
                    logger.error("POD6 failed")
            
            # Calculate total execution time
            self.end_time = datetime.now()
            execution_time = (self.end_time - self.start_time).total_seconds()
            
            # Save pipeline results
            self._save_pipeline_results(execution_time)
            
            # Print summary
            self._print_summary(execution_time)
            
            logger.info("=" * 50)
            logger.info(f"Pipeline completed successfully in {execution_time:.2f} seconds")
            logger.info("=" * 50)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}")
            self.results['error'] = str(e)
            return self.results
    
    def _determine_pods_to_run(self, skip_pods: Optional[list], 
                              only_pods: Optional[list]) -> list:
        """Determine which PODs to run based on skip/only parameters
        
        Args:
            skip_pods: PODs to skip
            only_pods: PODs to run exclusively
            
        Returns:
            List of POD numbers to run
        """
        all_pods = [1, 2, 3, 4, 5, 6]
        
        if only_pods:
            pods_to_run = [p for p in only_pods if p in all_pods]
        elif skip_pods:
            pods_to_run = [p for p in all_pods if p not in skip_pods]
        else:
            pods_to_run = all_pods
        
        logger.info(f"PODs to run: {pods_to_run}")
        return pods_to_run
    
    def _save_pipeline_results(self, execution_time: float) -> None:
        """Save complete pipeline results
        
        Args:
            execution_time: Total execution time in seconds
        """
        try:
            output_dir = self.config.output_dir
            
            pipeline_results = {
                'metadata': {
                    'start_time': self.start_time.isoformat(),
                    'end_time': self.end_time.isoformat(),
                    'execution_time_seconds': execution_time,
                    'version': '1.0.0'
                },
                'pod_results': self.results,
                'summary': self._generate_summary()
            }
            
            results_path = output_dir / f"pipeline_results_{self.start_time.strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
            
            logger.info(f"Pipeline results saved to: {results_path}")
            
        except Exception as e:
            logger.error(f"Error saving pipeline results: {str(e)}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate pipeline execution summary
        
        Returns:
            Summary dictionary
        """
        summary = {
            'completed_pods': [],
            'failed_pods': [],
            'statistics': {}
        }
        
        for pod_num in range(1, 7):
            pod_key = f'pod{pod_num}'
            if pod_key in self.results:
                if self.results[pod_key].get('status') == 'completed':
                    summary['completed_pods'].append(pod_num)
                else:
                    summary['failed_pods'].append(pod_num)
        
        # Collect key statistics
        if 'pod1' in self.results:
            pod1_stats = self.results['pod1'].get('statistics', {})
            summary['statistics']['total_images'] = pod1_stats.get('total_images', 0)
            summary['statistics']['total_parcels'] = pod1_stats.get('total_parcels', 0)
        
        if 'pod3' in self.results:
            pod3_stats = self.results['pod3'].get('statistics', {})
            summary['statistics']['total_tiles'] = pod3_stats.get('total_tiles', 0)
        
        if 'pod4' in self.results:
            pod4_stats = self.results['pod4'].get('statistics', {})
            summary['statistics']['total_detections'] = pod4_stats.get('total_detections', 0)
        
        if 'pod5' in self.results:
            pod5_stats = self.results['pod5'].get('statistics', {})
            summary['statistics']['merged_detections'] = pod5_stats.get('total_merged_detections', 0)
        
        if 'pod6' in self.results:
            summary['statistics']['gpkg_exported'] = bool(self.results['pod6'].get('gpkg_path'))
            summary['gpkg_path'] = self.results['pod6'].get('gpkg_path')
        
        return summary
    
    def _print_summary(self, execution_time: float) -> None:
        """Print pipeline execution summary
        
        Args:
            execution_time: Total execution time in seconds
        """
        print("\n" + "=" * 60)
        print(" NONG-VIEW2 PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        
        summary = self._generate_summary()
        
        print(f"\nExecution Time: {execution_time:.2f} seconds")
        print(f"Completed PODs: {summary['completed_pods']}")
        
        if summary['failed_pods']:
            print(f"Failed PODs: {summary['failed_pods']}")
        
        print("\nStatistics:")
        for key, value in summary['statistics'].items():
            print(f"  {key}: {value}")
        
        if summary.get('gpkg_path'):
            print(f"\nFinal Output: {summary['gpkg_path']}")
        
        print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Nong-View2: 지리정보 기반 AI 농업 분석 파이프라인'
    )
    
    # Input arguments
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input orthophoto image path (TIF/ECW)'
    )
    
    parser.add_argument(
        '--shapefile', '-s',
        type=str,
        help='Shapefile with parcel boundaries'
    )
    
    parser.add_argument(
        '--excel', '-e',
        type=str,
        help='Excel file with PNU data'
    )
    
    # Pipeline control arguments
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--skip-pods',
        type=int,
        nargs='+',
        help='POD numbers to skip (e.g., --skip-pods 1 2)'
    )
    
    parser.add_argument(
        '--only-pods',
        type=int,
        nargs='+',
        help='Only run specified PODs (e.g., --only-pods 3 4 5)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    # Create and run pipeline
    pipeline = NongViewPipeline(config_path=args.config)
    
    results = pipeline.run(
        input_image=args.input,
        input_shapefile=args.shapefile,
        input_excel=args.excel,
        skip_pods=args.skip_pods,
        only_pods=args.only_pods
    )
    
    # Exit with appropriate code
    if any(pod_result.get('status') == 'error' for pod_result in results.values()):
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()