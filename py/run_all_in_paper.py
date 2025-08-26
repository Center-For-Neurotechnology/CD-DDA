#!/usr/bin/env python3
"""
Complete DDA Analysis Pipeline for Roessler Systems.

This script orchestrates the complete analysis pipeline:
1. Generate synthetic data for 7 coupled Roessler systems
2. Perform DDA (Delay Differential Analysis)
3. Create comprehensive visualizations of results

The pipeline processes multiple coupling scenarios and noise conditions
to demonstrate the effectiveness of DDA for detecting causality and
ergodicity in complex dynamical systems.
"""

import sys
import time
from pathlib import Path

# Ensure we can import from the py directory
sys.path.append("py")

# Import all DDA functionality
from DDAfunctions import *


class DDAAnalysisPipeline:
    """Main pipeline class for DDA analysis of Roessler systems."""

    def __init__(self):
        """Initialize the analysis pipeline."""
        self.start_time = time.time()
        self.results = {}

    def run_stage(self, stage_name: str, script_name: str, description: str) -> None:
        """
        Execute a pipeline stage.

        Args:
            stage_name: Name of the stage for logging
            script_name: Python script to execute
            description: Human-readable description
        """
        print(f"\n{stage_name}: {description}")
        print("-" * 60)

        stage_start = time.time()

        try:
            # Execute the script
            with open(f"py/{script_name}") as script_file:
                script_code = compile(script_file.read(), script_name, "exec")
                exec(script_code, globals())

            stage_duration = time.time() - stage_start
            self.results[stage_name] = {
                "success": True,
                "duration": stage_duration,
                "description": description,
            }

            print(f"✓ Completed in {stage_duration:.2f} seconds")

        except Exception as e:
            stage_duration = time.time() - stage_start
            self.results[stage_name] = {
                "success": False,
                "duration": stage_duration,
                "error": str(e),
                "description": description,
            }

            print(f"✗ Failed after {stage_duration:.2f} seconds")
            print(f"  Error: {e}")
            raise

    def print_summary(self) -> None:
        """Print a summary of the pipeline execution."""
        total_duration = time.time() - self.start_time

        print("\n" + "=" * 60)
        print("DDA ANALYSIS PIPELINE SUMMARY")
        print("=" * 60)

        successful_stages = sum(1 for r in self.results.values() if r["success"])
        total_stages = len(self.results)

        print(f"Stages completed: {successful_stages}/{total_stages}")
        print(f"Total execution time: {total_duration:.2f} seconds")

        print("\nStage Details:")
        print("-" * 40)

        for stage_name, result in self.results.items():
            status = "✓ SUCCESS" if result["success"] else "✗ FAILED"
            duration = result["duration"]
            description = result["description"]

            print(f"{stage_name:20} {status:10} ({duration:6.2f}s) - {description}")

            if not result["success"]:
                print(f"{'':20} Error: {result['error']}")

        if successful_stages == total_stages:
            print("\n🎉 All stages completed successfully!")
            print("\nGenerated outputs:")
            print("  📊 DATA/ - Raw and processed time series data")
            print("  📈 DDA/ - DDA analysis results")
            print("  🎨 FIG/ - Visualization plots and network graphs")
        else:
            print(f"\n⚠️  {total_stages - successful_stages} stage(s) failed.")


def main():
    """Run the complete DDA analysis pipeline."""
    print("DDA ANALYSIS PIPELINE FOR ROESSLER SYSTEMS")
    print("=" * 50)
    print("This pipeline demonstrates Delay Differential Analysis")
    print("on coupled nonlinear dynamical systems.")
    print()
    print("Pipeline stages:")
    print("  1. Data Generation - Create synthetic Roessler system data")
    print("  2. DDA Analysis - Perform causality and ergodicity analysis")
    print("  3. Visualization - Generate comprehensive plots and networks")

    # Initialize pipeline
    pipeline = DDAAnalysisPipeline()

    try:
        # Stage 1: Generate synthetic data
        pipeline.run_stage(
            "STAGE_1_DATA",
            "make_data_7_systems.py",
            "Generate data for 7 coupled Roessler systems with different coupling scenarios",
        )

        # Stage 2: Perform DDA analysis
        pipeline.run_stage(
            "STAGE_2_ANALYSIS",
            "run_DDA_Roessler.py",
            "Run DDA analysis to detect causality and measure ergodicity",
        )

        # Stage 3: Create visualizations
        pipeline.run_stage(
            "STAGE_3_VISUALIZATION",
            "Roessler_ShowResults.py",
            "Generate comprehensive visualizations and network graphs",
        )

    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")

    except Exception as e:
        print(f"\n\n❌ Pipeline failed with error: {e}")

    finally:
        # Print summary regardless of success/failure
        pipeline.print_summary()

        # Check if all required outputs exist
        expected_files = [
            "DATA/Roessler_7syst_NoNoise.png",
            "DATA/Roessler_7syst_15dB.png",
            "FIG/E__WL2000_WS500_WN500_NoNoise.png",
            "FIG/C__WL2000_WS500_WN500_NoNoise.png",
            "FIG/CE__WL2000_WS500_WN500_NoNoise.png",
        ]

        missing_files = []
        for file_path in expected_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)

        if missing_files:
            print(f"\n⚠️  Some expected output files were not created:")
            for file_path in missing_files:
                print(f"    • {file_path}")

        print(f"\nFor more information about DDA, see the README.pdf")


if __name__ == "__main__":
    main()
