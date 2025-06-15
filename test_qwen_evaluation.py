#!/usr/bin/env python3
"""
Test script to run a small evaluation for Qwen model to verify everything works.
"""

import os
import sys
import asyncio
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
sys.path.append("dataset_benchmark")

async def test_qwen_evaluation():
    """Run a small test evaluation for Qwen model."""
    
    # Set the API key directly
    api_key = "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"
    os.environ["OPENROUTER_API_KEY"] = api_key
    
    try:
        # Import after setting the API key
        from benchmark_models import ModelBenchmark
        
        print("=== Test Evaluation for Qwen 2.5 VL 32B Model ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Create benchmark instance with fixed dataset
        print("Initializing benchmark...")
        benchmark = ModelBenchmark(
            dataset_dir="dataset_benchmark_hf_updated_fixed",
            api_key=api_key,  # Pass API key directly
            max_retries=5,
            initial_delay=10,
            max_delay=60
        )
        
        print(f"Dataset loaded: {len(benchmark.dataset)} examples")
        
        # Model to evaluate
        model_id = "qwen/qwen2.5-vl-32b-instruct"
        print(f"Testing model: {model_id}")
        print("Running evaluation on 3 examples from task 13...")
        
        print("\nStarting test evaluation...")
        start_time = datetime.now()
        
        # Run benchmark for just a few examples to test
        results, results_file = await benchmark.run_benchmark(
            task_id="13",  # Only task 13
            model_ids=[model_id],
            with_answer=True,
            without_answer=False,  # Skip for faster test
            with_true_solution=False,  # Skip for faster test
            max_examples=3,  # Only 3 examples
            prompt_variant="detailed",
            include_examples=False
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\nTest evaluation completed!")
        print(f"Duration: {duration}")
        print(f"Results saved to: {results_file}")
        
        # Analyze results
        print("\nAnalyzing results...")
        analysis = benchmark.analyze_results(results)
        analysis_file = benchmark.save_analysis(analysis, results_file)
        
        # Print summary
        print("\n" + "="*50)
        print("QWEN 2.5 VL 32B TEST RESULTS SUMMARY")
        print("="*50)
        print(f"Model: {model_id}")
        print(f"Total examples: {analysis['total_examples']}")
        print(f"Total evaluations: {analysis['total_evaluations']}")
        print(f"Total cost: ${analysis['summary']['total_cost']:.4f}")
        print(f"Average evaluation time: {analysis['summary']['avg_evaluation_time']:.2f}s")
        print(f"Total duration: {duration}")
        
        if analysis['summary'].get('avg_quality_score') is not None:
            print(f"Average quality score: {analysis['summary']['avg_quality_score'] * 100:.2f}%")
        
        # Print detailed results
        for model_id_key, model_data in analysis["models"].items():
            print(f"\nModel: {model_id_key}")
            for answer_type, metrics in model_data.items():
                print(f"\n  === {answer_type.upper()} ===")
                print(f"    Accuracy: {metrics['accuracy']:.2f}%")
                
                if metrics.get('quality_score') is not None:
                    print(f"    Quality score: {metrics['quality_score']:.2f}%")
                if metrics.get('avg_score_distance') is not None:
                    print(f"    Avg. score distance: {metrics['avg_score_distance']:.2f}")
                
                print(f"    Evaluations: {metrics['evaluations']}")
                print(f"    Avg. evaluation time: {metrics['avg_evaluation_time']:.2f}s")
                print(f"    Total cost: ${metrics['total_cost']:.4f}")
        
        print(f"\nDetailed results saved to: {results_file}")
        print(f"Analysis saved to: {analysis_file}")
        
        # Show individual results
        print(f"\nIndividual evaluation results:")
        for i, result in enumerate(results):
            print(f"  Example {i+1}: Score {result['score']} (expected {result['expected_score']}) - Cost: ${result['cost']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'benchmark' in locals():
            await benchmark.close()

def main():
    """Main function."""
    print("=== Qwen 2.5 VL 32B Test Script ===")
    print("This script will run a small test evaluation with the model:")
    print("qwen/qwen2.5-vl-32b-instruct")
    print()
    print("This will:")
    print("- Evaluate 3 examples from task 13")
    print("- Test only 'with_answer' approach")
    print("- Take about 1-2 minutes")
    print("- Cost very little money (< $0.10)")
    print()
    
    # Check if we have the required files
    required_files = [
        "dataset_benchmark_hf_updated_fixed",
        "dataset_benchmark/benchmark_models.py",
        "backend/app"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Error: Missing required files/directories:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    # Run the test evaluation
    try:
        success = asyncio.run(test_qwen_evaluation())
        if success:
            print("\n" + "="*50)
            print("✓ QWEN 2.5 VL 32B TEST COMPLETED SUCCESSFULLY!")
            print("="*50)
            print("The model is working correctly. Ready for full evaluation.")
        else:
            print("\n✗ Qwen test failed.")
        return success
    except KeyboardInterrupt:
        print("\nTest cancelled by user.")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
