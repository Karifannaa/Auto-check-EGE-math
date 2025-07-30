#!/usr/bin/env python3
"""
Script to run a test evaluation with a real API key.
This will test 1 example from task 13 to verify everything works.
"""

import os
import sys
import asyncio
import getpass
from datetime import datetime

# Add the backend directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
sys.path.append("dataset_benchmark")

async def run_test_evaluation():
    """Run a test evaluation with 1 example."""
    
    # Get API key from user
    api_key = getpass.getpass("Enter your OpenRouter API key: ")
    if not api_key or api_key.strip() == "":
        print("No API key provided. Exiting.")
        return False
    
    # Set the API key
    os.environ["OPENROUTER_API_KEY"] = api_key.strip()
    
    try:
        # Import after setting the API key
        from benchmark_models import ModelBenchmark
        
        print("=== Test Evaluation ===")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Create benchmark instance with updated dataset
        print("Initializing benchmark...")
        benchmark = ModelBenchmark(
            dataset_dir="dataset_benchmark_hf_updated",
            max_retries=3,
            initial_delay=5,
            max_delay=30
        )
        
        print(f"Dataset loaded: {len(benchmark.dataset)} examples")
        
        # Test with 1 example from task 13
        model_id = "google/gemini-2.0-flash-exp:free"
        print(f"Testing model: {model_id}")
        print("Running evaluation on 1 example from task 13...")
        
        # Run benchmark
        results, results_file = await benchmark.run_benchmark(
            task_id="13",  # Only task 13
            model_ids=[model_id],
            with_answer=True,
            without_answer=False,  # Skip without_answer for faster test
            with_true_solution=False,  # Skip true_solution for faster test
            max_examples=1,  # Only 1 example
            prompt_variant="detailed",
            include_examples=False
        )
        
        print(f"\nEvaluation completed!")
        print(f"Results saved to: {results_file}")
        
        # Analyze results
        print("\nAnalyzing results...")
        analysis = benchmark.analyze_results(results)
        analysis_file = benchmark.save_analysis(analysis, results_file)
        
        # Print summary
        print("\n=== Results Summary ===")
        print(f"Total examples: {analysis['total_examples']}")
        print(f"Total evaluations: {analysis['total_evaluations']}")
        print(f"Total cost: ${analysis['summary']['total_cost']:.4f}")
        
        if analysis['summary'].get('avg_quality_score') is not None:
            print(f"Average quality score: {analysis['summary']['avg_quality_score'] * 100:.2f}%")
        
        # Print detailed results
        for model_id, model_data in analysis["models"].items():
            print(f"\nModel: {model_id}")
            for answer_type, metrics in model_data.items():
                print(f"  {answer_type}:")
                print(f"    Accuracy: {metrics['accuracy']:.2f}%")
                if metrics.get('quality_score') is not None:
                    print(f"    Quality score: {metrics['quality_score']:.2f}%")
                print(f"    Evaluations: {metrics['evaluations']}")
                print(f"    Avg. evaluation time: {metrics['avg_evaluation_time']:.2f}s")
                print(f"    Total cost: ${metrics['total_cost']:.4f}")
        
        print(f"\nAnalysis saved to: {analysis_file}")
        
        # Show the actual result
        if results:
            result = results[0]
            print(f"\n=== Detailed Result ===")
            print(f"Solution ID: {result['solution_id']}")
            print(f"Expected score: {result['expected_score']}")
            print(f"Predicted score: {result['score']}")
            print(f"Evaluation time: {result['evaluation_time']:.2f}s")
            print(f"Tokens used: {result['total_tokens']}")
            print(f"Cost: ${result['cost']:.4f}")
            
            if result.get('result_text'):
                print(f"\nModel response (first 500 chars):")
                print(result['result_text'][:500] + "..." if len(result['result_text']) > 500 else result['result_text'])
        
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
    print("=== Test Evaluation Script ===")
    print("This script will run a test evaluation with 1 example to verify the system works.")
    print("You will need a valid OpenRouter API key.")
    print()
    
    # Check if we have the required files
    required_files = [
        "dataset_benchmark_hf_updated",
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
    
    # Run the test
    try:
        success = asyncio.run(run_test_evaluation())
        if success:
            print("\n✓ Test evaluation completed successfully!")
            print("The system is ready for full evaluation.")
        else:
            print("\n✗ Test evaluation failed.")
        return success
    except KeyboardInterrupt:
        print("\nEvaluation cancelled by user.")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
