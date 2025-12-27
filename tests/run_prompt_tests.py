#!/usr/bin/env python3
"""
Test runner for VLMChat prompt module test suite.

This script provides a convenient way to run different categories of tests
for the prompt module with various options and configurations.
"""

import sys
import argparse
import subprocess
from pathlib import Path


def run_tests(args):
    """Run tests based on command line arguments."""

    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test path
    test_path = "src/tests/test_prompt"
    cmd.append(test_path)

    # Add markers based on test type
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    elif args.performance:
        cmd.extend(["-m", "performance"])
    elif args.edge_case:
        cmd.extend(["-m", "edge_case"])
    elif args.slow:
        cmd.extend(["-m", "slow"])

    # Add verbosity
    if args.verbose:
        cmd.append("-vvv")
    elif args.quiet:
        cmd.append("-q")
    else:
        cmd.append("-v")

    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            "--cov=src.prompt",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])

    # Add specific test file if provided
    if args.test_file:
        cmd = ["python", "-m", "pytest", f"src/tests/test_prompt/{args.test_file}"]
        if args.verbose:
            cmd.append("-vvv")

    # Add parallel execution if requested
    if args.parallel and not args.test_file:
        try:
            import pytest_xdist
            cmd.extend(["-n", str(args.parallel)])
        except ImportError:
            print("Warning: pytest-xdist not installed. Running tests sequentially.")

    # Add timeout if specified
    if args.timeout:
        try:
            import pytest_timeout
            cmd.extend(["--timeout", str(args.timeout)])
        except ImportError:
            print("Warning: pytest-timeout not installed. No timeout will be applied.")

    # Add benchmark if requested
    if args.benchmark:
        try:
            import pytest_benchmark
            cmd.extend(["--benchmark-only"])
        except ImportError:
            print("Warning: pytest-benchmark not installed. Skipping benchmarks.")

    # Add additional pytest arguments
    if args.pytest_args:
        cmd.extend(args.pytest_args.split())

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)

    # Run the tests
    result = subprocess.run(cmd)
    return result.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run VLMChat prompt module tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_prompt_tests.py                    # Run all tests
  python run_prompt_tests.py --unit             # Run only unit tests
  python run_prompt_tests.py --integration      # Run only integration tests
  python run_prompt_tests.py --performance      # Run only performance tests
  python run_prompt_tests.py --coverage         # Run with coverage report
  python run_prompt_tests.py --parallel 4       # Run with 4 parallel processes
  python run_prompt_tests.py --test-file test_history.py  # Run specific test file
        """
    )

    # Test type selection
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument("--unit", action="store_true",
                          help="Run only unit tests")
    test_group.add_argument("--integration", action="store_true",
                          help="Run only integration tests")
    test_group.add_argument("--performance", action="store_true",
                          help="Run only performance tests")
    test_group.add_argument("--edge-case", action="store_true",
                          help="Run only edge case tests")
    test_group.add_argument("--slow", action="store_true",
                          help="Run only slow tests")

    # Output options
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("-v", "--verbose", action="store_true",
                            help="Verbose output")
    output_group.add_argument("-q", "--quiet", action="store_true",
                            help="Quiet output")

    # Coverage options
    parser.add_argument("--coverage", action="store_true",
                       help="Generate coverage report")

    # Parallel execution
    parser.add_argument("--parallel", type=int, metavar="N",
                       help="Run tests in N parallel processes")

    # Timeout
    parser.add_argument("--timeout", type=int, metavar="SECONDS",
                       help="Timeout for individual tests")

    # Specific test file
    parser.add_argument("--test-file", metavar="FILENAME",
                       help="Run specific test file")

    # Benchmark
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark tests only")

    # Additional pytest arguments
    parser.add_argument("--pytest-args", metavar="ARGS",
                       help="Additional arguments to pass to pytest")

    # Quick test suites
    parser.add_argument("--quick", action="store_true",
                       help="Run quick test suite (unit tests only)")
    parser.add_argument("--full", action="store_true",
                       help="Run full test suite with coverage")

    args = parser.parse_args()

    # Handle special modes
    if args.quick:
        args.unit = True
        args.verbose = True

    if args.full:
        args.coverage = True
        args.verbose = True

    # Check if in correct directory
    if not Path("src/tests/test_prompt").exists():
        print("Error: Must run from VLMChat project root directory")
        sys.exit(1)

    # Run tests
    exit_code = run_tests(args)

    if exit_code == 0:
        print("\n" + "=" * 50)
        print("All tests passed! ✅")
    else:
        print("\n" + "=" * 50)
        print("Some tests failed! ❌")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()