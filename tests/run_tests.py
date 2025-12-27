#!/usr/bin/env python3
"""
Platform-aware test runner for VLMChat.

Automatically detects platform and runs appropriate tests.
Default: Run smoke tests + platform-specific tests.

Usage:
    python tests/run_tests.py                    # Smoke + platform tests
    python tests/run_tests.py --smoke            # Only smoke tests
    python tests/run_tests.py --integration      # Integration tests
    python tests/run_tests.py --regression       # Full regression suite
    python tests/run_tests.py --all              # All tests for platform
    python tests/run_tests.py --verbose          # Detailed output
"""

import argparse
import sys
import subprocess
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from vlmchat.utils.platform_detect import detect_platform, Platform


class TestRunner:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.platform = detect_platform()
        self.tests_dir = Path(__file__).parent
        self.project_root = self.tests_dir.parent
        self.passed = 0
        self.failed = 0
    
    def print_platform_info(self):
        """Display platform and capabilities."""
        print("=" * 70)
        print(f"Platform: {self.platform.value if self.platform else 'Unknown'}")
        print(f"GStreamer: {'✓' if self._has_gstreamer() else '✗'}")
        print(f"OpenCV: {'✓' if self._has_opencv() else '✗'}")
        print(f"CUDA: {'✓' if self._has_cuda() else '✗'}")
        print("=" * 70)
    
    def run_smoke_tests(self):
        """Run fast unit tests (~seconds)."""
        print("\n🔥 Running Smoke Tests...")
        
        tests = [
            "smoke/test_buffer_pool.py",
            "smoke/test_contracts.py",
            "smoke/test_label_flow.py",
        ]
        return self._run_direct(tests)
    
    def run_integration_tests(self):
        """Run integration tests (~minutes)."""
        print("\n🔗 Running Integration Tests...")
        
        if not (self.tests_dir / "integration").exists():
            print("⚠️  No integration tests directory found")
            return True
        
        tests = [f"integration/{f.name}" for f in (self.tests_dir / "integration").glob("test_*.py")]
        if not tests:
            print("No integration tests found")
            return True
        
        return self._run_direct(tests)
    
    def run_regression_tests(self):
        """Run full regression suite (~10+ min)."""
        print("\n📊 Running Regression Tests...")
        
        if not (self.tests_dir / "regression").exists():
            print("⚠️  No regression tests directory found")
            return True
        
        tests = [f"regression/{f.name}" for f in (self.tests_dir / "regression").glob("test_*.py")]
        if not tests:
            print("No regression tests found")
            return True
        
        return self._run_direct(tests)
    
    def _run_direct(self, test_paths):
        """Run test files directly (exit code = number of failures)."""
        all_passed = True
        total_passed = 0
        total_failed = 0
        
        for path in test_paths:
            full_path = self.tests_dir / path
            if not full_path.exists():
                print(f"⚠️  Test path not found: {path}")
                continue
            
            print(f"\n{'='*70}")
            print(f"Running: {path}")
            print('='*70)
            
            result = subprocess.run(
                [sys.executable, str(full_path)],
                cwd=self.project_root
            )
            
            # Exit code 0 = success, non-zero = number of failures
            if result.returncode != 0:
                all_passed = False
                total_failed += result.returncode
            else:
                total_passed += 1
        
        self.passed += total_passed
        self.failed += total_failed
        
        print(f"\n{'='*70}")
        print(f"Suite results: {total_passed} files passed, {total_failed} failures")
        print('='*70)
        
        return all_passed
    
    @staticmethod
    def _has_gstreamer():
        try:
            import gi
            gi.require_version('Gst', '1.0')
            from gi.repository import Gst
            return True
        except:
            return False
    
    @staticmethod
    def _has_opencv():
        try:
            import cv2
            return True
        except:
            return False
    
    @staticmethod
    def _has_cuda():
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Run VLMChat tests with platform awareness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run smoke tests (default)
  %(prog)s --integration            # Run integration tests
  %(prog)s --regression             # Run full regression
  %(prog)s --all                    # Run all applicable tests
  %(prog)s --smoke --verbose        # Verbose smoke tests

Platform Behavior:
  - Tests marked 'all_platforms' or no marker: Run on all platforms
  - Tests marked with platform (jetson, rpi, mac): Run only on that platform
  - Platform-specific tests auto-skipped on other platforms
        """
    )
    
    parser.add_argument('--smoke', action='store_true',
                       help='Run smoke tests (fast, ~seconds)')
    parser.add_argument('--integration', action='store_true',
                       help='Run integration tests (~minutes)')
    parser.add_argument('--regression', action='store_true',
                       help='Run regression tests (~10+ minutes)')
    parser.add_argument('--all', action='store_true',
                       help='Run all applicable tests for platform')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Default to smoke tests if nothing specified
    if not any([args.smoke, args.integration, args.regression, args.all]):
        args.smoke = True
    
    runner = TestRunner(verbose=args.verbose)
    runner.print_platform_info()
    
    success = True
    
    if args.smoke or args.all:
        if not runner.run_smoke_tests():
            success = False
    
    if args.integration or args.all:
        if not runner.run_integration_tests():
            success = False
    
    if args.regression or args.all:
        if not runner.run_regression_tests():
            success = False
    
    print("\n" + "=" * 70)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("=" * 70)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
