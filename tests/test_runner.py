#!/usr/bin/env python3
"""
Test runner for all test files in the tests directory
"""

import sys
import subprocess
from pathlib import Path
import importlib.util

def run_test_file(test_file_path):
    """Run a single test file and return success status"""
    
    print(f"\n{'='*60}")
    print(f"🧪 Running {test_file_path.name}")
    print(f"{'='*60}")
    
    try:
        # Load and run the test module
        spec = importlib.util.spec_from_file_location("test_module", test_file_path)
        test_module = importlib.util.module_from_spec(spec)
        
        # Add the test file's directory to path
        sys.path.insert(0, str(test_file_path.parent))
        
        # Execute the test module
        spec.loader.exec_module(test_module)
        
        print(f"✅ {test_file_path.name} completed successfully")
        return True
        
    except SystemExit as e:
        if e.code == 0:
            print(f"✅ {test_file_path.name} completed successfully")
            return True
        else:
            print(f"❌ {test_file_path.name} failed with exit code {e.code}")
            return False
            
    except Exception as e:
        print(f"❌ {test_file_path.name} failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Clean up the path
        if str(test_file_path.parent) in sys.path:
            sys.path.remove(str(test_file_path.parent))

def discover_and_run_tests():
    """Discover and run all test files"""
    
    tests_dir = Path(__file__).parent
    test_files = []
    
    # Find all test files
    for test_file in tests_dir.glob("test_*.py"):
        if test_file.name != "test_runner.py":  # Skip this file
            test_files.append(test_file)
    
    test_files.sort()  # Run in alphabetical order
    
    print(f"🔍 Discovered {len(test_files)} test files:")
    for test_file in test_files:
        print(f"  - {test_file.name}")
    
    # Run all tests
    results = []
    passed = 0
    failed = 0
    
    for test_file in test_files:
        success = run_test_file(test_file)
        results.append((test_file.name, success))
        
        if success:
            passed += 1
        else:
            failed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(test_files)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/len(test_files)*100):.1f}%")
    
    print(f"\nDetailed results:")
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    if failed == 0:
        print(f"\n🎉 All tests passed successfully!")
        return True
    else:
        print(f"\n❌ {failed} test(s) failed")
        return False

if __name__ == "__main__":
    print("🚀 Mental Health Text Classifier - Test Suite")
    print("=" * 60)
    
    success = discover_and_run_tests()
    
    if not success:
        sys.exit(1)
    
    print("\n✅ Test suite completed successfully!")
