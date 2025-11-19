#!/usr/bin/env python3
"""
Wrapper script for precise-train command
Usage: python3 precise_train_wrapper.py [args...]
This script provides a fallback if precise-train command is not available
"""
import sys
import os

def main():
    # Try to import and use precise training tools
    try:
        # Try precise.train module
        from precise import train
        # Call the main function with sys.argv
        train.main()
    except ImportError:
        try:
            # Try precise_runner.train
            from precise_runner import train
            train.main()
        except ImportError:
            try:
                # Try precise_runner.runner.PreciseTrainer
                from precise_runner.runner import PreciseTrainer
                # This is a class, not a command - would need different approach
                print("Error: PreciseTrainer class found but command-line interface not available")
                print("You may need to use the Python API directly")
                sys.exit(1)
            except ImportError:
                print("Error: Could not find precise training module")
                print("Install with: pip install precise precise-runner")
                sys.exit(1)

if __name__ == "__main__":
    main()

