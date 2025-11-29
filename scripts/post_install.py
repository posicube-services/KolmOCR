#!/usr/bin/env python
"""
Post-installation script for olmocr.
Installs playwright chromium browser.
"""
import subprocess
import sys


def run_command(cmd, description):
    """Run a command and report status."""
    print(f"\n📦 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description} completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        return False


def main():
    """Run all post-installation tasks."""
    print("🎯 Running post-installation setup for olmocr...")
    
    success = run_command(
        "playwright install chromium",
        "Installing Playwright Chromium browser"
    )
    
    if success:
        print("\n✨ All setup tasks completed successfully!")
        print("You can now use olmocr. Run: python -m olmocr.pipeline --help")
        return 0
    else:
        print("\n⚠️  Some setup tasks failed. Please run manually:")
        print("   playwright install chromium")
        return 1


if __name__ == "__main__":
    sys.exit(main())
