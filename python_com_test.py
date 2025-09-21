# Complete Library Test
# Save as complete_test.py

print("=== Testing All Required Libraries ===\n")

def test_library(name, import_statement):
    try:
        exec(import_statement)
        print(f"✅ {name}: SUCCESS")
        return True
    except ImportError as e:
        print(f"❌ {name}: FAILED - {e}")
        return False
    except Exception as e:
        print(f"⚠️  {name}: WARNING - {e}")
        return False

# Test all libraries
libraries = [
    ("OpenCV", "import cv2; print(f'   Version: {cv2.__version__}')"),
    ("MediaPipe", "import mediapipe as mp; print('   MediaPipe imported')"),
    ("NumPy", "import numpy as np; print(f'   Version: {np.__version__}')"),
    ("SciPy", "import scipy; print(f'   Version: {scipy.__version__}')"),
    ("Pygame", "import pygame; print('   Pygame imported')")
]

success_count = 0
for name, import_stmt in libraries:
    if test_library(name, import_stmt):
        success_count += 1
    print()

print(f"=== Results: {success_count}/{len(libraries)} libraries working ===")

if success_count == len(libraries):
    print("🎉 ALL LIBRARIES INSTALLED SUCCESSFULLY!")
    print("✅ Ready to start building the sleep detection system!")
else:
    print("❌ Some libraries missing. Install missing ones with:")
    print("pip install opencv-python mediapipe pygame")