import sys
sys.path.insert(0, '/root/pest-detect')

print("=== Testing imports ===")
import ultralytics
print(f"Ultralytics: {ultralytics.__version__}")

from ultralytics import YOLO
print("Loading model...")
m = YOLO('/root/pest-detect/backend/models/best.pt')
print("Model loaded successfully!")
print(f"Model type: {type(m).__name__}")
print("DONE_OK")
