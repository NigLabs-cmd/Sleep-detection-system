# MediaPipe Test Only
# Save as mediapipe_test.py

import cv2
import mediapipe as mp
import numpy as np

def test_mediapipe_only():
    print("=== Testing MediaPipe Face Detection Only ===")
    
    # Test MediaPipe import
    try:
        print(f"✅ MediaPipe version: {mp.__version__}")
    except:
        print("❌ MediaPipe version check failed")
    
    # Initialize MediaPipe with very basic settings
    print("Initializing MediaPipe Face Detection...")
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    
    # Use basic face detection (easier than face mesh)
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,  # Short range model
        min_detection_confidence=0.2  # Very low threshold
    )
    
    print("✅ MediaPipe Face Detection initialized")
    
    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera failed")
        return False
    
    print("✅ Camera opened")
    print("\n📋 Testing MediaPipe face detection...")
    print("👤 Look directly at the camera")
    print("💡 You should see a GREEN RECTANGLE around your face")
    print("⌨️ Press 'q' when done testing")
    
    frame_count = 0
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)  # Mirror
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # CRITICAL: Process the frame with MediaPipe
        try:
            results = face_detection.process(rgb_frame)
            
            # Check if face detected
            if results.detections:
                detection_count += 1
                print(f"🎉 FACE DETECTED in frame {frame_count}!")
                
                for detection in results.detections:
                    # Draw detection
                    mp_drawing.draw_detection(frame, detection)
                    
                    # Add big success message
                    cv2.putText(frame, "FACE FOUND!", (50, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
            else:
                if frame_count % 10 == 0:  # Every 10 frames
                    print(f"❌ No face in frame {frame_count}")
        
        except Exception as e:
            print(f"❌ MediaPipe error in frame {frame_count}: {e}")
        
        # Always show frame info
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {detection_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if detection_count == 0:
            cv2.putText(frame, "Looking for face...", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.putText(frame, "Press 'q' to quit", (10, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Show frame
        cv2.imshow('MediaPipe Face Detection Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Results
    detection_rate = (detection_count / frame_count) * 100 if frame_count > 0 else 0
    
    print(f"\n📊 MediaPipe Test Results:")
    print(f"   Total frames processed: {frame_count}")
    print(f"   Face detections: {detection_count}")
    print(f"   Detection rate: {detection_rate:.1f}%")
    
    if detection_count > 0:
        print("🎉 SUCCESS: MediaPipe face detection is working!")
        return True
    else:
        print("❌ FAILED: MediaPipe could not detect face")
        print("💡 Possible issues:")
        print("   - MediaPipe model not downloaded properly")
        print("   - Lighting too poor")
        print("   - Face angle/distance issues")
        print("   - MediaPipe compatibility issues")
        return False

if __name__ == "__main__":
    test_mediapipe_only()