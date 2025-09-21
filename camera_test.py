# Camera Test - Next Step
# Save as camera_test.py

import cv2

def test_camera():
    print("=== Camera Test for Sleep Detection ===")
    print("Testing camera access...")
    
    # Try to open camera with DirectShow to avoid PRN errors
    print("Trying DirectShow backend...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("DirectShow failed, trying default...")
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Default failed, trying camera index 1...")
        cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print(" No camera found!")
        print("Solutions:")
        print("- Close Zoom, Teams, Skype, or other camera apps")
        print("- Check Windows camera permissions")
        print("- Try restarting VS Code")
        return False
    
    print(" Camera opened successfully!")
    
    # Test frame capture
    ret, frame = cap.read()
    if not ret:
        print(" Cannot read frames from camera")
        cap.release()
        return False
    
    print(f" Camera resolution: {frame.shape[1]}x{frame.shape[0]}")
    print("\n Starting live camera preview...")
    print(" Look at the camera - you should see yourself!")
    print(" Move around to test if video is smooth")
    print(" Press 'q' to quit when ready")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Lost camera connection")
            break
        
        frame_count += 1
        
        # Add helpful text overlays
        cv2.putText(frame, "Sleep Detection - Camera Test", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to continue to face detection", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Camera working perfectly!", (10, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show the frame
        cv2.imshow('Sleep Detection - Camera Test', frame)
        
        # Break on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n Camera test completed successfully!")
    print(f" Processed {frame_count} frames")
    print(" Camera is ready for face detection!")
    
    return True

if __name__ == "__main__":
    if test_camera():
        print("\n SUCCESS: Camera working perfectly!")
        print("Next step: Face detection test")
        print(" Run: python face_detection_test.py")
    else:
        print("\n FAILED: Fix camera issues first")
        print(" Try closing other camera applications")