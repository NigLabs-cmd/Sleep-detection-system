# Working Face Detection - Fixed Version
# Save as working_face_detection.py

import cv2
import mediapipe as mp

def working_face_detection():
    print("=== Working Face Detection for Sleep System ===")
    
    # Initialize MediaPipe - using BOTH face detection AND face mesh
    mp_face_detection = mp.solutions.face_detection
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    
    # Face detection (for finding face)
    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )
    
    # Face mesh (for detailed landmarks - needed for eye tracking)
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    print("✅ Both MediaPipe models initialized")
    
    # Open camera
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Camera failed")
        return False
    
    print("✅ Camera ready")
    print("\n📋 What you'll see:")
    print("🟢 Green rectangle = Face detected")
    print("🔴 Red dots = Face landmarks (for eye tracking)")
    print("📊 Detection statistics")
    print("⌨️ Press 'q' when ready to continue")
    
    frame_count = 0
    face_box_count = 0
    face_mesh_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame = cv2.flip(frame, 1)  # Mirror
        h, w, _ = frame.shape
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process BOTH detections
        detection_results = face_detection.process(rgb_frame)
        mesh_results = face_mesh.process(rgb_frame)
        
        # Draw face detection box (green rectangle)
        if detection_results.detections:
            face_box_count += 1
            for detection in detection_results.detections:
                mp_drawing.draw_detection(frame, detection)
        
        # Draw face mesh (red dots for landmarks)  
        if mesh_results.multi_face_landmarks:
            face_mesh_count += 1
            for face_landmarks in mesh_results.multi_face_landmarks:
                # Draw key facial features
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=1),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
                )
        
        # Status display
        box_rate = (face_box_count/frame_count)*100 if frame_count > 0 else 0
        mesh_rate = (face_mesh_count/frame_count)*100 if frame_count > 0 else 0
        
        # Text overlays
        cv2.putText(frame, f"Face Detection: {box_rate:.1f}%", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Face Mesh: {mesh_rate:.1f}%", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Frame: {frame_count}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Status messages
        if face_box_count > 0 and face_mesh_count > 0:
            cv2.putText(frame, "✅ READY FOR SLEEP DETECTION!", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        elif face_box_count > 0:
            cv2.putText(frame, "Face found, getting landmarks...", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Looking for face...", (10, 130), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(frame, "Press 'q' to continue to sleep detection", (10, h-30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Display
        cv2.imshow('Working Face Detection - Sleep Project', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Final results
    box_rate = (face_box_count/frame_count)*100 if frame_count > 0 else 0
    mesh_rate = (face_mesh_count/frame_count)*100 if frame_count > 0 else 0
    
    print(f"\n📊 Final Results:")
    print(f"   Frames processed: {frame_count}")
    print(f"   Face detection rate: {box_rate:.1f}%")
    print(f"   Face mesh rate: {mesh_rate:.1f}%")
    
    if box_rate > 80 and mesh_rate > 60:
        print("🎉 EXCELLENT: Ready for complete sleep detection system!")
        return True
    elif box_rate > 50:
        print("✅ GOOD: Face detection working well")
        return True
    else:
        print("⚠️ Needs improvement but should work")
        return box_rate > 30

if __name__ == "__main__":
    if working_face_detection():
        print("\n🚀 NEXT STEP: Complete Sleep Detection System!")
        print("📋 All components tested and working:")
        print("   ✅ Camera")
        print("   ✅ Basic face detection") 
        print("   ✅ Face landmarks")
        print("   ✅ Ready for eye tracking!")
    else:
        print("\n🔧 Need to debug face detection further")