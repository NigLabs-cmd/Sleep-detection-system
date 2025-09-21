# Complete Sleep Detection System - Final Version
# Professional College Project Implementation
# Sleep Detection with Music Volume Control

import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dist
import pygame
import threading

class SleepDetectionSystem:
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Eye landmark indices for EAR (Eye Aspect Ratio) calculation
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Sleep detection threshold parameters
        self.EAR_THRESHOLD = 0.25
        self.CONSECUTIVE_FRAMES = 30
        self.DROWSY_FRAMES = 60
        self.ASLEEP_FRAMES = 120
        
        # System state variables
        self.eye_closed_counter = 0
        self.total_blinks = 0
        self.start_time = time.time()
        self.current_state = "AWAKE"
        self.state_history = []
        self.frame_count = 0
        
        # Audio system initialization
        try:
            pygame.mixer.init()
            self.audio_initialized = True
            print("SUCCESS: Audio system initialized")
            
            # Try to load music file
            music_files = ["relaxing_music.mp3", "music.mp3", "sleep_music.mp3", "test.mp3"]
            self.music_file_loaded = False
            
            for music_file in music_files:
                try:
                    pygame.mixer.music.load(music_file)
                    self.music_file_loaded = True
                    print(f"SUCCESS: Music file '{music_file}' loaded successfully")
                    break
                except:
                    continue
            
            if not self.music_file_loaded:
                print("INFO: No music file found - running in simulation mode")
                print("To add music: Place 'relaxing_music.mp3' in the project folder")
            
        except:
            self.audio_initialized = False
            self.music_file_loaded = False
            print("WARNING: Audio system not available (demo mode)")
        
        self.current_volume = 0.7
        self.is_music_playing = False
        
        print("Sleep Detection System Initialized Successfully")
        print("System will track: Eye closure, Blink rate, Sleep states")
        print("Music control: Volume adjustment based on detected sleep state")
    
    def calculate_ear(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) for drowsiness detection
        EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
        """
        # Calculate vertical distances
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Calculate horizontal distance
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # Calculate and return EAR
        ear = (A + B) / (2.0 * C)
        return ear
    
    def extract_eye_coordinates(self, landmarks, eye_indices, width, height):
        """Extract normalized eye landmark coordinates"""
        coordinates = []
        for idx in eye_indices:
            x = int(landmarks.landmark[idx].x * width)
            y = int(landmarks.landmark[idx].y * height)
            coordinates.append([x, y])
        return coordinates
    
    def classify_sleep_state(self, ear, blink_rate):
        """
        Classify current sleep state based on eye metrics
        States: AWAKE -> TIRED -> RESTING -> DROWSY -> ASLEEP
        """
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        
        # Calculate blinks per minute
        if elapsed_time > 0:
            blinks_per_minute = (self.total_blinks / elapsed_time) * 60
        else:
            blinks_per_minute = 0
        
        # State classification algorithm
        if ear < 0.2 and self.eye_closed_counter > self.ASLEEP_FRAMES:
            state = "ASLEEP"
        elif ear < 0.22 and self.eye_closed_counter > self.DROWSY_FRAMES:
            state = "DROWSY"
        elif ear < self.EAR_THRESHOLD and self.eye_closed_counter > self.CONSECUTIVE_FRAMES:
            state = "RESTING"
        elif blinks_per_minute < 8 and ear < 0.28:
            state = "TIRED"
        else:
            state = "AWAKE"
        
        return state
    
    def control_music_volume(self, state):
        """Control music volume based on detected sleep state"""
        if state == "AWAKE":
            if not self.is_music_playing and self.music_file_loaded:
                try:
                    pygame.mixer.music.play(-1)  # Play indefinitely
                    self.is_music_playing = True
                    print("MUSIC: Started playing")
                except:
                    print("MUSIC: Started (simulated)")
                    self.is_music_playing = True
            elif not self.is_music_playing:
                print("MUSIC: Started (simulated - no file)")
                self.is_music_playing = True
            self.current_volume = min(0.8, self.current_volume + 0.01)
            
        elif state == "TIRED":
            self.current_volume = max(0.6, self.current_volume - 0.005)
            
        elif state == "RESTING":
            self.current_volume = max(0.4, self.current_volume - 0.01)
            if self.frame_count % 60 == 0:  # Print occasionally to avoid spam
                print(f"AUDIO: Resting detected - Volume: {self.current_volume:.2f}")
            
        elif state == "DROWSY":
            self.current_volume = max(0.2, self.current_volume - 0.02)
            if self.frame_count % 60 == 0:
                print(f"AUDIO: Drowsiness detected - Volume: {self.current_volume:.2f}")
            
        elif state == "ASLEEP":
            self.current_volume = max(0.0, self.current_volume - 0.05)
            if self.current_volume <= 0.1 and self.is_music_playing:
                if self.music_file_loaded:
                    try:
                        pygame.mixer.music.stop()
                        print("MUSIC: Stopped - User detected as asleep")
                    except:
                        print("MUSIC: Stopped (simulated)")
                else:
                    print("MUSIC: Stopped (simulated - no file)")
                self.is_music_playing = False
        
        # Apply volume control
        if self.music_file_loaded and self.audio_initialized:
            try:
                pygame.mixer.music.set_volume(self.current_volume)
            except:
                pass  # Handle any pygame errors silently
    
    def run_detection(self):
        """Main sleep detection processing loop"""
        # Initialize camera
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Cannot access camera device")
            return False
        
        print("\nSleep Detection System Started")
        print("INSTRUCTIONS:")
        print("  - Look at camera normally (AWAKE state)")
        print("  - Close eyes briefly (RESTING state)")  
        print("  - Keep eyes closed longer (DROWSY state)")
        print("  - Keep eyes closed extended time (ASLEEP state)")
        print("  - Press 'q' to quit system")
        print("=" * 50)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read camera frame")
                break
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)  # Create mirror effect
            h, w, _ = frame.shape
            
            # Convert BGR to RGB for MediaPipe processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process frame with MediaPipe Face Mesh
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract eye landmark coordinates
                    left_eye_coords = self.extract_eye_coordinates(
                        face_landmarks, self.LEFT_EYE, w, h)
                    right_eye_coords = self.extract_eye_coordinates(
                        face_landmarks, self.RIGHT_EYE, w, h)
                    
                    # Calculate Eye Aspect Ratio for both eyes
                    left_ear = self.calculate_ear(left_eye_coords)
                    right_ear = self.calculate_ear(right_eye_coords)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Process blink detection and eye closure
                    if avg_ear < self.EAR_THRESHOLD:
                        self.eye_closed_counter += 1
                    else:
                        if self.eye_closed_counter >= self.CONSECUTIVE_FRAMES:
                            self.total_blinks += 1
                        self.eye_closed_counter = 0
                    
                    # Determine current sleep state
                    blink_rate = self.total_blinks / (self.frame_count / 30) if self.frame_count > 30 else 0
                    self.current_state = self.classify_sleep_state(avg_ear, blink_rate)
                    
                    # Update state history for analysis
                    self.state_history.append(self.current_state)
                    if len(self.state_history) > 100:
                        self.state_history.pop(0)
                    
                    # Execute music volume control
                    self.control_music_volume(self.current_state)
                    
                    # Draw eye landmark points
                    for coord in left_eye_coords + right_eye_coords:
                        cv2.circle(frame, tuple(coord), 2, (0, 255, 0), -1)
                    
                    # Define color coding for different states
                    state_colors = {
                        "AWAKE": (0, 255, 0),      # Green
                        "TIRED": (0, 255, 255),    # Yellow  
                        "RESTING": (0, 165, 255),  # Orange
                        "DROWSY": (0, 100, 255),   # Dark Orange
                        "ASLEEP": (0, 0, 255)      # Red
                    }
                    
                    color = state_colors.get(self.current_state, (255, 255, 255))
                    
                    # Display system information on frame
                    cv2.putText(frame, f"State: {self.current_state}", (10, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                    
                    cv2.putText(frame, f"EAR: {avg_ear:.3f}", (10, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    cv2.putText(frame, f"Blinks: {self.total_blinks}", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    cv2.putText(frame, f"Volume: {self.current_volume:.2f}", (10, 140), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    cv2.putText(frame, f"Eye Closed Frames: {self.eye_closed_counter}", (10, 170), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display music status
                    music_status = "PLAYING" if self.is_music_playing else "STOPPED"
                    cv2.putText(frame, f"Music: {music_status}", (10, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Create EAR level visualization bar
                    bar_width = int(avg_ear * 300)
                    bar_color = (0, 255, 0) if avg_ear > 0.25 else (0, 0, 255)
                    cv2.rectangle(frame, (w-220, 50), (w-220 + bar_width, 70), bar_color, -1)
                    cv2.rectangle(frame, (w-220, 50), (w-145, 70), (255, 255, 255), 2)
                    cv2.putText(frame, "EAR Level", (w-220, 45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
            else:
                # Handle case when no face is detected
                cv2.putText(frame, "NO FACE DETECTED", (10, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, "Please look at the camera", (10, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Display user instructions
            cv2.putText(frame, "Press 'q' to quit", (10, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Show processed frame
            cv2.imshow('Sleep Detection System - College Project', frame)
            
            # Check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Generate final session statistics
        total_time = time.time() - self.start_time
        print("\nSESSION SUMMARY:")
        print(f"  Duration: {total_time:.1f} seconds")
        print(f"  Frames processed: {self.frame_count}")
        print(f"  Total blinks detected: {self.total_blinks}")
        print(f"  Final state: {self.current_state}")
        print(f"  Final volume level: {self.current_volume:.2f}")
        
        if len(self.state_history) > 0:
            state_counts = {state: self.state_history.count(state) for state in set(self.state_history)}
            print(f"  State distribution: {state_counts}")
        
        print("\nSleep Detection System session completed successfully")
        return True

def main():
    """Main function to initialize and run the sleep detection system"""
    print("=" * 60)
    print("SLEEP DETECTION SYSTEM - COLLEGE FINAL PROJECT")
    print("=" * 60)
    print("SYSTEM FEATURES:")
    print("  - Real-time face and eye tracking")
    print("  - Multi-state sleep classification (5 states)")
    print("  - Automated music volume control") 
    print("  - Blink detection and counting")
    print("  - Eye Aspect Ratio monitoring")
    print("  - Live performance statistics")
    print("=" * 60)
    
    try:
        detection_system = SleepDetectionSystem()
        detection_system.run_detection()
    except KeyboardInterrupt:
        print("\n\nSystem stopped by user")
    except Exception as e:
        print(f"\nERROR: {e}")
        print("Please ensure camera is available and not being used by other applications")

if __name__ == "__main__":
    main()