# Advanced Sleep Detection System - Enhanced Version
# Addresses "Eyes Closed ≠ Asleep" Problem with Multi-Modal Detection
# College Final Year Project - Advanced Implementation

import cv2
import mediapipe as mp
import numpy as np
import time
from scipy.spatial import distance as dist
import pygame
import threading
from collections import deque
import statistics

class AdvancedSleepDetectionSystem:
    def __init__(self):
        # Initialize MediaPipe components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Eye landmark indices for EAR calculation
        self.LEFT_EYE = [362, 385, 387, 263, 373, 380]
        self.RIGHT_EYE = [33, 160, 158, 133, 153, 144]
        
        # Nose landmarks for breathing detection
        self.NOSE_TIP = 1
        self.NOSE_BRIDGE = 6
        
        # Advanced detection thresholds
        self.EAR_THRESHOLD = 0.25
        self.SLEEP_EAR_THRESHOLD = 0.18
        self.CONSECUTIVE_FRAMES = 30
        self.DROWSY_FRAMES = 90    # 3 seconds
        self.SLEEP_FRAMES = 240    # 8 seconds (more conservative)
        
        # Multi-modal detection variables
        self.eye_closed_counter = 0
        self.total_blinks = 0
        self.start_time = time.time()
        self.frame_count = 0
        
        # Advanced tracking arrays
        self.ear_history = deque(maxlen=60)  # 2 seconds of EAR values
        self.movement_history = deque(maxlen=90)  # 3 seconds of movement
        self.breathing_history = deque(maxlen=120)  # 4 seconds of breathing data
        self.state_history = deque(maxlen=150)  # 5 seconds of states
        
        # Confidence tracking
        self.sleep_confidence = 0.0
        self.current_state = "AWAKE"
        self.state_start_time = time.time()
        
        # Audio system with file detection
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
                
        except:
            self.audio_initialized = False
            self.music_file_loaded = False
            print("WARNING: Audio system not available")
        
        self.current_volume = 0.7
        self.is_music_playing = False
        
        print("Advanced Sleep Detection System Initialized")
        print("Features: Multi-modal detection, Confidence scoring, Movement analysis")
        print("Enhanced logic addresses 'eyes closed ≠ asleep' problem")
    
    def calculate_ear(self, eye_landmarks):
        """Calculate Eye Aspect Ratio with enhanced precision"""
        A = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        B = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        C = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        if C == 0:  # Prevent division by zero
            return 0.3
        
        ear = (A + B) / (2.0 * C)
        return ear
    
    def extract_landmarks(self, landmarks, indices, width, height):
        """Extract landmark coordinates with normalized positioning"""
        coords = []
        for idx in indices:
            x = int(landmarks.landmark[idx].x * width)
            y = int(landmarks.landmark[idx].y * height)
            coords.append([x, y])
        return coords
    
    def detect_breathing_pattern(self, face_landmarks):
        """
        Advanced breathing detection using nose region movement
        Returns breathing regularity score and pattern type
        """
        nose_tip_y = face_landmarks.landmark[self.NOSE_TIP].y
        nose_bridge_y = face_landmarks.landmark[self.NOSE_BRIDGE].y
        
        # Calculate relative nose movement
        nose_movement = abs(nose_tip_y - nose_bridge_y)
        self.breathing_history.append(nose_movement)
        
        if len(self.breathing_history) < 30:  # Need sufficient data
            return 0.5, "UNKNOWN"
        
        # Analyze breathing pattern over last 30 frames (1 second)
        recent_breathing = list(self.breathing_history)[-30:]
        
        # Calculate breathing metrics
        breathing_variance = statistics.variance(recent_breathing)
        breathing_mean = statistics.mean(recent_breathing)
        
        # Classify breathing pattern
        if breathing_variance < 0.00001 and breathing_mean < 0.002:
            return 0.9, "DEEP_SLEEP_BREATHING"  # Very regular, minimal movement
        elif breathing_variance < 0.0001 and breathing_mean < 0.005:
            return 0.7, "RELAXED_BREATHING"     # Somewhat regular
        elif breathing_variance < 0.001:
            return 0.4, "CALM_BREATHING"        # Irregular but calm
        else:
            return 0.1, "ACTIVE_BREATHING"      # Irregular, likely awake
    
    def detect_micro_movements(self, face_landmarks):
        """
        Detect micro-movements in facial features
        Sleeping people have significantly reduced micro-movements
        """
        # Track key facial points
        key_points = [1, 9, 10, 151, 175, 396, 397, 172, 136, 150]  # Various facial landmarks
        
        current_positions = []
        for point_idx in key_points:
            landmark = face_landmarks.landmark[point_idx]
            current_positions.append([landmark.x, landmark.y])
        
        if hasattr(self, 'previous_face_positions'):
            # Calculate total movement across all tracked points
            total_movement = 0
            for i, (curr_pos, prev_pos) in enumerate(zip(current_positions, self.previous_face_positions)):
                movement = dist.euclidean(curr_pos, prev_pos)
                total_movement += movement
            
            # Average movement per point
            avg_movement = total_movement / len(key_points)
            self.movement_history.append(avg_movement)
            
            # Analyze movement pattern
            if len(self.movement_history) >= 30:
                recent_movement = list(self.movement_history)[-30:]
                avg_recent_movement = statistics.mean(recent_movement)
                
                # Classification based on movement level
                if avg_recent_movement < 0.0005:
                    movement_score = 0.9  # Very still - likely asleep
                elif avg_recent_movement < 0.002:
                    movement_score = 0.6  # Somewhat still - drowsy
                elif avg_recent_movement < 0.005:
                    movement_score = 0.3  # Some movement - resting
                else:
                    movement_score = 0.1  # Active movement - awake
                
                self.previous_face_positions = current_positions
                return movement_score, avg_recent_movement
        
        self.previous_face_positions = current_positions
        return 0.5, 0.001  # Default values
    
    def calculate_sleep_confidence(self, ear, breathing_score, movement_score, time_in_state):
        """
        Multi-modal confidence calculation for sleep detection
        Combines eye closure, breathing, movement, and time factors
        """
        confidence_factors = {}
        
        # 1. Eye closure confidence
        if ear < self.SLEEP_EAR_THRESHOLD:
            confidence_factors['eye_closure'] = min(0.9, (0.25 - ear) * 4)
        elif ear < self.EAR_THRESHOLD:
            confidence_factors['eye_closure'] = 0.5
        else:
            confidence_factors['eye_closure'] = 0.1
        
        # 2. Breathing pattern confidence
        confidence_factors['breathing'] = breathing_score
        
        # 3. Movement stillness confidence
        confidence_factors['stillness'] = movement_score
        
        # 4. Time consistency confidence
        if time_in_state > 8.0:  # 8+ seconds in current state
            confidence_factors['time_consistency'] = 0.9
        elif time_in_state > 4.0:  # 4+ seconds
            confidence_factors['time_consistency'] = 0.6
        elif time_in_state > 2.0:  # 2+ seconds
            confidence_factors['time_consistency'] = 0.3
        else:
            confidence_factors['time_consistency'] = 0.1
        
        # 5. EAR stability confidence
        if len(self.ear_history) >= 30:
            ear_variance = statistics.variance(list(self.ear_history)[-30:])
            if ear_variance < 0.001:  # Very stable EAR
                confidence_factors['ear_stability'] = 0.8
            elif ear_variance < 0.01:
                confidence_factors['ear_stability'] = 0.5
            else:
                confidence_factors['ear_stability'] = 0.2
        else:
            confidence_factors['ear_stability'] = 0.3
        
        # Calculate weighted confidence
        weights = {
            'eye_closure': 0.3,
            'breathing': 0.25,
            'stillness': 0.25,
            'time_consistency': 0.15,
            'ear_stability': 0.05
        }
        
        total_confidence = sum(confidence_factors[factor] * weights[factor] 
                             for factor in confidence_factors)
        
        return total_confidence, confidence_factors
    
    def advanced_state_classification(self, ear, breathing_score, movement_score):
        """
        Advanced state classification using multi-modal confidence
        Addresses the 'eyes closed ≠ asleep' problem
        """
        current_time = time.time()
        time_in_state = current_time - self.state_start_time
        
        # Calculate comprehensive sleep confidence
        confidence, factors = self.calculate_sleep_confidence(
            ear, breathing_score, movement_score, time_in_state
        )
        
        self.sleep_confidence = confidence
        
        # Conservative state classification
        if confidence > 0.85 and ear < self.SLEEP_EAR_THRESHOLD:
            new_state = "DEEP_SLEEP"
        elif confidence > 0.7 and ear < 0.2:
            new_state = "LIGHT_SLEEP"
        elif confidence > 0.55 and ear < self.EAR_THRESHOLD:
            new_state = "DROWSY"
        elif ear < self.EAR_THRESHOLD and time_in_state < 3.0:
            new_state = "RESTING_EYES"  # Eyes closed but likely still awake
        elif ear < 0.28:
            new_state = "TIRED"
        else:
            new_state = "AWAKE"
        
        # Update state timing
        if new_state != self.current_state:
            self.state_start_time = current_time
        
        # Add to state history
        self.state_history.append(new_state)
        
        return new_state
    
    def intelligent_music_control(self, state, confidence):
        """
        Intelligent music control with conservative approach
        Only stops music with very high confidence
        """
        if state == "AWAKE":
            if not self.is_music_playing and self.music_file_loaded:
                try:
                    pygame.mixer.music.play(-1)
                    self.is_music_playing = True
                    print("MUSIC: Started playing")
                except:
                    self.is_music_playing = True
                    print("MUSIC: Started (simulation)")
            self.current_volume = min(0.8, self.current_volume + 0.005)
            
        elif state == "TIRED":
            self.current_volume = max(0.65, self.current_volume - 0.003)
            
        elif state == "RESTING_EYES":
            # Very gentle reduction - person might still be awake
            self.current_volume = max(0.5, self.current_volume - 0.002)
            if self.frame_count % 90 == 0:
                print(f"GENTLE: Eyes closed, gentle adjustment - Volume: {self.current_volume:.2f}")
            
        elif state == "DROWSY":
            self.current_volume = max(0.3, self.current_volume - 0.008)
            if self.frame_count % 90 == 0:
                print(f"DROWSY: Confidence {confidence:.2f} - Volume: {self.current_volume:.2f}")
            
        elif state == "LIGHT_SLEEP":
            self.current_volume = max(0.15, self.current_volume - 0.015)
            if self.frame_count % 90 == 0:
                print(f"LIGHT SLEEP: Confidence {confidence:.2f} - Volume: {self.current_volume:.2f}")
            
        elif state == "DEEP_SLEEP" and confidence > 0.85:
            # Only stop music with very high confidence
            self.current_volume = max(0.05, self.current_volume - 0.02)
            if self.current_volume <= 0.1 and self.is_music_playing:
                if self.music_file_loaded:
                    try:
                        pygame.mixer.music.stop()
                    except:
                        pass
                self.is_music_playing = False
                print(f"HIGH CONFIDENCE SLEEP: Music stopped (Confidence: {confidence:.2f})")
        
        # Apply volume control
        if self.music_file_loaded and self.audio_initialized:
            try:
                pygame.mixer.music.set_volume(self.current_volume)
            except:
                pass
    
    def run_advanced_detection(self):
        """Main advanced detection loop with enhanced processing"""
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("ERROR: Cannot access camera")
            return False
        
        print("\nAdvanced Sleep Detection System Started")
        print("ENHANCED FEATURES:")
        print("  - Multi-modal detection (eyes + breathing + movement)")
        print("  - Confidence-based classification")
        print("  - Conservative music control")
        print("  - Addresses 'eyes closed ≠ asleep' problem")
        print("  - Press 'q' to quit")
        print("=" * 60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Extract eye coordinates
                    left_eye = self.extract_landmarks(face_landmarks, self.LEFT_EYE, w, h)
                    right_eye = self.extract_landmarks(face_landmarks, self.RIGHT_EYE, w, h)
                    
                    # Calculate EAR
                    left_ear = self.calculate_ear(left_eye)
                    right_ear = self.calculate_ear(right_eye)
                    avg_ear = (left_ear + right_ear) / 2.0
                    self.ear_history.append(avg_ear)
                    
                    # Advanced detections
                    breathing_score, breathing_type = self.detect_breathing_pattern(face_landmarks)
                    movement_score, movement_amount = self.detect_micro_movements(face_landmarks)
                    
                    # Blink detection
                    if avg_ear < self.EAR_THRESHOLD:
                        self.eye_closed_counter += 1
                    else:
                        if self.eye_closed_counter >= 15:
                            self.total_blinks += 1
                        self.eye_closed_counter = 0
                    
                    # Advanced state classification
                    new_state = self.advanced_state_classification(avg_ear, breathing_score, movement_score)
                    
                    # Update state if changed
                    if new_state != self.current_state:
                        print(f"STATE CHANGE: {self.current_state} → {new_state} (Confidence: {self.sleep_confidence:.2f})")
                    self.current_state = new_state
                    
                    # Intelligent music control
                    self.intelligent_music_control(self.current_state, self.sleep_confidence)
                    
                    # Visual feedback
                    self.draw_advanced_interface(frame, avg_ear, breathing_type, movement_score, w, h)
            
            else:
                cv2.putText(frame, "NO FACE DETECTED", (10, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow('Advanced Sleep Detection System', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.generate_advanced_report()
        return True
    
    def draw_advanced_interface(self, frame, ear, breathing_type, movement_score, w, h):
        """Draw advanced interface with detailed metrics"""
        # State color coding
        state_colors = {
            "AWAKE": (0, 255, 0),
            "TIRED": (0, 255, 255),
            "RESTING_EYES": (0, 200, 255),
            "DROWSY": (0, 150, 255),
            "LIGHT_SLEEP": (0, 100, 255),
            "DEEP_SLEEP": (0, 0, 255)
        }
        
        color = state_colors.get(self.current_state, (255, 255, 255))
        
        # Main state display
        cv2.putText(frame, f"State: {self.current_state}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        
        # Confidence score
        confidence_color = (0, 255, 0) if self.sleep_confidence < 0.5 else (0, 255, 255) if self.sleep_confidence < 0.8 else (0, 0, 255)
        cv2.putText(frame, f"Sleep Confidence: {self.sleep_confidence:.2f}", (10, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, confidence_color, 2)
        
        # Detailed metrics
        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Breathing: {breathing_type}", (10, 135), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Movement Score: {movement_score:.2f}", (10, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Volume: {self.current_volume:.2f}", (10, 185), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (10, 210), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Music status
        music_status = "PLAYING" if self.is_music_playing else "STOPPED"
        cv2.putText(frame, f"Music: {music_status}", (10, 235), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Confidence visualization bar
        conf_bar_width = int(self.sleep_confidence * 200)
        conf_color = (0, 255, 0) if self.sleep_confidence < 0.5 else (0, 255, 255) if self.sleep_confidence < 0.8 else (0, 0, 255)
        cv2.rectangle(frame, (w-220, 30), (w-220 + conf_bar_width, 50), conf_color, -1)
        cv2.rectangle(frame, (w-220, 30), (w-20, 50), (255, 255, 255), 2)
        cv2.putText(frame, "Sleep Confidence", (w-220, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        cv2.putText(frame, "Advanced Multi-Modal Detection", (10, h-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, h-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    def generate_advanced_report(self):
        """Generate comprehensive session report"""
        total_time = time.time() - self.start_time
        
        print("\n" + "="*60)
        print("ADVANCED SLEEP DETECTION SESSION REPORT")
        print("="*60)
        print(f"Session Duration: {total_time:.1f} seconds")
        print(f"Frames Processed: {self.frame_count}")
        print(f"Processing Rate: {self.frame_count/total_time:.1f} FPS")
        print(f"Total Blinks Detected: {self.total_blinks}")
        print(f"Final State: {self.current_state}")
        print(f"Final Sleep Confidence: {self.sleep_confidence:.2f}")
        print(f"Final Volume Level: {self.current_volume:.2f}")
        
        if len(self.state_history) > 0:
            state_counts = {}
            for state in self.state_history:
                state_counts[state] = state_counts.get(state, 0) + 1
            
            print("\nState Distribution:")
            for state, count in sorted(state_counts.items()):
                percentage = (count / len(self.state_history)) * 100
                print(f"  {state}: {count} frames ({percentage:.1f}%)")
        
        print(f"\nAdvanced Features Utilized:")
        print(f"  - Multi-modal confidence scoring")
        print(f"  - Breathing pattern analysis")
        print(f"  - Micro-movement detection")
        print(f"  - Conservative state transitions")
        print(f"  - Enhanced accuracy algorithms")
        
        print("\nAdvanced Sleep Detection completed successfully")

def main():
    """Main function for advanced sleep detection system"""
    print("="*70)
    print("ADVANCED SLEEP DETECTION SYSTEM")
    print("Enhanced Multi-Modal Approach")
    print("="*70)
    print("ADVANCED FEATURES:")
    print("✓ Multi-factor confidence scoring")
    print("✓ Breathing pattern analysis")
    print("✓ Micro-movement detection")
    print("✓ Conservative state classification")
    print("✓ Addresses 'eyes closed ≠ asleep' problem")
    print("✓ Enhanced accuracy and reliability")
    print("="*70)
    
    try:
        system = AdvancedSleepDetectionSystem()
        system.run_advanced_detection()
    except KeyboardInterrupt:
        print("\n\nAdvanced system stopped by user")
    except Exception as e:
        print(f"\nERROR: {e}")

if __name__ == "__main__":
    main()