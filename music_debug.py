# Music Debug Test
# Save as music_debug.py
# This will test if your music file can be loaded and played

import pygame
import os
import time

def test_music_file():
    print("=== Music System Debug Test ===")
    
    # Check if music file exists
    music_files = ["relaxing_music.mp3", "music.mp3", "sleep_music.mp3", "test.mp3"]
    found_file = None
    
    print("Checking for music files...")
    for filename in music_files:
        if os.path.exists(filename):
            print(f"FOUND: {filename}")
            found_file = filename
            break
        else:
            print(f"NOT FOUND: {filename}")
    
    if not found_file:
        print("\nERROR: No music file found!")
        print("Please ensure you have one of these files:")
        for f in music_files:
            print(f"  - {f}")
        return False
    
    # Test pygame mixer
    print(f"\nTesting pygame with file: {found_file}")
    
    try:
        pygame.mixer.init()
        print("SUCCESS: Pygame mixer initialized")
    except Exception as e:
        print(f"ERROR: Pygame mixer failed - {e}")
        return False
    
    # Test loading the music file
    try:
        pygame.mixer.music.load(found_file)
        print("SUCCESS: Music file loaded successfully")
    except Exception as e:
        print(f"ERROR: Cannot load music file - {e}")
        print("Possible issues:")
        print("  - File is corrupted")
        print("  - Unsupported format (try MP3)")
        print("  - File is being used by another program")
        return False
    
    # Test playing music
    try:
        print("\nTesting music playback...")
        pygame.mixer.music.play()
        print("SUCCESS: Music started playing")
        
        # Play for 5 seconds
        for i in range(5):
            print(f"Playing... {i+1}/5 seconds")
            time.sleep(1)
            
            # Check if still playing
            if not pygame.mixer.music.get_busy():
                print("WARNING: Music stopped unexpectedly")
                break
        
        # Stop music
        pygame.mixer.music.stop()
        print("SUCCESS: Music stopped")
        
    except Exception as e:
        print(f"ERROR: Cannot play music - {e}")
        return False
    
    # Test volume control
    try:
        print("\nTesting volume control...")
        pygame.mixer.music.play()
        
        for volume in [1.0, 0.5, 0.2, 0.8]:
            pygame.mixer.music.set_volume(volume)
            print(f"Volume set to: {volume}")
            time.sleep(1)
        
        pygame.mixer.music.stop()
        print("SUCCESS: Volume control working")
        
    except Exception as e:
        print(f"ERROR: Volume control failed - {e}")
        return False
    
    pygame.mixer.quit()
    print("\n=== Music Test PASSED ===")
    print("Your music system should work in the main program!")
    return True

if __name__ == "__main__":
    test_music_file()