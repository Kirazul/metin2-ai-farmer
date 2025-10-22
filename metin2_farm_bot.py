import os
import sys
import time
import json
import threading
from queue import Queue, Empty
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk
from ultralytics import YOLO
import win32gui
import win32ui
import win32con
from pynput.mouse import Controller as MouseController, Button
from pynput.keyboard import Controller as KeyboardController, Key, Listener
import ctypes
from ctypes import wintypes

# Global lock to prevent actions bleeding between windows
INPUT_LOCK = threading.Lock()

# Set the app to run as administrator
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except:
        return False

if not is_admin():
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, " ".join(sys.argv), None, 1)
    sys.exit()

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class WindowCapture:
    """Handles window capture for a specific game window"""

    def __init__(self, window_handle):
        self.hwnd = window_handle
        self.w = 0
        self.h = 0
        self.update_window_size()

    def update_window_size(self):
        """Update window dimensions"""
        if self.hwnd:
            rect = win32gui.GetClientRect(self.hwnd)
            self.w = rect[2]
            self.h = rect[3]

    def get_screenshot(self):
        """Capture screenshot of the window"""
        try:
            hwndDC = win32gui.GetWindowDC(self.hwnd)
            mfcDC = win32ui.CreateDCFromHandle(hwndDC)
            saveDC = mfcDC.CreateCompatibleDC()

            saveBitMap = win32ui.CreateBitmap()
            saveBitMap.CreateCompatibleBitmap(mfcDC, self.w, self.h)
            saveDC.SelectObject(saveBitMap)

            saveDC.BitBlt((0, 0), (self.w, self.h), mfcDC, (0, 0), win32con.SRCCOPY)

            bmpinfo = saveBitMap.GetInfo()
            bmpstr = saveBitMap.GetBitmapBits(True)
            img = np.frombuffer(bmpstr, dtype=np.uint8).reshape((bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4))

            win32gui.DeleteObject(saveBitMap.GetHandle())
            saveDC.DeleteDC()
            mfcDC.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwndDC)

            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        except:
            return None

    def get_window_position(self):
        """Get window position on screen"""
        rect = win32gui.GetWindowRect(self.hwnd)
        return rect[0], rect[1]

class ImageMatcher:
    """Handles template matching for game state detection"""

    def __init__(self, images_folder="Images"):
        self.images_folder = images_folder
        self.templates = {}
        self.load_templates()

    def load_templates(self):
        """Load all template images"""
        template_files = {
            'selected_metin': 'selectedmetin.png',
            'false_metin': 'falsemetin.png',
            'target': 'target.png',
            'fullhp_metin': 'FullhpMetin.png'
        }

        for key, filename in template_files.items():
            path = os.path.join(self.images_folder, filename)
            if os.path.exists(path):
                self.templates[key] = cv2.imread(path)

    def find_template(self, screenshot, template_name, threshold=0.8):
        """Find template in screenshot"""
        if template_name not in self.templates:
            return False, None

        template = self.templates[template_name]
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            return True, max_loc
        return False, None

class MetinDetector:
    """Handles YOLOv8 metin detection"""

    def __init__(self, model_path="models/best.pt", confidence=0.5):
        # Force GPU usage for faster inference - falls back to CPU if CUDA unavailable
        self.model = YOLO(model_path)
        self.using_cuda = False
        # Try to force CUDA device
        try:
            import torch
            if torch.cuda.is_available():
                self.model.to('cuda')
                self.using_cuda = True
                print(f"[YOLO] Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("[YOLO] CUDA not available - using CPU (slower)")
        except Exception as e:
            print(f"[YOLO] Could not force GPU, using default device: {e}")

        self.confidence = confidence

    def detect(self, image):
        """Detect metins in image"""
        results = self.model(image, conf=self.confidence, verbose=False)
        detections = []

        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'center': [center_x, center_y],
                    'confidence': conf
                })

        return detections

    def find_nearest_metin(self, detections, player_pos):
        """Find nearest metin to player position"""
        if not detections:
            return None

        min_distance = float('inf')
        nearest = None

        for detection in detections:
            center = detection['center']
            distance = np.sqrt((center[0] - player_pos[0])**2 + (center[1] - player_pos[1])**2)

            if distance < min_distance:
                min_distance = distance
                nearest = detection

        return nearest

class BotWorker:
    """Bot worker for a single window"""

    def __init__(self, window_id, hwnd, log_queue, counter_queue, frame_queue, settings):
        self.window_id = window_id
        self.hwnd = hwnd
        self.log_queue = log_queue
        self.counter_queue = counter_queue
        self.frame_queue = frame_queue
        self.settings = settings

        self.running = False
        self.capture = WindowCapture(hwnd)
        model_path = settings.get('model_path', 'models/best.pt')
        self.detector = MetinDetector(model_path=model_path, confidence=settings['model_confidence'])
        self.matcher = ImageMatcher()
        self.mouse = MouseController()
        self.keyboard = KeyboardController()

        self.metin_count = 0
        self.stuck_start_time = None
        self.stuck_recovery_stage = 0
        self.last_action_time = 0

        self.auto_loot_enabled = False
        self.last_loot_time = 0
        # Loot interval is now FIXED at 0.5s for continuous 24/7 looting

        # Track metin selection time for stuck detection
        self.metin_selection_time = 0
        self.max_kill_duration = settings.get('max_kill_duration', 20.0)

        # Click cooldown to prevent rapid clicking issues
        self.last_click_time = 0
        self.click_cooldown = settings.get('click_cooldown', 3.0)

        # Target locking to prevent switching targets during cooldown
        self.locked_target = None
        self.target_lock_time = 0

        # Focus failure tracking
        self.consecutive_focus_failures = 0
        self.last_focus_warning_time = 0

        # Post-kill delay before next target
        self.post_kill_delay = settings.get('post_kill_delay', 0.1)

        # Screenshot settings
        self.save_screenshots = settings.get('save_screenshots', False)
        self.screenshot_folder = "screenshots"
        if self.save_screenshots and not os.path.exists(self.screenshot_folder):
            os.makedirs(self.screenshot_folder)

        # Size filtering for false positives
        self.min_metin_width = settings.get('min_metin_width', 40)
        self.min_metin_height = settings.get('min_metin_height', 40)
        self.min_metin_area = settings.get('min_metin_area', 1600)

        # Color filter sensitivity
        self.color_filter_threshold = settings.get('color_filter_threshold', 5.0)

        # Click timing settings
        self.mouse_movement_delay = settings.get('mouse_movement_delay', 0.5)
        self.click_verification_delay = settings.get('click_verification_delay', 0.3)

        # Rotation timing settings
        self.rotation_key_duration = settings.get('rotation_key_duration', 0.03)
        self.rotation_interval = settings.get('rotation_interval', 0.02)
        self.search_rotation_interval = settings.get('search_rotation_interval', 1.5)

        # Advanced timing settings
        self.main_loop_delay = settings.get('main_loop_delay', 0.01)
        self.key_press_duration = settings.get('key_press_duration', 0.05)
        self.key_press_post_delay = settings.get('key_press_post_delay', 0.01)

    def log(self, message):
        """Send log message"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_queue.put(f"[Win{self.window_id}] [{timestamp}] {message}")

    def focus_window(self):
        """Focus the game window with aggressive multi-strategy approach - MUST be called with INPUT_LOCK held"""
        try:
            # Get current foreground window thread
            foreground_hwnd = win32gui.GetForegroundWindow()
            if foreground_hwnd == self.hwnd:
                return True  # Already focused

            # Strategy 1: Standard SetForegroundWindow
            try:
                # Restore window if minimized
                if win32gui.IsIconic(self.hwnd):
                    win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
                    time.sleep(0.05)

                # Bring to top
                win32gui.BringWindowToTop(self.hwnd)
                time.sleep(0.01)

                # Set foreground
                win32gui.SetForegroundWindow(self.hwnd)
                time.sleep(0.02)

                if win32gui.GetForegroundWindow() == self.hwnd:
                    return True
            except:
                pass

            # Strategy 2: AttachThreadInput bypass (aggressive)
            try:
                # Get thread IDs
                foreground_thread = ctypes.windll.user32.GetWindowThreadProcessId(foreground_hwnd, None)
                target_thread = ctypes.windll.user32.GetWindowThreadProcessId(self.hwnd, None)

                if foreground_thread != target_thread:
                    # Attach input threads to bypass focus restrictions
                    ctypes.windll.user32.AttachThreadInput(foreground_thread, target_thread, True)

                    # Now try to focus
                    win32gui.BringWindowToTop(self.hwnd)
                    win32gui.SetForegroundWindow(self.hwnd)
                    time.sleep(0.02)

                    # Detach threads
                    ctypes.windll.user32.AttachThreadInput(foreground_thread, target_thread, False)

                    if win32gui.GetForegroundWindow() == self.hwnd:
                        return True
            except:
                pass

            # Strategy 3: Simulate ALT keypress (Windows trick to allow focus changes)
            try:
                # This tricks Windows into allowing foreground window changes
                VK_MENU = 0x12  # ALT key
                KEYEVENTF_EXTENDEDKEY = 0x0001
                KEYEVENTF_KEYUP = 0x0002

                ctypes.windll.user32.keybd_event(VK_MENU, 0, KEYEVENTF_EXTENDEDKEY, 0)
                ctypes.windll.user32.keybd_event(VK_MENU, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)

                time.sleep(0.01)
                win32gui.SetForegroundWindow(self.hwnd)
                time.sleep(0.02)

                if win32gui.GetForegroundWindow() == self.hwnd:
                    return True
            except:
                pass

            # Final verification with retries
            for attempt in range(3):
                try:
                    win32gui.SetForegroundWindow(self.hwnd)
                    time.sleep(0.02)
                    if win32gui.GetForegroundWindow() == self.hwnd:
                        return True
                except:
                    pass
                time.sleep(0.01)

            return False
        except Exception as e:
            self.log(f"Focus error: {e}")
            return False

    def click_at(self, x, y, force=False):
        """Click at specific position in window - ATOMIC with lock

        Args:
            x, y: Click coordinates
            force: If True, bypasses click cooldown (for stuck recovery)
        """
        # Check cooldown unless forced
        if not force:
            current_time = time.time()
            time_since_last_click = current_time - self.last_click_time
            if time_since_last_click < self.click_cooldown:
                remaining = self.click_cooldown - time_since_last_click
                self.log(f"⏱️ Click cooldown active ({remaining:.1f}s remaining)")
                return False

        with INPUT_LOCK:  # Ensure no other window can act during this
            # Retry focus up to 3 times with increasing delays
            for attempt in range(3):
                try:
                    if self.focus_window():
                        self.consecutive_focus_failures = 0  # Reset on success
                        break

                    if attempt < 2:
                        time.sleep(0.1 * (attempt + 1))  # 0.1s, 0.2s
                except:
                    if attempt < 2:
                        time.sleep(0.1 * (attempt + 1))
            else:
                # All focus attempts failed
                self.consecutive_focus_failures += 1

                # Only log warning every 30 seconds to reduce spam
                current_time = time.time()
                if current_time - self.last_focus_warning_time > 30:
                    self.log(f"⚠️ Focus failure #{self.consecutive_focus_failures} - Check if game windows are minimized or blocked")
                    self.last_focus_warning_time = current_time

                return False

            try:
                win_x, win_y = self.capture.get_window_position()
                screen_x = win_x + x
                screen_y = win_y + y

                # Move mouse to position
                self.mouse.position = (screen_x, screen_y)
                time.sleep(self.mouse_movement_delay)

                # Click
                self.mouse.click(Button.left, 1)
                time.sleep(0.01)

                # Update last click time
                self.last_click_time = time.time()
                return True
            except Exception as e:
                self.log(f"Click failed: {e}")
                return False

    def press_key(self, key, duration=None):
        """Press a key - ATOMIC with lock, guaranteed release"""
        if duration is None:
            duration = self.key_press_duration
        with INPUT_LOCK:  # Ensure no other window can act during this
            # Retry focus up to 3 times with increasing delays
            for attempt in range(3):
                try:
                    if self.focus_window():
                        self.consecutive_focus_failures = 0  # Reset on success
                        break

                    if attempt < 2:
                        time.sleep(0.05 * (attempt + 1))  # 0.05s, 0.1s
                except:
                    if attempt < 2:
                        time.sleep(0.05 * (attempt + 1))
            else:
                # All focus attempts failed - silently fail for key presses to reduce log spam
                self.consecutive_focus_failures += 1
                return False

            try:
                # Press and release atomically
                if isinstance(key, str):
                    try:
                        self.keyboard.press(key)
                        time.sleep(duration)
                    finally:
                        self.keyboard.release(key)
                else:
                    try:
                        self.keyboard.press(key)
                        time.sleep(duration)
                    finally:
                        self.keyboard.release(key)

                time.sleep(self.key_press_post_delay)  # Configurable delay after release
                return True
            except Exception as e:
                # Silently fail for key presses to reduce log spam
                return False

    def rotate_camera(self):
        """Rotate camera by pressing E"""
        self.press_key('e', self.rotation_key_duration)
        time.sleep(self.rotation_interval)

    def check_game_state(self, screenshot):
        """Check current game state using template matching"""
        selected_found, _ = self.matcher.find_template(
            screenshot, 'selected_metin', self.settings['selected_metin_confidence']
        )
        false_metin_found, _ = self.matcher.find_template(
            screenshot, 'false_metin', self.settings['false_metin_confidence']
        )
        target_found, _ = self.matcher.find_template(
            screenshot, 'target', self.settings['target_confidence']
        )
        fullhp_found, _ = self.matcher.find_template(
            screenshot, 'fullhp_metin', self.settings['fullhp_metin_confidence']
        )

        return {
            'selected_metin': selected_found,
            'false_metin': false_metin_found,
            'target': target_found,
            'fullhp_metin': fullhp_found
        }

    def handle_stuck_condition_1(self, screenshot):
        """Handle stuck condition 1: target.png exists but NOT false_metin.png"""
        self.log("STUCK1: Wrong target - searching for valid metin")

        raw_detections = self.detector.detect(screenshot)
        detections = self.filter_detections_by_size(raw_detections)
        player_pos = [self.capture.w // 2, self.capture.h // 2]

        if detections:
            nearest = self.detector.find_nearest_metin(detections, player_pos)
            if nearest:
                distance = int(np.sqrt((nearest['center'][0] - player_pos[0])**2 + (nearest['center'][1] - player_pos[1])**2))
                self.log(f"STUCK1: Found metin at {distance}px, switching target")
                self.save_detection_screenshot(screenshot, detections, player_pos, "STUCK: WRONG TARGET")
                self.click_at(nearest['center'][0], nearest['center'][1], force=True)  # Force click
                time.sleep(0.5)
                return True

        self.log("STUCK1: No valid metin visible, rotating camera")
        for i in range(5):
            self.rotate_camera()
            time.sleep(0.1)

            screenshot = self.capture.get_screenshot()
            if screenshot is not None:
                raw_detections = self.detector.detect(screenshot)
                detections = self.filter_detections_by_size(raw_detections)
                if detections:
                    nearest = self.detector.find_nearest_metin(detections, player_pos)
                    if nearest:
                        self.log(f"STUCK1: Found metin after rotation #{i+1}")
                        self.save_detection_screenshot(screenshot, detections, player_pos, "STUCK: FOUND AFTER ROTATION")
                        self.click_at(nearest['center'][0], nearest['center'][1], force=True)  # Force click
                        time.sleep(0.5)
                        return True

        self.log("STUCK1: Recovery failed after 5 rotations")
        return False

    def handle_stuck_condition_2(self, screenshot):
        """Handle stuck condition 2: fullhp_metin.png AND false_metin.png exist"""
        current_time = time.time()

        if self.stuck_start_time is None:
            self.stuck_start_time = current_time
            self.stuck_recovery_stage = 0
            self.log("STUCK2: Can't attack metin - starting recovery")
            return False

        stuck_duration = current_time - self.stuck_start_time

        if stuck_duration < 5 and self.stuck_recovery_stage == 0:
            return False

        if stuck_duration >= 5 and self.stuck_recovery_stage == 0:
            game_state = self.check_game_state(screenshot)
            if game_state['fullhp_metin'] and game_state['false_metin']:
                self.log("STUCK2: Stage 1 - Pressing F1")
                self.press_key(Key.f1, 0.1)
                self.stuck_recovery_stage = 1
                self.last_action_time = current_time
            else:
                self.log("STUCK2: Resolved naturally")
                self.stuck_start_time = None
                return True

        if self.stuck_recovery_stage == 1:
            if current_time - self.last_action_time < 5:
                return False

            game_state = self.check_game_state(screenshot)
            if game_state['fullhp_metin'] and game_state['false_metin']:
                self.log("STUCK2: Stage 2 - Trying movement")
                self.stuck_recovery_stage = 2
                self.last_action_time = current_time
            else:
                self.log("STUCK2: Resolved after F1")
                self.stuck_start_time = None
                return True

        if self.stuck_recovery_stage == 2:
            if stuck_duration < 35:
                if current_time - self.last_action_time >= 5:
                    import random
                    keys = ['q', 'z', 's', 'd']
                    random_key = random.choice(keys)
                    self.log(f"STUCK2: Movement - pressing {random_key.upper()}")
                    self.press_key(random_key, 0.5)
                    self.last_action_time = current_time
                return False
            else:
                self.log("STUCK2: Stage 3 - Rotating camera")
                self.stuck_recovery_stage = 3

        if self.stuck_recovery_stage == 3:
            self.rotate_camera()
            time.sleep(0.1)

            raw_detections = self.detector.detect(screenshot)
            detections = self.filter_detections_by_size(raw_detections)
            if detections:
                player_pos = [self.capture.w // 2, self.capture.h // 2]
                nearest = self.detector.find_nearest_metin(detections, player_pos)
                if nearest:
                    self.log("STUCK2: Found new metin, switching target")
                    self.save_detection_screenshot(screenshot, detections, player_pos, "STUCK: RECOVERED")
                    self.click_at(nearest['center'][0], nearest['center'][1], force=True)  # Force click
                    self.stuck_start_time = None
                    self.stuck_recovery_stage = 0
                    return True

        return False

    def auto_loot(self):
        """Continuous auto-loot by pressing W key every 0.5 seconds on both windows"""
        if not self.auto_loot_enabled:
            return

        current_time = time.time()
        if current_time - self.last_loot_time >= 0.5:  # Fixed 0.5s interval for 24/7 looting
            # Use a lightweight key press without excessive locking
            with INPUT_LOCK:
                # Quick focus check without retries
                if win32gui.GetForegroundWindow() == self.hwnd or self.focus_window():
                    try:
                        self.keyboard.press('w')
                        time.sleep(0.01)  # Short press
                        self.keyboard.release('w')
                    except:
                        pass
            self.last_loot_time = current_time

    def filter_detections_by_size(self, detections):
        """Filter out small detections (monsters/objects) that are not metins"""
        filtered = []
        for detection in detections:
            bbox = detection['bbox']
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height

            # Filter by minimum dimensions and area
            if width >= self.min_metin_width and height >= self.min_metin_height and area >= self.min_metin_area:
                filtered.append(detection)

        return filtered

    def has_metin_color(self, screenshot, bbox):
        """Check if detection has metin-like reddish-orange/rust coloring"""
        try:
            x1, y1, x2, y2 = bbox

            # Extract region of interest
            roi = screenshot[y1:y2, x1:x2]

            if roi.size == 0:
                return False

            # Convert to HSV (better for color detection)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

            # Define reddish-orange/rust color ranges for metin stone
            # Orange-red range (Hue: 0-20) - covers red to orange
            lower_red_orange1 = np.array([0, 60, 80])
            upper_red_orange1 = np.array([20, 255, 255])

            # Red wrap-around range (Hue: 160-180) - covers deep red
            lower_red_orange2 = np.array([160, 60, 80])
            upper_red_orange2 = np.array([180, 255, 255])

            # Create masks for both ranges (HSV wraps around at 180)
            mask_orange = cv2.inRange(hsv, lower_red_orange1, upper_red_orange1)
            mask_red = cv2.inRange(hsv, lower_red_orange2, upper_red_orange2)

            # Combine masks
            mask = cv2.bitwise_or(mask_orange, mask_red)

            # Calculate percentage of metin-colored pixels
            total_pixels = roi.shape[0] * roi.shape[1]
            colored_pixels = cv2.countNonZero(mask)
            color_percentage = (colored_pixels / total_pixels) * 100

            # Check against configurable threshold
            return color_percentage >= self.color_filter_threshold

        except Exception as e:
            # If color check fails, allow it through (conservative approach)
            return True

    def filter_detections_by_color(self, screenshot, detections):
        """Filter detections by metin color (reddish-orange/rust glow)"""
        filtered = []
        for detection in detections:
            bbox = detection['bbox']
            if self.has_metin_color(screenshot, bbox):
                filtered.append(detection)
        return filtered

    def save_detection_screenshot(self, screenshot, detections, player_pos, reason=""):
        """Save screenshot with detection drawings"""
        if not self.save_screenshots:
            return

        try:
            # Create annotated frame
            frame = screenshot.copy()

            # Draw player position
            cv2.circle(frame, tuple(player_pos), 5, (0, 255, 0), -1)
            cv2.putText(frame, "PLAYER", (player_pos[0] + 10, player_pos[1]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw all detections
            for det in detections:
                bbox = det['bbox']
                center = det['center']
                conf = det['confidence']

                # Draw bounding box
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

                # Draw center point
                cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)

                # Draw line from player to metin
                cv2.line(frame, tuple(player_pos), tuple(center), (255, 255, 0), 2)

                # Calculate and display distance
                distance = int(np.sqrt((center[0] - player_pos[0])**2 + (center[1] - player_pos[1])**2))

                # Display confidence and distance
                text = f"{conf:.2f} | {distance}px"
                cv2.putText(frame, text, (center[0] - 50, center[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Add timestamp and reason
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            if reason:
                cv2.putText(frame, reason, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            # Save screenshot
            filename = f"Win{self.window_id}_{timestamp}.png"
            filepath = os.path.join(self.screenshot_folder, filename)
            cv2.imwrite(filepath, frame)
            self.log(f"Screenshot saved: {filename}")

        except Exception as e:
            self.log(f"Failed to save screenshot: {e}")

    def run(self):
        """Main bot loop"""
        self.running = True

        # Log CUDA status
        if self.detector.using_cuda:
            self.log("🚀 Bot started with GPU acceleration (CUDA)")
        else:
            self.log("🐌 Bot started with CPU (slower - consider installing CUDA)")

        player_pos = [self.capture.w // 2, self.capture.h // 2]
        waiting_for_kill = False
        last_detection_time = time.time()

        while self.running:
            try:
                screenshot = self.capture.get_screenshot()
                if screenshot is None:
                    time.sleep(0.1)
                    continue

                game_state = self.check_game_state(screenshot)
                raw_detections = self.detector.detect(screenshot)

                # Filter detections by size to remove false positives
                detections = self.filter_detections_by_size(raw_detections)

                # Apply color filter if enabled
                if self.settings.get('use_color_filter', False):
                    detections = self.filter_detections_by_color(screenshot, detections)

                # Send frame to preview (with ALL detections for visualization)
                if not self.frame_queue.full():
                    preview_frame = screenshot.copy()

                    cv2.circle(preview_frame, tuple(player_pos), 5, (0, 255, 0), -1)

                    for det in detections:  # Only show valid detections
                        bbox = det['bbox']
                        center = det['center']

                        cv2.rectangle(preview_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                        cv2.circle(preview_frame, tuple(center), 3, (0, 0, 255), -1)
                        cv2.line(preview_frame, tuple(player_pos), tuple(center), (255, 255, 0), 1)

                        distance = int(np.sqrt((center[0] - player_pos[0])**2 + (center[1] - player_pos[1])**2))
                        cv2.putText(preview_frame, f"{distance}px", (center[0], center[1] - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                    self.frame_queue.put((self.window_id, preview_frame))

                # Handle stuck conditions - ONLY if we're actively fighting
                if waiting_for_kill:
                    # Stuck Condition 1: target present but no false_metin
                    if game_state['target'] and not game_state['false_metin']:
                        self.log("⚠️ Wrong target - switching to valid metin")
                        recovered = self.handle_stuck_condition_1(screenshot)
                        if recovered:
                            # Stuck handler clicked new metin, verify selection
                            self.log("Verifying new target selection...")
                            time.sleep(0.3)
                            screenshot = self.capture.get_screenshot()
                            if screenshot is not None:
                                game_state = self.check_game_state(screenshot)
                                if game_state['selected_metin'] and game_state['false_metin']:
                                    self.log("✓ Target switched successfully")
                                    # Update selection time but KEEP waiting_for_kill = True
                                    self.metin_selection_time = time.time()
                                    self.stuck_start_time = None
                                    self.stuck_recovery_stage = 0
                                else:
                                    self.log("✗ New target not selected properly")
                                    # Reset and try again next loop
                                    waiting_for_kill = False
                                    self.metin_selection_time = 0
                                    self.last_click_time = 0
                                    self.locked_target = None
                                    self.target_lock_time = 0
                        continue

                    # Stuck Condition 2: Only check if 5s have passed since selection
                    if game_state['fullhp_metin'] and game_state['false_metin']:
                        time_since_selection = time.time() - self.metin_selection_time
                        if time_since_selection >= 5.0:
                            self.log(f"⚠️ Can't attack after {time_since_selection:.1f}s - recovering")
                            recovered = self.handle_stuck_condition_2(screenshot)
                            if recovered:
                                # Stuck handler clicked new metin, verify selection
                                self.log("Verifying recovered target selection...")
                                time.sleep(0.3)
                                screenshot = self.capture.get_screenshot()
                                if screenshot is not None:
                                    game_state = self.check_game_state(screenshot)
                                    if game_state['selected_metin'] and game_state['false_metin']:
                                        self.log("✓ Attack recovered successfully")
                                        # Update selection time but KEEP waiting_for_kill = True
                                        self.metin_selection_time = time.time()
                                        self.stuck_start_time = None
                                        self.stuck_recovery_stage = 0
                                    else:
                                        self.log("✗ Recovered target not selected properly")
                                        # Reset and try again next loop
                                        waiting_for_kill = False
                                        self.metin_selection_time = 0
                                        self.stuck_start_time = None
                                        self.stuck_recovery_stage = 0
                                        self.last_click_time = 0
                                        self.locked_target = None
                                        self.target_lock_time = 0
                            continue
                        else:
                            # Too soon after selection, ignore (metin is supposed to have full HP)
                            pass
                else:
                    # NOT in combat - reset all stuck tracking
                    if self.stuck_start_time is not None:
                        self.stuck_start_time = None
                        self.stuck_recovery_stage = 0

                # Check if waiting for kill
                if waiting_for_kill:
                    current_time = time.time()
                    time_since_selection = current_time - self.metin_selection_time

                    # Timeout: If taking too long to kill, abandon and search for new metin
                    if time_since_selection >= self.max_kill_duration:
                        self.log(f"⏱️ TIMEOUT: Metin taking too long ({time_since_selection:.1f}s) - searching new target")
                        self.save_detection_screenshot(screenshot, detections, player_pos, "TIMEOUT: METIN ABANDONED")

                        # Reset and search for new metin
                        waiting_for_kill = False
                        self.metin_selection_time = 0
                        self.stuck_start_time = None
                        self.stuck_recovery_stage = 0
                        self.last_click_time = 0  # Reset click cooldown
                        self.locked_target = None
                        self.target_lock_time = 0

                        # Brief pause before searching new
                        time.sleep(0.2)
                        continue

                    if game_state['selected_metin']:
                        # Still attacking
                        self.auto_loot()
                        time.sleep(0.02)
                        continue
                    else:
                        # Lost selection - metin dead
                        self.log("✓ Metin killed successfully")
                        self.save_detection_screenshot(screenshot, detections, player_pos, "METIN KILLED")

                        # Continuous looting handles loot collection - no burst needed

                        # Reset tracking variables
                        waiting_for_kill = False
                        self.metin_selection_time = 0
                        self.stuck_start_time = None
                        self.stuck_recovery_stage = 0
                        self.locked_target = None
                        self.target_lock_time = 0

                        # IMPORTANT: Reset click cooldown so next metin can be clicked immediately
                        self.last_click_time = 0

                        # Wait before targeting next metin (configurable delay)
                        time.sleep(self.post_kill_delay)
                        continue

                # Look for new metin to select
                if detections:
                    current_time = time.time()

                    # Check if we have a locked target and cooldown is still active
                    if self.locked_target is not None:
                        time_since_lock = current_time - self.target_lock_time
                        if time_since_lock < self.click_cooldown:
                            # Still in cooldown - stick with locked target
                            target_x, target_y, target_distance = self.locked_target
                            remaining_cooldown = self.click_cooldown - time_since_lock
                            # Don't spam logs - only log once per second
                            if int(time_since_lock) != int(time_since_lock - 0.01):
                                self.log(f"⏱️ Locked on metin ({target_distance}px) - cooldown {remaining_cooldown:.1f}s")
                        else:
                            # Cooldown expired - try to click the locked target
                            distance = int(np.sqrt((target_x - player_pos[0])**2 + (target_y - player_pos[1])**2))
                            self.log(f"→ Targeting locked metin ({distance}px away)")

                            # Save screenshot before clicking
                            self.save_detection_screenshot(screenshot, detections, player_pos, "TARGETING LOCKED METIN")

                            # Try to click
                            if self.click_at(target_x, target_y):
                                time.sleep(self.click_verification_delay)

                                screenshot = self.capture.get_screenshot()
                                if screenshot is not None:
                                    game_state = self.check_game_state(screenshot)

                                    if game_state['selected_metin'] and game_state['false_metin']:
                                        self.metin_count += 1
                                        self.counter_queue.put((self.window_id, self.metin_count))
                                        self.log(f"✓ Metin #{self.metin_count} selected")
                                        self.save_detection_screenshot(screenshot, detections, player_pos, f"SELECTED #{self.metin_count}")

                                        # Record selection time and clear target lock
                                        self.metin_selection_time = time.time()
                                        self.locked_target = None
                                        self.target_lock_time = 0
                                        waiting_for_kill = True
                                    else:
                                        self.log("✗ Selection failed - trying again")
                                        self.save_detection_screenshot(screenshot, detections, player_pos, "SELECTION FAILED")
                                        # Clear lock to find new target
                                        self.locked_target = None
                                        self.target_lock_time = 0
                            else:
                                # Click failed (shouldn't happen since cooldown expired)
                                self.locked_target = None
                                self.target_lock_time = 0
                    else:
                        # No locked target - find nearest metin
                        nearest = self.detector.find_nearest_metin(detections, player_pos)

                        if nearest:
                            distance = int(np.sqrt((nearest['center'][0] - player_pos[0])**2 + (nearest['center'][1] - player_pos[1])**2))

                            # Check if we can click immediately or need to lock target
                            time_since_last_click = current_time - self.last_click_time
                            if time_since_last_click >= self.click_cooldown:
                                # Cooldown expired - click immediately
                                self.log(f"→ Targeting metin ({distance}px away)")

                                # Save screenshot before clicking
                                self.save_detection_screenshot(screenshot, detections, player_pos, "TARGETING METIN")

                                # Try to click
                                if self.click_at(nearest['center'][0], nearest['center'][1]):
                                    time.sleep(self.click_verification_delay)

                                    screenshot = self.capture.get_screenshot()
                                    if screenshot is not None:
                                        game_state = self.check_game_state(screenshot)

                                        if game_state['selected_metin'] and game_state['false_metin']:
                                            self.metin_count += 1
                                            self.counter_queue.put((self.window_id, self.metin_count))
                                            self.log(f"✓ Metin #{self.metin_count} selected")
                                            self.save_detection_screenshot(screenshot, detections, player_pos, f"SELECTED #{self.metin_count}")

                                            # Record selection time
                                            self.metin_selection_time = time.time()
                                            waiting_for_kill = True
                                        else:
                                            self.log("✗ Selection failed - trying again")
                                            self.save_detection_screenshot(screenshot, detections, player_pos, "SELECTION FAILED")
                            else:
                                # Cooldown active - lock this target and wait
                                self.locked_target = (nearest['center'][0], nearest['center'][1], distance)
                                self.target_lock_time = current_time
                                remaining_cooldown = self.click_cooldown - time_since_last_click
                                self.log(f"🎯 Target locked ({distance}px) - waiting {remaining_cooldown:.1f}s cooldown")
                else:
                    current_time = time.time()
                    if current_time - last_detection_time > self.search_rotation_interval:
                        self.log("🔍 Searching for metin")
                        self.rotate_camera()
                        self.auto_loot()
                        last_detection_time = current_time

                self.auto_loot()
                time.sleep(self.main_loop_delay)

            except Exception as e:
                self.log(f"Error in main loop: {e}")
                time.sleep(0.1)

        self.log("Bot stopped")

    def stop(self):
        """Stop the bot"""
        self.running = False

class Metin2FarmBot(ctk.CTk):
    """Main GUI application"""

    def __init__(self):
        super().__init__()

        self.title("Metin2 Farm Bot v2.0")
        self.geometry("350x500")
        self.resizable(False, False)
        self.attributes('-topmost', True)

        self.settings = self.load_settings()

        self.workers = {}
        self.worker_threads = {}
        self.window_handles = {1: None, 2: None}

        self.log_queue = Queue()
        self.counter_queue = Queue()
        self.frame_queues = {1: Queue(maxsize=2), 2: Queue(maxsize=2)}
        self.counters = {1: 0, 2: 0}

        # Hotkey listener
        self.hotkey_listener = None
        self.hotkey_enabled = False

        self.create_gui()

        self.update_logs()
        self.update_counters()
        self.update_previews()

        # Start hotkey listener if enabled
        if self.settings.get('enable_hotkey', False):
            self.start_hotkey_listener()

    def load_settings(self):
        """Load settings from file"""
        default_settings = {
            # Model settings
            'model_path': 'models/best.pt',

            # Detection confidence
            'model_confidence': 0.5,
            'selected_metin_confidence': 0.8,
            'false_metin_confidence': 0.8,
            'target_confidence': 0.8,
            'fullhp_metin_confidence': 0.8,

            # Features
            'auto_loot': False,
            'save_screenshots': True,
            'use_color_filter': False,
            'color_filter_threshold': 5.0,

            # Hotkey settings
            'enable_hotkey': False,
            'hotkey': 'f9',

            # Size filtering
            'min_metin_width': 40,
            'min_metin_height': 40,
            'min_metin_area': 1600,

            # Combat timing
            'max_kill_duration': 20.0,
            'post_kill_delay': 0.1,

            # Click timing
            'click_cooldown': 3.0,
            'mouse_movement_delay': 0.5,
            'click_verification_delay': 0.3,

            # Rotation timing
            'rotation_key_duration': 0.03,
            'rotation_interval': 0.02,
            'search_rotation_interval': 1.5,

            # Advanced timing
            'main_loop_delay': 0.01,
            'key_press_duration': 0.05,
            'key_press_post_delay': 0.01
        }

        try:
            if os.path.exists('bot_settings.json'):
                with open('bot_settings.json', 'r') as f:
                    loaded = json.load(f)
                    default_settings.update(loaded)
        except:
            pass

        return default_settings

    def save_settings(self):
        """Save settings to file"""
        try:
            with open('bot_settings.json', 'w') as f:
                json.dump(self.settings, f, indent=4)
        except:
            pass

    def get_available_models(self):
        """Get list of available YOLO models from models folder"""
        models_folder = "models"
        if not os.path.exists(models_folder):
            return ["models/best.pt"]

        models = []
        for file in os.listdir(models_folder):
            if file.endswith('.pt'):
                models.append(f"models/{file}")

        return models if models else ["models/best.pt"]

    def start_hotkey_listener(self):
        """Start the global hotkey listener"""
        if self.hotkey_listener is not None:
            return

        def on_press(key):
            try:
                # Get the configured hotkey
                hotkey = self.settings.get('hotkey', 'f9').lower()

                # Convert key to string
                key_str = None
                if hasattr(key, 'name'):
                    key_str = key.name.lower()
                elif hasattr(key, 'char'):
                    key_str = key.char.lower() if key.char else None

                # Check if pressed key matches configured hotkey
                if key_str == hotkey:
                    # Toggle bot state
                    self.after(0, self.toggle_bot_from_hotkey)

            except Exception as e:
                pass

        # Start listener in background thread
        self.hotkey_listener = Listener(on_press=on_press)
        self.hotkey_listener.daemon = True
        self.hotkey_listener.start()
        self.hotkey_enabled = True
        self.log_message(f"Hotkey listener started ({self.settings.get('hotkey', 'f9').upper()} to start/stop)")

    def stop_hotkey_listener(self):
        """Stop the global hotkey listener"""
        if self.hotkey_listener is not None:
            self.hotkey_listener.stop()
            self.hotkey_listener = None
            self.hotkey_enabled = False
            self.log_message("Hotkey listener stopped")

    def toggle_hotkey_listener(self):
        """Toggle hotkey listener on/off"""
        if self.enable_hotkey_var.get():
            self.start_hotkey_listener()
        else:
            self.stop_hotkey_listener()

    def toggle_bot_from_hotkey(self):
        """Toggle bot state (called from hotkey)"""
        if len(self.workers) > 0:
            # Bot is running, stop it
            self.stop_bot()
        else:
            # Bot is stopped, start it
            self.start_bot()

    def create_gui(self):
        """Create the GUI"""
        self.tabview = ctk.CTkTabview(self, width=330, height=480)
        self.tabview.pack(padx=10, pady=10)

        self.tab1 = self.tabview.add("Windows")
        self.tab2 = self.tabview.add("Preview")
        self.tab3 = self.tabview.add("Logs")
        self.tab4 = self.tabview.add("Settings")

        self.create_window_tab()
        self.create_preview_tab()
        self.create_logs_tab()
        self.create_settings_tab()

    def create_window_tab(self):
        """Create window selection tab"""
        frame1 = ctk.CTkFrame(self.tab1)
        frame1.pack(padx=10, pady=5, fill="x")

        ctk.CTkLabel(frame1, text="Window 1: (Required)", font=("Arial", 12, "bold")).pack(anchor="w", padx=5, pady=2)

        self.window1_var = ctk.StringVar(value="Select Window...")
        self.window1_dropdown = ctk.CTkComboBox(frame1, variable=self.window1_var,
                                                 values=self.get_window_list(), width=300)
        self.window1_dropdown.pack(padx=5, pady=2)

        counter_frame1 = ctk.CTkFrame(frame1)
        counter_frame1.pack(padx=5, pady=2, fill="x")
        ctk.CTkLabel(counter_frame1, text="Metins:").pack(side="left", padx=5)
        self.counter1_label = ctk.CTkLabel(counter_frame1, text="0", font=("Arial", 14, "bold"))
        self.counter1_label.pack(side="left", padx=5)

        frame2 = ctk.CTkFrame(self.tab1)
        frame2.pack(padx=10, pady=5, fill="x")

        ctk.CTkLabel(frame2, text="Window 2: (Optional)", font=("Arial", 12, "bold")).pack(anchor="w", padx=5, pady=2)

        self.window2_var = ctk.StringVar(value="Select Window...")
        self.window2_dropdown = ctk.CTkComboBox(frame2, variable=self.window2_var,
                                                 values=self.get_window_list(), width=300)
        self.window2_dropdown.pack(padx=5, pady=2)

        counter_frame2 = ctk.CTkFrame(frame2)
        counter_frame2.pack(padx=5, pady=2, fill="x")
        ctk.CTkLabel(counter_frame2, text="Metins:").pack(side="left", padx=5)
        self.counter2_label = ctk.CTkLabel(counter_frame2, text="0", font=("Arial", 14, "bold"))
        self.counter2_label.pack(side="left", padx=5)

        refresh_btn = ctk.CTkButton(self.tab1, text="Refresh Windows", command=self.refresh_windows)
        refresh_btn.pack(padx=10, pady=5)

        control_frame = ctk.CTkFrame(self.tab1)
        control_frame.pack(padx=10, pady=10, fill="x")

        self.start_btn = ctk.CTkButton(control_frame, text="Start", command=self.start_bot,
                                       fg_color="green", hover_color="darkgreen", width=140)
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ctk.CTkButton(control_frame, text="Stop", command=self.stop_bot,
                                      fg_color="red", hover_color="darkred", width=140, state="disabled")
        self.stop_btn.pack(side="left", padx=5)

    def create_preview_tab(self):
        """Create live preview tab"""
        frame1 = ctk.CTkFrame(self.tab2)
        frame1.pack(padx=5, pady=5, fill="both", expand=True)

        ctk.CTkLabel(frame1, text="Window 1 Preview", font=("Arial", 10, "bold")).pack()
        self.preview1_label = ctk.CTkLabel(frame1, text="No preview")
        self.preview1_label.pack(padx=5, pady=5)

        frame2 = ctk.CTkFrame(self.tab2)
        frame2.pack(padx=5, pady=5, fill="both", expand=True)

        ctk.CTkLabel(frame2, text="Window 2 Preview", font=("Arial", 10, "bold")).pack()
        self.preview2_label = ctk.CTkLabel(frame2, text="No preview")
        self.preview2_label.pack(padx=5, pady=5)

    def create_logs_tab(self):
        """Create logs tab"""
        self.log_text = ctk.CTkTextbox(self.tab3, width=310, height=440)
        self.log_text.pack(padx=5, pady=5)

        clear_btn = ctk.CTkButton(self.tab3, text="Clear Logs", command=self.clear_logs, width=100)
        clear_btn.pack(pady=5)

    def create_settings_tab(self):
        """Create reorganized settings tab with all timing controls"""
        scroll_frame = ctk.CTkScrollableFrame(self.tab4, width=310, height=400)
        scroll_frame.pack(padx=5, pady=5)

        # ===== MODEL SELECTION =====
        ctk.CTkLabel(scroll_frame, text="═══ MODEL SELECTION ═══",
                    font=("Arial", 12, "bold"), text_color="#4A90E2").pack(anchor="w", padx=5, pady=(5, 5))

        # Model selector
        model_frame = ctk.CTkFrame(scroll_frame)
        model_frame.pack(padx=5, pady=5, fill="x")

        ctk.CTkLabel(model_frame, text="YOLO Model:").pack(side="left", padx=5)

        self.model_path_var = ctk.StringVar(value=self.settings.get('model_path', 'models/best.pt'))
        available_models = self.get_available_models()
        self.model_dropdown = ctk.CTkComboBox(
            model_frame,
            variable=self.model_path_var,
            values=available_models,
            width=180,
            state="readonly"
        )
        self.model_dropdown.pack(side="left", padx=5)

        # Refresh models button
        refresh_models_btn = ctk.CTkButton(
            model_frame,
            text="↻",
            width=30,
            command=self.refresh_models
        )
        refresh_models_btn.pack(side="left", padx=2)

        # ===== DETECTION CONFIDENCE =====
        ctk.CTkLabel(scroll_frame, text="═══ DETECTION CONFIDENCE ═══",
                    font=("Arial", 12, "bold"), text_color="#4A90E2").pack(anchor="w", padx=5, pady=(5, 5))

        # Model Confidence
        ctk.CTkLabel(scroll_frame, text="Model Confidence:").pack(anchor="w", padx=5, pady=(5, 0))
        self.model_conf_var = ctk.DoubleVar(value=self.settings['model_confidence'])
        ctk.CTkSlider(scroll_frame, from_=0.1, to=1.0, variable=self.model_conf_var, width=280).pack(padx=5, pady=2)
        self.model_conf_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings['model_confidence']:.2f}")
        self.model_conf_label.pack(padx=5, pady=2)

        # Selected Metin Confidence
        ctk.CTkLabel(scroll_frame, text="Selected Metin Confidence:").pack(anchor="w", padx=5, pady=(5, 0))
        self.selected_conf_var = ctk.DoubleVar(value=self.settings['selected_metin_confidence'])
        ctk.CTkSlider(scroll_frame, from_=0.1, to=1.0, variable=self.selected_conf_var, width=280).pack(padx=5, pady=2)
        self.selected_conf_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings['selected_metin_confidence']:.2f}")
        self.selected_conf_label.pack(padx=5, pady=2)

        # False Metin Confidence
        ctk.CTkLabel(scroll_frame, text="False Metin Confidence:").pack(anchor="w", padx=5, pady=(5, 0))
        self.false_conf_var = ctk.DoubleVar(value=self.settings['false_metin_confidence'])
        ctk.CTkSlider(scroll_frame, from_=0.1, to=1.0, variable=self.false_conf_var, width=280).pack(padx=5, pady=2)
        self.false_conf_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings['false_metin_confidence']:.2f}")
        self.false_conf_label.pack(padx=5, pady=2)

        # Target Confidence
        ctk.CTkLabel(scroll_frame, text="Target Confidence:").pack(anchor="w", padx=5, pady=(5, 0))
        self.target_conf_var = ctk.DoubleVar(value=self.settings['target_confidence'])
        ctk.CTkSlider(scroll_frame, from_=0.1, to=1.0, variable=self.target_conf_var, width=280).pack(padx=5, pady=2)
        self.target_conf_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings['target_confidence']:.2f}")
        self.target_conf_label.pack(padx=5, pady=2)

        # Full HP Metin Confidence
        ctk.CTkLabel(scroll_frame, text="Full HP Metin Confidence:").pack(anchor="w", padx=5, pady=(5, 0))
        self.fullhp_conf_var = ctk.DoubleVar(value=self.settings['fullhp_metin_confidence'])
        ctk.CTkSlider(scroll_frame, from_=0.1, to=1.0, variable=self.fullhp_conf_var, width=280).pack(padx=5, pady=2)
        self.fullhp_conf_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings['fullhp_metin_confidence']:.2f}")
        self.fullhp_conf_label.pack(padx=5, pady=2)

        # ===== FEATURES =====
        ctk.CTkLabel(scroll_frame, text="═══ FEATURES ═══",
                    font=("Arial", 12, "bold"), text_color="#4A90E2").pack(anchor="w", padx=5, pady=(15, 5))

        self.auto_loot_var = ctk.BooleanVar(value=self.settings['auto_loot'])
        ctk.CTkCheckBox(scroll_frame, text="Auto Loot (W spam)", variable=self.auto_loot_var).pack(anchor="w", padx=5, pady=2)

        self.save_screenshots_var = ctk.BooleanVar(value=self.settings.get('save_screenshots', True))
        ctk.CTkCheckBox(scroll_frame, text="Save Detection Screenshots", variable=self.save_screenshots_var).pack(anchor="w", padx=5, pady=2)

        self.use_color_filter_var = ctk.BooleanVar(value=self.settings.get('use_color_filter', False))
        ctk.CTkCheckBox(scroll_frame, text="Enable Color Filter (Reddish-Orange Metin Stone)", variable=self.use_color_filter_var).pack(anchor="w", padx=5, pady=2)

        # Color filter threshold slider (only visible when filter is enabled)
        color_filter_frame = ctk.CTkFrame(scroll_frame)
        color_filter_frame.pack(padx=5, pady=5, fill="x")

        ctk.CTkLabel(color_filter_frame, text="Color Filter Sensitivity (%):").pack(anchor="w", padx=5, pady=(5, 0))
        ctk.CTkLabel(color_filter_frame, text="Lower = More strict (fewer detections)",
                    font=("Arial", 9), text_color="gray").pack(anchor="w", padx=5)

        self.color_filter_threshold_var = ctk.DoubleVar(value=self.settings.get('color_filter_threshold', 5.0))
        ctk.CTkSlider(color_filter_frame, from_=1.0, to=20.0, variable=self.color_filter_threshold_var, width=280, number_of_steps=19).pack(padx=5, pady=2)
        self.color_filter_threshold_label = ctk.CTkLabel(color_filter_frame, text=f"{self.settings.get('color_filter_threshold', 5.0):.1f}%")
        self.color_filter_threshold_label.pack(padx=5, pady=2)

        # ===== HOTKEY SETTINGS =====
        ctk.CTkLabel(scroll_frame, text="═══ HOTKEY SETTINGS ═══",
                    font=("Arial", 12, "bold"), text_color="#4A90E2").pack(anchor="w", padx=5, pady=(15, 5))

        self.enable_hotkey_var = ctk.BooleanVar(value=self.settings.get('enable_hotkey', False))
        hotkey_checkbox = ctk.CTkCheckBox(scroll_frame, text="Enable Start/Stop Hotkey",
                                          variable=self.enable_hotkey_var,
                                          command=self.toggle_hotkey_listener)
        hotkey_checkbox.pack(anchor="w", padx=5, pady=2)

        # Hotkey input
        hotkey_frame = ctk.CTkFrame(scroll_frame)
        hotkey_frame.pack(padx=5, pady=5, fill="x")

        ctk.CTkLabel(hotkey_frame, text="Hotkey:").pack(side="left", padx=5)
        self.hotkey_var = ctk.StringVar(value=self.settings.get('hotkey', 'f9'))
        self.hotkey_entry = ctk.CTkEntry(hotkey_frame, textvariable=self.hotkey_var, width=100)
        self.hotkey_entry.pack(side="left", padx=5)

        ctk.CTkLabel(hotkey_frame, text="(e.g., f9, f10, f11)",
                    font=("Arial", 9), text_color="gray").pack(side="left", padx=5)

        # ===== SIZE FILTER =====
        ctk.CTkLabel(scroll_frame, text="═══ SIZE FILTER (False Positive Protection) ═══",
                    font=("Arial", 12, "bold"), text_color="#4A90E2").pack(anchor="w", padx=5, pady=(15, 5))

        # Min Width
        ctk.CTkLabel(scroll_frame, text="Min Width (px):").pack(anchor="w", padx=5, pady=(5, 0))
        self.min_width_var = ctk.IntVar(value=self.settings.get('min_metin_width', 40))
        ctk.CTkSlider(scroll_frame, from_=20, to=100, variable=self.min_width_var, width=280, number_of_steps=80).pack(padx=5, pady=2)
        self.min_width_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings.get('min_metin_width', 40)}px")
        self.min_width_label.pack(padx=5, pady=2)

        # Min Height
        ctk.CTkLabel(scroll_frame, text="Min Height (px):").pack(anchor="w", padx=5, pady=(5, 0))
        self.min_height_var = ctk.IntVar(value=self.settings.get('min_metin_height', 40))
        ctk.CTkSlider(scroll_frame, from_=20, to=100, variable=self.min_height_var, width=280, number_of_steps=80).pack(padx=5, pady=2)
        self.min_height_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings.get('min_metin_height', 40)}px")
        self.min_height_label.pack(padx=5, pady=2)

        # ===== COMBAT TIMING =====
        ctk.CTkLabel(scroll_frame, text="═══ COMBAT TIMING ═══",
                    font=("Arial", 12, "bold"), text_color="#4A90E2").pack(anchor="w", padx=5, pady=(15, 5))

        # Max Kill Duration
        ctk.CTkLabel(scroll_frame, text="Max Kill Duration (seconds):").pack(anchor="w", padx=5, pady=(5, 0))
        self.max_kill_duration_var = ctk.DoubleVar(value=self.settings.get('max_kill_duration', 20.0))
        ctk.CTkSlider(scroll_frame, from_=10.0, to=60.0, variable=self.max_kill_duration_var, width=280, number_of_steps=50).pack(padx=5, pady=2)
        self.max_kill_duration_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings.get('max_kill_duration', 20.0):.0f}s")
        self.max_kill_duration_label.pack(padx=5, pady=2)

        # Post Kill Delay
        ctk.CTkLabel(scroll_frame, text="Delay After Kill (seconds):").pack(anchor="w", padx=5, pady=(5, 0))
        self.post_kill_delay_var = ctk.DoubleVar(value=self.settings.get('post_kill_delay', 0.1))
        ctk.CTkSlider(scroll_frame, from_=0.05, to=3.0, variable=self.post_kill_delay_var, width=280, number_of_steps=59).pack(padx=5, pady=2)
        self.post_kill_delay_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings.get('post_kill_delay', 0.1):.2f}s")
        self.post_kill_delay_label.pack(padx=5, pady=2)

        # ===== CLICK TIMING =====
        ctk.CTkLabel(scroll_frame, text="═══ CLICK TIMING ═══",
                    font=("Arial", 12, "bold"), text_color="#4A90E2").pack(anchor="w", padx=5, pady=(15, 5))

        # Click Cooldown
        ctk.CTkLabel(scroll_frame, text="Click Cooldown (seconds):").pack(anchor="w", padx=5, pady=(5, 0))
        self.click_cooldown_var = ctk.DoubleVar(value=self.settings.get('click_cooldown', 3.0))
        ctk.CTkSlider(scroll_frame, from_=1.0, to=10.0, variable=self.click_cooldown_var, width=280, number_of_steps=90).pack(padx=5, pady=2)
        self.click_cooldown_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings.get('click_cooldown', 3.0):.1f}s")
        self.click_cooldown_label.pack(padx=5, pady=2)

        # Mouse Movement Delay
        ctk.CTkLabel(scroll_frame, text="Mouse Movement Delay (seconds):").pack(anchor="w", padx=5, pady=(5, 0))
        self.mouse_movement_delay_var = ctk.DoubleVar(value=self.settings.get('mouse_movement_delay', 0.5))
        ctk.CTkSlider(scroll_frame, from_=0.1, to=2.0, variable=self.mouse_movement_delay_var, width=280, number_of_steps=19).pack(padx=5, pady=2)
        self.mouse_movement_delay_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings.get('mouse_movement_delay', 0.5):.2f}s")
        self.mouse_movement_delay_label.pack(padx=5, pady=2)

        # Click Verification Delay
        ctk.CTkLabel(scroll_frame, text="Click Verification Delay (seconds):").pack(anchor="w", padx=5, pady=(5, 0))
        self.click_verification_delay_var = ctk.DoubleVar(value=self.settings.get('click_verification_delay', 0.3))
        ctk.CTkSlider(scroll_frame, from_=0.1, to=2.0, variable=self.click_verification_delay_var, width=280, number_of_steps=19).pack(padx=5, pady=2)
        self.click_verification_delay_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings.get('click_verification_delay', 0.3):.2f}s")
        self.click_verification_delay_label.pack(padx=5, pady=2)

        # ===== ROTATION TIMING =====
        ctk.CTkLabel(scroll_frame, text="═══ ROTATION TIMING ═══",
                    font=("Arial", 12, "bold"), text_color="#4A90E2").pack(anchor="w", padx=5, pady=(15, 5))

        # Rotation Key Duration
        ctk.CTkLabel(scroll_frame, text="Rotation Key Duration (seconds):").pack(anchor="w", padx=5, pady=(5, 0))
        self.rotation_key_duration_var = ctk.DoubleVar(value=self.settings.get('rotation_key_duration', 0.03))
        ctk.CTkSlider(scroll_frame, from_=0.01, to=0.2, variable=self.rotation_key_duration_var, width=280, number_of_steps=19).pack(padx=5, pady=2)
        self.rotation_key_duration_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings.get('rotation_key_duration', 0.03):.3f}s")
        self.rotation_key_duration_label.pack(padx=5, pady=2)

        # Rotation Interval
        ctk.CTkLabel(scroll_frame, text="Rotation Interval (seconds):").pack(anchor="w", padx=5, pady=(5, 0))
        self.rotation_interval_var = ctk.DoubleVar(value=self.settings.get('rotation_interval', 0.02))
        ctk.CTkSlider(scroll_frame, from_=0.01, to=0.5, variable=self.rotation_interval_var, width=280, number_of_steps=49).pack(padx=5, pady=2)
        self.rotation_interval_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings.get('rotation_interval', 0.02):.3f}s")
        self.rotation_interval_label.pack(padx=5, pady=2)

        # Search Rotation Interval
        ctk.CTkLabel(scroll_frame, text="Search Rotation Interval (seconds):").pack(anchor="w", padx=5, pady=(5, 0))
        self.search_rotation_interval_var = ctk.DoubleVar(value=self.settings.get('search_rotation_interval', 1.5))
        ctk.CTkSlider(scroll_frame, from_=0.5, to=5.0, variable=self.search_rotation_interval_var, width=280, number_of_steps=45).pack(padx=5, pady=2)
        self.search_rotation_interval_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings.get('search_rotation_interval', 1.5):.1f}s")
        self.search_rotation_interval_label.pack(padx=5, pady=2)

        # ===== LOOTING TIMING =====
        ctk.CTkLabel(scroll_frame, text="═══ LOOTING ═══",
                    font=("Arial", 12, "bold"), text_color="#4A90E2").pack(anchor="w", padx=5, pady=(15, 5))

        # Info label about fixed looting
        info_label = ctk.CTkLabel(scroll_frame,
                                 text="Looting is FIXED at 0.5 seconds for 24/7 continuous looting on all windows.\nNo burst looting - continuous loot handles all item collection.",
                                 font=("Arial", 10),
                                 text_color="#FFA500",
                                 wraplength=280,
                                 justify="left")
        info_label.pack(anchor="w", padx=5, pady=5)

        # ===== ADVANCED TIMING =====
        ctk.CTkLabel(scroll_frame, text="═══ ADVANCED TIMING ═══",
                    font=("Arial", 12, "bold"), text_color="#4A90E2").pack(anchor="w", padx=5, pady=(15, 5))

        # Main Loop Delay
        ctk.CTkLabel(scroll_frame, text="Main Loop Delay (seconds):").pack(anchor="w", padx=5, pady=(5, 0))
        self.main_loop_delay_var = ctk.DoubleVar(value=self.settings.get('main_loop_delay', 0.01))
        ctk.CTkSlider(scroll_frame, from_=0.001, to=0.1, variable=self.main_loop_delay_var, width=280, number_of_steps=99).pack(padx=5, pady=2)
        self.main_loop_delay_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings.get('main_loop_delay', 0.01):.3f}s")
        self.main_loop_delay_label.pack(padx=5, pady=2)

        # Key Press Duration
        ctk.CTkLabel(scroll_frame, text="Key Press Duration (seconds):").pack(anchor="w", padx=5, pady=(5, 0))
        self.key_press_duration_var = ctk.DoubleVar(value=self.settings.get('key_press_duration', 0.05))
        ctk.CTkSlider(scroll_frame, from_=0.01, to=0.5, variable=self.key_press_duration_var, width=280, number_of_steps=49).pack(padx=5, pady=2)
        self.key_press_duration_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings.get('key_press_duration', 0.05):.3f}s")
        self.key_press_duration_label.pack(padx=5, pady=2)

        # Key Press Post Delay
        ctk.CTkLabel(scroll_frame, text="Key Press Post-Delay (seconds):").pack(anchor="w", padx=5, pady=(5, 0))
        self.key_press_post_delay_var = ctk.DoubleVar(value=self.settings.get('key_press_post_delay', 0.01))
        ctk.CTkSlider(scroll_frame, from_=0.001, to=0.1, variable=self.key_press_post_delay_var, width=280, number_of_steps=99).pack(padx=5, pady=2)
        self.key_press_post_delay_label = ctk.CTkLabel(scroll_frame, text=f"{self.settings.get('key_press_post_delay', 0.01):.3f}s")
        self.key_press_post_delay_label.pack(padx=5, pady=2)

        # ===== SAVE BUTTON =====
        ctk.CTkButton(scroll_frame, text="💾 Save All Settings", command=self.save_settings_gui,
                     width=250, height=40, font=("Arial", 13, "bold"),
                     fg_color="#2ECC71", hover_color="#27AE60").pack(pady=15)

        # ===== UPDATE LABELS FUNCTION =====
        def update_labels(*args):
            self.model_conf_label.configure(text=f"{self.model_conf_var.get():.2f}")
            self.selected_conf_label.configure(text=f"{self.selected_conf_var.get():.2f}")
            self.false_conf_label.configure(text=f"{self.false_conf_var.get():.2f}")
            self.target_conf_label.configure(text=f"{self.target_conf_var.get():.2f}")
            self.fullhp_conf_label.configure(text=f"{self.fullhp_conf_var.get():.2f}")
            self.color_filter_threshold_label.configure(text=f"{self.color_filter_threshold_var.get():.1f}%")
            self.min_width_label.configure(text=f"{self.min_width_var.get()}px")
            self.min_height_label.configure(text=f"{self.min_height_var.get()}px")
            self.max_kill_duration_label.configure(text=f"{self.max_kill_duration_var.get():.0f}s")
            self.post_kill_delay_label.configure(text=f"{self.post_kill_delay_var.get():.2f}s")
            self.click_cooldown_label.configure(text=f"{self.click_cooldown_var.get():.1f}s")
            self.mouse_movement_delay_label.configure(text=f"{self.mouse_movement_delay_var.get():.2f}s")
            self.click_verification_delay_label.configure(text=f"{self.click_verification_delay_var.get():.2f}s")
            self.rotation_key_duration_label.configure(text=f"{self.rotation_key_duration_var.get():.3f}s")
            self.rotation_interval_label.configure(text=f"{self.rotation_interval_var.get():.3f}s")
            self.search_rotation_interval_label.configure(text=f"{self.search_rotation_interval_var.get():.1f}s")
            self.main_loop_delay_label.configure(text=f"{self.main_loop_delay_var.get():.3f}s")
            self.key_press_duration_label.configure(text=f"{self.key_press_duration_var.get():.3f}s")
            self.key_press_post_delay_label.configure(text=f"{self.key_press_post_delay_var.get():.3f}s")

        # Trace all variables
        self.model_conf_var.trace_add("write", update_labels)
        self.selected_conf_var.trace_add("write", update_labels)
        self.false_conf_var.trace_add("write", update_labels)
        self.target_conf_var.trace_add("write", update_labels)
        self.fullhp_conf_var.trace_add("write", update_labels)
        self.min_width_var.trace_add("write", update_labels)
        self.min_height_var.trace_add("write", update_labels)
        self.max_kill_duration_var.trace_add("write", update_labels)
        self.post_kill_delay_var.trace_add("write", update_labels)
        self.click_cooldown_var.trace_add("write", update_labels)
        self.mouse_movement_delay_var.trace_add("write", update_labels)
        self.click_verification_delay_var.trace_add("write", update_labels)
        self.rotation_key_duration_var.trace_add("write", update_labels)
        self.rotation_interval_var.trace_add("write", update_labels)
        self.search_rotation_interval_var.trace_add("write", update_labels)
        self.main_loop_delay_var.trace_add("write", update_labels)
        self.key_press_duration_var.trace_add("write", update_labels)
        self.key_press_post_delay_var.trace_add("write", update_labels)
        self.color_filter_threshold_var.trace_add("write", update_labels)
    def get_window_list(self):
        """Get list of all windows"""
        windows = []

        def callback(hwnd, extra):
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if title:
                    windows.append(f"{title} [{hwnd}]")

        win32gui.EnumWindows(callback, None)
        return windows

    def refresh_windows(self):
        """Refresh window list"""
        windows = self.get_window_list()
        self.window1_dropdown.configure(values=windows)
        self.window2_dropdown.configure(values=windows)

    def refresh_models(self):
        """Refresh model list"""
        models = self.get_available_models()
        self.model_dropdown.configure(values=models)
        self.log_message("Model list refreshed")

    def start_bot(self):
        """Start the bot"""
        win1_text = self.window1_var.get()
        win2_text = self.window2_var.get()

        # Check if at least Window 1 is selected
        if win1_text == "Select Window...":
            self.log_message("Please select at least Window 1")
            return

        # Determine which windows to run
        windows_to_run = []

        try:
            # Window 1 is always required
            hwnd1 = int(win1_text.split("[")[1].split("]")[0])
            self.window_handles[1] = hwnd1
            windows_to_run.append(1)

            # Window 2 is optional
            if win2_text != "Select Window...":
                hwnd2 = int(win2_text.split("[")[1].split("]")[0])
                self.window_handles[2] = hwnd2
                windows_to_run.append(2)
            else:
                self.window_handles[2] = None
        except:
            self.log_message("Invalid window selection")
            return

        self.settings['model_confidence'] = self.model_conf_var.get()
        self.settings['model_path'] = self.model_path_var.get()
        self.settings['selected_metin_confidence'] = self.selected_conf_var.get()
        self.settings['false_metin_confidence'] = self.false_conf_var.get()
        self.settings['target_confidence'] = self.target_conf_var.get()
        self.settings['fullhp_metin_confidence'] = self.fullhp_conf_var.get()
        self.settings['auto_loot'] = self.auto_loot_var.get()
        self.settings['save_screenshots'] = self.save_screenshots_var.get()
        self.settings['use_color_filter'] = self.use_color_filter_var.get()
        self.settings['color_filter_threshold'] = self.color_filter_threshold_var.get()
        self.settings['min_metin_width'] = self.min_width_var.get()
        self.settings['min_metin_height'] = self.min_height_var.get()
        self.settings['min_metin_area'] = self.min_width_var.get() * self.min_height_var.get()
        self.settings['max_kill_duration'] = self.max_kill_duration_var.get()
        self.settings['click_cooldown'] = self.click_cooldown_var.get()
        self.settings['post_kill_delay'] = self.post_kill_delay_var.get()
        self.settings['mouse_movement_delay'] = self.mouse_movement_delay_var.get()
        self.settings['click_verification_delay'] = self.click_verification_delay_var.get()
        self.settings['rotation_key_duration'] = self.rotation_key_duration_var.get()
        self.settings['rotation_interval'] = self.rotation_interval_var.get()
        self.settings['search_rotation_interval'] = self.search_rotation_interval_var.get()
        self.settings['main_loop_delay'] = self.main_loop_delay_var.get()
        self.settings['key_press_duration'] = self.key_press_duration_var.get()
        self.settings['key_press_post_delay'] = self.key_press_post_delay_var.get()

        # Create workers only for selected windows
        for window_id in windows_to_run:
            worker = BotWorker(
                window_id,
                self.window_handles[window_id],
                self.log_queue,
                self.counter_queue,
                self.frame_queues[window_id],
                self.settings
            )
            worker.auto_loot_enabled = self.settings['auto_loot']

            self.workers[window_id] = worker

            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            self.worker_threads[window_id] = thread

        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.window1_dropdown.configure(state="disabled")
        self.window2_dropdown.configure(state="disabled")

        # Update log message based on number of windows
        if len(windows_to_run) == 1:
            self.log_message("Bot started for Window 1 only")
        else:
            self.log_message("Bot started for both windows")

    def stop_bot(self):
        """Stop the bot"""
        for worker in self.workers.values():
            worker.stop()

        for thread in self.worker_threads.values():
            thread.join(timeout=2)

        self.workers.clear()
        self.worker_threads.clear()

        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.window1_dropdown.configure(state="normal")
        self.window2_dropdown.configure(state="normal")

        self.log_message("Bot stopped")

    def log_message(self, message):
        """Add message to log"""
        self.log_queue.put(message)

    def update_logs(self):
        """Update log display"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.insert("end", message + "\n")
                self.log_text.see("end")
        except Empty:
            pass

        self.after(100, self.update_logs)

    def update_counters(self):
        """Update metin counters"""
        try:
            while True:
                window_id, count = self.counter_queue.get_nowait()
                self.counters[window_id] = count

                if window_id == 1:
                    self.counter1_label.configure(text=str(count))
                else:
                    self.counter2_label.configure(text=str(count))
        except Empty:
            pass

        self.after(100, self.update_counters)

    def update_previews(self):
        """Update preview displays"""
        try:
            if not self.frame_queues[1].empty():
                window_id, frame = self.frame_queues[1].get_nowait()

                h, w = frame.shape[:2]
                scale = min(150 / w, 100 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h))

                img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)

                # Use CTkImage for proper HiDPI scaling
                ctk_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))

                self.preview1_label.configure(image=ctk_image, text="")
                self.preview1_label.image = ctk_image  # Keep reference

            if not self.frame_queues[2].empty():
                window_id, frame = self.frame_queues[2].get_nowait()

                h, w = frame.shape[:2]
                scale = min(150 / w, 100 / h)
                new_w, new_h = int(w * scale), int(h * scale)
                resized = cv2.resize(frame, (new_w, new_h))

                img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)

                # Use CTkImage for proper HiDPI scaling
                ctk_image = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(new_w, new_h))

                self.preview2_label.configure(image=ctk_image, text="")
                self.preview2_label.image = ctk_image  # Keep reference
        except Empty:
            pass

        self.after(16, self.update_previews)

    def clear_logs(self):
        """Clear log text"""
        self.log_text.delete("1.0", "end")

    def save_settings_gui(self):
        """Save settings from GUI"""
        self.settings['model_confidence'] = self.model_conf_var.get()
        self.settings['model_path'] = self.model_path_var.get()
        self.settings['selected_metin_confidence'] = self.selected_conf_var.get()
        self.settings['false_metin_confidence'] = self.false_conf_var.get()
        self.settings['target_confidence'] = self.target_conf_var.get()
        self.settings['fullhp_metin_confidence'] = self.fullhp_conf_var.get()
        self.settings['auto_loot'] = self.auto_loot_var.get()
        self.settings['save_screenshots'] = self.save_screenshots_var.get()
        self.settings['use_color_filter'] = self.use_color_filter_var.get()
        self.settings['color_filter_threshold'] = self.color_filter_threshold_var.get()
        self.settings['enable_hotkey'] = self.enable_hotkey_var.get()
        self.settings['hotkey'] = self.hotkey_var.get().lower()
        self.settings['min_metin_width'] = self.min_width_var.get()
        self.settings['min_metin_height'] = self.min_height_var.get()
        self.settings['min_metin_area'] = self.min_width_var.get() * self.min_height_var.get()
        self.settings['max_kill_duration'] = self.max_kill_duration_var.get()
        self.settings['click_cooldown'] = self.click_cooldown_var.get()
        self.settings['post_kill_delay'] = self.post_kill_delay_var.get()
        self.settings['mouse_movement_delay'] = self.mouse_movement_delay_var.get()
        self.settings['click_verification_delay'] = self.click_verification_delay_var.get()
        self.settings['rotation_key_duration'] = self.rotation_key_duration_var.get()
        self.settings['rotation_interval'] = self.rotation_interval_var.get()
        self.settings['search_rotation_interval'] = self.search_rotation_interval_var.get()
        self.settings['main_loop_delay'] = self.main_loop_delay_var.get()
        self.settings['key_press_duration'] = self.key_press_duration_var.get()
        self.settings['key_press_post_delay'] = self.key_press_post_delay_var.get()

        self.save_settings()
        self.log_message("Settings saved successfully")

    def destroy(self):
        """Clean up when closing the application"""
        # Stop hotkey listener
        if self.hotkey_listener is not None:
            self.stop_hotkey_listener()

        # Stop bot if running
        if len(self.workers) > 0:
            self.stop_bot()

        # Call parent destroy
        super().destroy()

if __name__ == "__main__":
    app = Metin2FarmBot()
    app.mainloop()
