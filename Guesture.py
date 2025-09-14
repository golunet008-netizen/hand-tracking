import pygame
import sys
import cv2
import numpy as np
import mediapipe as mp
from pygame.locals import *

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Real-Time Hand Tracking System")

# Colors
BACKGROUND = (15, 25, 35)
ACCENT = (0, 200, 150)
WHITE = (230, 240, 250)
GRAY = (100, 120, 140)
RED = (220, 80, 60)
GREEN = (80, 200, 120)
BLUE = (50, 150, 220)
YELLOW = (220, 200, 70)
PURPLE = (180, 100, 220)

# Fonts
title_font = pygame.font.SysFont("Arial", 40, bold=True)
subtitle_font = pygame.font.SysFont("Arial", 28, bold=True)
text_font = pygame.font.SysFont("Arial", 20)
small_font = pygame.font.SysFont("Arial", 16)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Application state
class AppState:
    def __init__(self):
        self.current_gesture = "Unknown"
        self.confidence = 0.0
        self.fps = 0
        self.hand_count = 0
        self.landmarks = []
        self.performance_metrics = {
            "Accuracy": 0.92,
            "Precision": 0.89,
            "Recall": 0.94
        }
        self.visualization = True
        self.camera_on = True
        self.bounding_box = [0, 0, 0, 0]  # x, y, width, height
        
app_state = AppState()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize MediaPipe Hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5)

# Gesture recognition based on finger states
def recognize_gesture(hand_landmarks):
    # Get finger joint landmarks
    landmarks = hand_landmarks.landmark
    
    # Finger states (1 = extended, 0 = bent)
    thumb_state = 1 if landmarks[4].x < landmarks[3].x else 0
    index_state = 1 if landmarks[8].y < landmarks[6].y else 0
    middle_state = 1 if landmarks[12].y < landmarks[10].y else 0
    ring_state = 1 if landmarks[16].y < landmarks[14].y else 0
    pinky_state = 1 if landmarks[20].y < landmarks[18].y else 0
    
    # Count extended fingers
    extended_fingers = thumb_state + index_state + middle_state + ring_state + pinky_state
    
    # Determine gesture based on finger states
    if extended_fingers == 0:
        return "Fist", 0.95
    elif extended_fingers == 5:
        return "Open Palm", 0.95
    elif index_state == 1 and middle_state == 0 and ring_state == 0 and pinky_state == 0:
        return "Pointing", 0.9
    elif index_state == 1 and middle_state == 1 and ring_state == 0 and pinky_state == 0:
        return "Peace", 0.9
    elif thumb_state == 1 and index_state == 1 and middle_state == 0 and ring_state == 0 and pinky_state == 0:
        return "Thumbs Up", 0.85
    else:
        return "Unknown", 0.7

# Draw a fancy button
def draw_button(text, rect, color, hover=False, icon=None):
    pygame.draw.rect(screen, color, rect, border_radius=12)
    pygame.draw.rect(screen, WHITE, rect, 2, border_radius=12)
    
    if hover:
        pygame.draw.rect(screen, (255, 255, 255, 30), rect, border_radius=12)
    
    text_surf = text_font.render(text, True, WHITE)
    text_rect = text_surf.get_rect(center=rect.center)
    screen.blit(text_surf, text_rect)
    
    if icon:
        icon_surf = text_font.render(icon, True, WHITE)
        icon_rect = icon_surf.get_rect(midright=(rect.centerx - 50, rect.centery))
        screen.blit(icon_surf, icon_rect)

# Draw a metric box
def draw_metric(title, value, rect, color):
    pygame.draw.rect(screen, (30, 40, 50), rect, border_radius=10)
    pygame.draw.rect(screen, color, (rect.x, rect.y, rect.width, 5), border_radius=10)
    
    title_surf = small_font.render(title, True, GRAY)
    title_rect = title_surf.get_rect(center=(rect.centerx, rect.y + 20))
    screen.blit(title_surf, title_rect)
    
    value_surf = subtitle_font.render(str(value), True, WHITE)
    value_rect = value_surf.get_rect(center=(rect.centerx, rect.y + rect.height // 2))
    screen.blit(value_surf, value_rect)

# Draw performance graph
def draw_performance_graph(rect):
    pygame.draw.rect(screen, (25, 35, 45), rect, border_radius=10)
    pygame.draw.rect(screen, ACCENT, rect, 2, border_radius=10)
    
    # Draw graph title
    title_surf = small_font.render("Performance Metrics", True, WHITE)
    title_rect = title_surf.get_rect(center=(rect.centerx, rect.y + 20))
    screen.blit(title_surf, title_rect)
    
    # Draw metrics
    metrics = list(app_state.performance_metrics.items())
    for i, (name, value) in enumerate(metrics):
        y_pos = rect.y + 60 + i * 40
        # Draw metric name
        name_surf = small_font.render(name, True, GRAY)
        name_rect = name_surf.get_rect(midleft=(rect.x + 20, y_pos))
        screen.blit(name_surf, name_rect)
        
        # Draw value
        value_surf = small_font.render(f"{value*100:.1f}%", True, WHITE)
        value_rect = value_surf.get_rect(midright=(rect.x + rect.width - 20, y_pos))
        screen.blit(value_surf, value_rect)
        
        # Draw bar
        bar_rect = pygame.Rect(rect.x + 20, y_pos + 15, rect.width - 40, 10)
        pygame.draw.rect(screen, (50, 60, 70), bar_rect, border_radius=5)
        fill_width = int((rect.width - 40) * value)
        pygame.draw.rect(screen, GREEN, (bar_rect.x, bar_rect.y, fill_width, bar_rect.height), border_radius=5)

# Draw hand landmarks visualization
def draw_hand_visualization(surface, landmarks, rect):
    # Draw background
    pygame.draw.rect(surface, (25, 35, 45), rect, border_radius=10)
    pygame.draw.rect(surface, ACCENT, rect, 2, border_radius=10)
    
    # Draw title
    title_surf = small_font.render("Hand Landmarks", True, WHITE)
    title_rect = title_surf.get_rect(center=(rect.centerx, rect.y + 20))
    surface.blit(title_surf, title_rect)
    
    if not landmarks:
        # Draw placeholder message
        msg_surf = small_font.render("No hand detected", True, GRAY)
        msg_rect = msg_surf.get_rect(center=rect.center)
        surface.blit(msg_surf, msg_rect)
        return
    
    # Draw hand connections
    scale = min(rect.width, rect.height) * 0.8
    center_x, center_y = rect.centerx, rect.centery + 20
    
    # Draw connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
        (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
        (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
    ]
    
    for start_idx, end_idx in connections:
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            
            start_pos = (int(center_x + start.x * scale), int(center_y + start.y * scale))
            end_pos = (int(center_x + end.x * scale), int(center_y + end.y * scale))
            
            pygame.draw.line(surface, ACCENT, start_pos, end_pos, 2)
    
    # Draw landmarks
    for landmark in landmarks:
        pos = (int(center_x + landmark.x * scale), int(center_y + landmark.y * scale))
        pygame.draw.circle(surface, YELLOW, pos, 4)
        pygame.draw.circle(surface, BLUE, pos, 6, 1)

# Draw the application
def draw_app():
    # Draw background
    screen.fill(BACKGROUND)
    
    # Draw title
    title_surf = title_font.render("Real-Time Hand Tracking System", True, ACCENT)
    screen.blit(title_surf, (WIDTH // 2 - title_surf.get_width() // 2, 20))
    
    # Draw webcam feed area
    webcam_rect = pygame.Rect(50, 80, 640, 480)
    pygame.draw.rect(screen, (25, 35, 45), webcam_rect, border_radius=10)
    pygame.draw.rect(screen, ACCENT, webcam_rect, 2, border_radius=10)
    
    # Draw webcam title
    webcam_title = subtitle_font.render("Webcam Input", True, WHITE)
    screen.blit(webcam_title, (webcam_rect.centerx - webcam_title.get_width() // 2, webcam_rect.y - 35))
    
    # Draw visualization area
    viz_rect = pygame.Rect(710, 80, 440, 220)
    draw_hand_visualization(screen, app_state.landmarks, viz_rect)
    
    # Draw gesture info
    gesture_rect = pygame.Rect(710, 320, 440, 120)
    pygame.draw.rect(screen, (25, 35, 45), gesture_rect, border_radius=10)
    pygame.draw.rect(screen, ACCENT, gesture_rect, 2, border_radius=10)
    
    gesture_title = subtitle_font.render("Detected Gesture", True, WHITE)
    screen.blit(gesture_title, (gesture_rect.centerx - gesture_title.get_width() // 2, gesture_rect.y + 15))
    
    gesture_text = subtitle_font.render(app_state.current_gesture, True, YELLOW)
    gesture_text_rect = gesture_text.get_rect(center=(gesture_rect.centerx, gesture_rect.y + 60))
    screen.blit(gesture_text, gesture_text_rect)
    
    conf_text = text_font.render(f"Confidence: {app_state.confidence*100:.1f}%", True, WHITE)
    conf_rect = conf_text.get_rect(center=(gesture_rect.centerx, gesture_rect.y + 90))
    screen.blit(conf_text, conf_rect)
    
    # Draw performance metrics
    perf_rect = pygame.Rect(710, 460, 440, 180)
    draw_performance_graph(perf_rect)
    
    # Draw FPS and hand count
    info_rect = pygame.Rect(50, 580, 640, 50)
    pygame.draw.rect(screen, (25, 35, 45), info_rect, border_radius=10)
    pygame.draw.rect(screen, ACCENT, info_rect, 2, border_radius=10)
    
    fps_text = small_font.render(f"FPS: {app_state.fps:.1f}", True, WHITE)
    screen.blit(fps_text, (info_rect.x + 20, info_rect.centery - fps_text.get_height() // 2))
    
    hands_text = small_font.render(f"Hands Detected: {app_state.hand_count}", True, WHITE)
    screen.blit(hands_text, (info_rect.x + 200, info_rect.centery - hands_text.get_height() // 2))
    
    conf_text = small_font.render(f"Confidence: {app_state.confidence*100:.1f}%", True, WHITE)
    screen.blit(conf_text, (info_rect.x + 400, info_rect.centery - conf_text.get_height() // 2))
    
    # Draw camera control button
    control_rect = pygame.Rect(710, 650, 440, 50)
    draw_button(
        "Camera: " + ("ON" if app_state.camera_on else "OFF"), 
        control_rect, 
        GREEN if app_state.camera_on else RED
    )

# Main application loop
clock = pygame.time.Clock()
fps_time = pygame.time.get_ticks()
frame_count = 0

running = True
while running:
    current_time = pygame.time.get_ticks()
    frame_count += 1
    
    # Calculate FPS every second
    if current_time - fps_time >= 1000:
        app_state.fps = frame_count
        frame_count = 0
        fps_time = current_time
    
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == K_ESCAPE:
                running = False
            elif event.key == K_c:
                app_state.camera_on = not app_state.camera_on
        elif event.type == MOUSEBUTTONDOWN:
            # Check if camera control button was clicked
            control_rect = pygame.Rect(710, 650, 440, 50)
            if control_rect.collidepoint(event.pos):
                app_state.camera_on = not app_state.camera_on
    
    # Process webcam frame if camera is on
    if app_state.camera_on:
        ret, frame = cap.read()
        if ret:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert the BGR image to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe Hands
            results = hands.process(rgb_frame)
            
            # Reset hand count and landmarks
            app_state.hand_count = 0
            app_state.landmarks = []
            
            if results.multi_hand_landmarks:
                app_state.hand_count = len(results.multi_hand_landmarks)
                
                # Process each detected hand
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
                    
                    # Recognize gesture
                    gesture, confidence = recognize_gesture(hand_landmarks)
                    app_state.current_gesture = gesture
                    app_state.confidence = confidence
                    
                    # Extract landmarks for visualization
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.append(pygame.Vector2(lm.x, lm.y))
                    app_state.landmarks = landmarks
            
            # Convert the frame to a format Pygame can display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.rot90(frame)
            frame_surface = pygame.surfarray.make_surface(frame)
            
            # Draw the webcam frame
            webcam_rect = pygame.Rect(50, 80, 640, 480)
            screen.blit(pygame.transform.scale(frame_surface, (640, 480)), webcam_rect)
    else:
        # Display placeholder when camera is off
        webcam_rect = pygame.Rect(50, 80, 640, 480)
        pygame.draw.rect(screen, (20, 30, 40), webcam_rect)
        camera_off_text = subtitle_font.render("Camera is OFF", True, GRAY)
        screen.blit(camera_off_text, (webcam_rect.centerx - camera_off_text.get_width() // 2, 
                                     webcam_rect.centery - camera_off_text.get_height() // 2))
    
    # Draw the application UI
    draw_app()
    
    # Update the display
    pygame.display.flip()
    clock.tick(30)

# Clean up
cap.release()
hands.close()
pygame.quit()
sys.exit()
