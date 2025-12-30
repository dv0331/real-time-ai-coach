"""
Vision Pipeline: Face detection, landmarks, gaze estimation, head pose, presence scoring.
Uses MediaPipe for efficient cross-platform face analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time
import logging
import io

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    logging.warning("OpenCV not installed")

HAS_MEDIAPIPE = False
mp_face_mesh = None
try:
    import mediapipe as mp
    # Try solutions API first (older versions)
    if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
        mp_face_mesh = mp.solutions.face_mesh
        HAS_MEDIAPIPE = True
    # Try newer task-based API
    elif hasattr(mp, 'tasks'):
        logging.info("MediaPipe tasks API detected - using simplified vision")
        HAS_MEDIAPIPE = False  # We'll handle without advanced face mesh
    else:
        logging.warning("MediaPipe installed but no compatible API found")
except ImportError:
    logging.warning("MediaPipe not installed")

from PIL import Image
from collections import deque
from config import config

logger = logging.getLogger(__name__)

@dataclass
class VisionFeatures:
    """Features extracted from video frame."""
    timestamp: float = 0.0
    
    # Face detection
    face_detected: bool = False
    face_bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)  # x, y, w, h normalized
    face_size: float = 0.0            # Face size as fraction of frame
    num_faces: int = 0
    
    # Gaze estimation (normalized, 0.5 = center)
    gaze_x: float = 0.5
    gaze_y: float = 0.5
    looking_at_camera: bool = True
    
    # Head pose (degrees)
    head_yaw: float = 0.0             # Left/right rotation
    head_pitch: float = 0.0           # Up/down tilt
    head_roll: float = 0.0            # Head tilt
    
    # Motion
    head_motion: float = 0.0          # Movement magnitude
    
    # Expression (basic)
    mouth_open_ratio: float = 0.0
    eye_aspect_ratio: float = 0.0     # For blink detection
    
    # Scores (0-1, higher is better)
    eye_contact_score: float = 0.5
    presence_score: float = 0.5
    stability_score: float = 0.5

class VisionPipeline:
    """
    Processes video frames and extracts presentation-relevant features.
    Uses MediaPipe FaceMesh for detailed face analysis.
    """
    
    def __init__(self):
        self.target_fps = config.vision.TARGET_FPS
        self.frame_interval = 1.0 / self.target_fps
        self.last_process_time = 0.0
        
        # Initialize MediaPipe
        self.face_mesh = None
        if HAS_MEDIAPIPE and mp_face_mesh:
            try:
                self.face_mesh = mp_face_mesh.FaceMesh(
                    max_num_faces=1,
                    refine_landmarks=True,  # Include iris landmarks
                    min_detection_confidence=config.vision.MIN_DETECTION_CONFIDENCE,
                    min_tracking_confidence=config.vision.MIN_TRACKING_CONFIDENCE
                )
                logger.info("MediaPipe FaceMesh initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize FaceMesh: {e}")
                self.face_mesh = None
        
        # History for smoothing and motion detection
        self.gaze_history: deque = deque(maxlen=10)
        self.pose_history: deque = deque(maxlen=10)
        self.face_center_history: deque = deque(maxlen=5)
        
        # Landmark indices for key points (MediaPipe FaceMesh)
        # Nose tip
        self.NOSE_TIP = 1
        # Eye landmarks for gaze
        self.LEFT_EYE_CENTER = 468   # Left iris center (with refine_landmarks)
        self.RIGHT_EYE_CENTER = 473  # Right iris center
        # Eye corners for EAR calculation
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        # Mouth landmarks
        self.MOUTH_TOP = 13
        self.MOUTH_BOTTOM = 14
        self.MOUTH_LEFT = 61
        self.MOUTH_RIGHT = 291
        
        # 3D model points for head pose estimation
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # Camera matrix (will be set based on frame size)
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1))
        
        logger.info(f"VisionPipeline initialized: target {self.target_fps} FPS")
    
    def should_process(self) -> bool:
        """Check if enough time has passed to process a new frame."""
        now = time.time()
        if now - self.last_process_time >= self.frame_interval:
            self.last_process_time = now
            return True
        return False
    
    def process_frame(self, image_data: bytes) -> Optional[VisionFeatures]:
        """
        Process a video frame (JPEG bytes).
        Returns features or None if processing should be skipped.
        """
        if not self.should_process():
            return None
        
        features = VisionFeatures(timestamp=time.time())
        
        if not HAS_CV2 or not HAS_MEDIAPIPE:
            logger.warning("OpenCV or MediaPipe not available")
            return features
        
        try:
            # Decode JPEG
            img_array = np.frombuffer(image_data, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.warning("Failed to decode frame")
                return features
            
            h, w = frame.shape[:2]
            
            # Initialize camera matrix if needed
            if self.camera_matrix is None:
                focal_length = w
                center = (w / 2, h / 2)
                self.camera_matrix = np.array([
                    [focal_length, 0, center[0]],
                    [0, focal_length, center[1]],
                    [0, 0, 1]
                ], dtype=np.float64)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with FaceMesh
            results = self.face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                features.face_detected = True
                features.num_faces = len(results.multi_face_landmarks)
                
                landmarks = results.multi_face_landmarks[0]
                
                # Extract features
                self._extract_face_bbox(landmarks, w, h, features)
                self._extract_gaze(landmarks, w, h, features)
                self._extract_head_pose(landmarks, w, h, features)
                self._extract_expression(landmarks, features)
                self._compute_motion(features)
                
            # Compute scores
            self._compute_scores(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return features
    
    def _extract_face_bbox(self, landmarks, w: int, h: int, features: VisionFeatures):
        """Extract face bounding box from landmarks."""
        xs = [lm.x for lm in landmarks.landmark]
        ys = [lm.y for lm in landmarks.landmark]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        features.face_bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
        features.face_size = (x_max - x_min) * (y_max - y_min)
        
        # Track face center for motion detection
        center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        self.face_center_history.append(center)
    
    def _extract_gaze(self, landmarks, w: int, h: int, features: VisionFeatures):
        """Estimate gaze direction using iris landmarks."""
        try:
            # Get iris centers (requires refine_landmarks=True)
            left_iris = landmarks.landmark[self.LEFT_EYE_CENTER]
            right_iris = landmarks.landmark[self.RIGHT_EYE_CENTER]
            
            # Average iris position
            gaze_x = (left_iris.x + right_iris.x) / 2
            gaze_y = (left_iris.y + right_iris.y) / 2
            
            # Get eye corners for relative position
            left_corner = landmarks.landmark[33]
            right_corner = landmarks.landmark[263]
            
            # Normalize gaze relative to eye position
            eye_center_x = (left_corner.x + right_corner.x) / 2
            eye_width = abs(right_corner.x - left_corner.x)
            
            if eye_width > 0:
                # Relative gaze: 0.5 = looking straight
                relative_x = (gaze_x - eye_center_x) / eye_width + 0.5
                features.gaze_x = np.clip(relative_x, 0, 1)
            
            features.gaze_y = gaze_y
            
            # Smooth gaze
            self.gaze_history.append((features.gaze_x, features.gaze_y))
            if len(self.gaze_history) >= 3:
                avg_gaze = np.mean(list(self.gaze_history), axis=0)
                features.gaze_x, features.gaze_y = avg_gaze
            
            # Check if looking at camera
            tolerance = config.vision.GAZE_CENTER_TOLERANCE
            features.looking_at_camera = (
                abs(features.gaze_x - 0.5) < tolerance and
                abs(features.gaze_y - 0.4) < tolerance  # Slightly above center is natural
            )
            
        except Exception as e:
            logger.debug(f"Gaze extraction error: {e}")
    
    def _extract_head_pose(self, landmarks, w: int, h: int, features: VisionFeatures):
        """Estimate head pose using PnP."""
        try:
            # Get 2D image points for pose estimation
            image_points = np.array([
                (landmarks.landmark[1].x * w, landmarks.landmark[1].y * h),    # Nose tip
                (landmarks.landmark[152].x * w, landmarks.landmark[152].y * h),  # Chin
                (landmarks.landmark[33].x * w, landmarks.landmark[33].y * h),   # Left eye corner
                (landmarks.landmark[263].x * w, landmarks.landmark[263].y * h), # Right eye corner
                (landmarks.landmark[61].x * w, landmarks.landmark[61].y * h),   # Left mouth corner
                (landmarks.landmark[291].x * w, landmarks.landmark[291].y * h)  # Right mouth corner
            ], dtype=np.float64)
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                self.model_points,
                image_points,
                self.camera_matrix,
                self.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Convert rotation vector to Euler angles
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                angles = self._rotation_matrix_to_euler(rotation_matrix)
                
                features.head_pitch = float(angles[0])
                features.head_yaw = float(angles[1])
                features.head_roll = float(angles[2])
                
                # Smooth pose
                self.pose_history.append((features.head_yaw, features.head_pitch, features.head_roll))
                if len(self.pose_history) >= 3:
                    avg_pose = np.mean(list(self.pose_history), axis=0)
                    features.head_yaw, features.head_pitch, features.head_roll = avg_pose
                    
        except Exception as e:
            logger.debug(f"Head pose error: {e}")
    
    def _rotation_matrix_to_euler(self, R: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation matrix to Euler angles (degrees)."""
        sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.degrees(x), np.degrees(y), np.degrees(z)
    
    def _extract_expression(self, landmarks, features: VisionFeatures):
        """Extract basic expression features."""
        try:
            # Mouth open ratio
            top = landmarks.landmark[self.MOUTH_TOP]
            bottom = landmarks.landmark[self.MOUTH_BOTTOM]
            left = landmarks.landmark[self.MOUTH_LEFT]
            right = landmarks.landmark[self.MOUTH_RIGHT]
            
            mouth_height = abs(bottom.y - top.y)
            mouth_width = abs(right.x - left.x)
            
            if mouth_width > 0:
                features.mouth_open_ratio = mouth_height / mouth_width
            
            # Eye aspect ratio (for blink detection)
            left_ear = self._eye_aspect_ratio(landmarks, self.LEFT_EYE)
            right_ear = self._eye_aspect_ratio(landmarks, self.RIGHT_EYE)
            features.eye_aspect_ratio = (left_ear + right_ear) / 2
            
        except Exception as e:
            logger.debug(f"Expression extraction error: {e}")
    
    def _eye_aspect_ratio(self, landmarks, eye_indices: List[int]) -> float:
        """Calculate eye aspect ratio for blink detection."""
        try:
            points = [landmarks.landmark[i] for i in eye_indices]
            
            # Vertical distances
            v1 = np.sqrt((points[1].x - points[5].x)**2 + (points[1].y - points[5].y)**2)
            v2 = np.sqrt((points[2].x - points[4].x)**2 + (points[2].y - points[4].y)**2)
            
            # Horizontal distance
            h = np.sqrt((points[0].x - points[3].x)**2 + (points[0].y - points[3].y)**2)
            
            if h > 0:
                return (v1 + v2) / (2.0 * h)
        except:
            pass
        return 0.3  # Default open eye
    
    def _compute_motion(self, features: VisionFeatures):
        """Compute head motion from face center history."""
        if len(self.face_center_history) >= 2:
            centers = list(self.face_center_history)
            motion = 0.0
            for i in range(1, len(centers)):
                dx = centers[i][0] - centers[i-1][0]
                dy = centers[i][1] - centers[i-1][1]
                motion += np.sqrt(dx**2 + dy**2)
            features.head_motion = motion / (len(centers) - 1)
    
    def _compute_scores(self, features: VisionFeatures):
        """Compute normalized scores from features."""
        
        # Eye contact score
        if not features.face_detected:
            features.eye_contact_score = 0.0
        else:
            # Based on gaze proximity to center
            gaze_offset = np.sqrt(
                (features.gaze_x - 0.5)**2 + 
                (features.gaze_y - 0.4)**2
            )
            features.eye_contact_score = max(0, 1.0 - gaze_offset * 3)
            
            # Penalize head yaw (looking away)
            yaw_penalty = abs(features.head_yaw) / config.vision.HEAD_YAW_TOLERANCE
            features.eye_contact_score *= max(0, 1.0 - yaw_penalty * 0.5)
        
        # Presence score (good framing, face size, stable)
        if not features.face_detected:
            features.presence_score = 0.0
        else:
            # Ideal face size
            size_score = 1.0
            if features.face_size < config.vision.FACE_SIZE_MIN:
                size_score = features.face_size / config.vision.FACE_SIZE_MIN
            elif features.face_size > config.vision.FACE_SIZE_MAX:
                size_score = config.vision.FACE_SIZE_MAX / features.face_size
            
            # Face centering
            center_x = features.face_bbox[0] + features.face_bbox[2] / 2
            center_offset = abs(center_x - 0.5)
            center_score = max(0, 1.0 - center_offset * 2)
            
            features.presence_score = (size_score + center_score) / 2
        
        # Stability score (low motion is good)
        features.stability_score = max(0, 1.0 - features.head_motion * 10)
    
    def reset(self):
        """Reset for new session."""
        self.gaze_history.clear()
        self.pose_history.clear()
        self.face_center_history.clear()
        self.last_process_time = 0.0
        logger.info("VisionPipeline reset")
    
    def get_session_stats(self) -> dict:
        """Get session summary statistics."""
        return {
            "frames_processed": len(self.gaze_history),
            "average_gaze_x": float(np.mean([g[0] for g in self.gaze_history])) if self.gaze_history else 0.5,
            "average_gaze_y": float(np.mean([g[1] for g in self.gaze_history])) if self.gaze_history else 0.5,
        }

