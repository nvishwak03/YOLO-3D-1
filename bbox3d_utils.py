import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter
from collections import defaultdict
import math

# Default camera intrinsic matrix (can be overridden)
DEFAULT_K = np.array([
    [718.856, 0.0, 607.1928],
    [0.0, 718.856, 185.2157],
    [0.0, 0.0, 1.0]
])

# Default camera projection matrix (can be overridden)
DEFAULT_P = np.array([
    [718.856, 0.0, 607.1928, 45.38225],
    [0.0, 718.856, 185.2157, -0.1130887],
    [0.0, 0.0, 1.0, 0.003779761]
])

# Average dimensions for common objects (height, width, length) in meters
DEFAULT_DIMS = {
    'car': np.array([1.52, 1.64, 3.85]),
    'truck': np.array([3.07, 2.63, 11.17]),
    'bus': np.array([3.07, 2.63, 11.17]),
    'motorcycle': np.array([1.50, 0.90, 2.20]),
    'bicycle': np.array([1.40, 0.70, 1.80]),
    'person': np.array([1.75, 0.60, 0.60]),  # Adjusted width/length for person
    'dog': np.array([0.80, 0.50, 1.10]),
    'cat': np.array([0.40, 0.30, 0.70]),
    # Add indoor objects
    'potted plant': np.array([0.80, 0.40, 0.40]),  # Reduced size for indoor plants
    'plant': np.array([0.80, 0.40, 0.40]),  # Alias for potted plant
    'chair': np.array([0.80, 0.60, 0.60]),
    'sofa': np.array([0.80, 0.85, 2.00]),
    'table': np.array([0.75, 1.20, 1.20]),
    'bed': np.array([0.60, 1.50, 2.00]),
    'tv': np.array([0.80, 0.15, 1.20]),
    'laptop': np.array([0.02, 0.25, 0.35]),
    'keyboard': np.array([0.03, 0.15, 0.45]),
    'mouse': np.array([0.03, 0.06, 0.10]),
    'book': np.array([0.03, 0.20, 0.15]),
    'bottle': np.array([0.25, 0.10, 0.10]),
    'cup': np.array([0.10, 0.08, 0.08]),
    'vase': np.array([0.30, 0.15, 0.15])
}

class BBox3DEstimator:
    """
    3D bounding box estimation from 2D detections and depth
    """
    def __init__(self, camera_matrix=None, projection_matrix=None, class_dims=None):
        """
        Initialize the 3D bounding box estimator
        
        Args:
            camera_matrix (numpy.ndarray): Camera intrinsic matrix (3x3)
            projection_matrix (numpy.ndarray): Camera projection matrix (3x4)
            class_dims (dict): Dictionary mapping class names to dimensions (height, width, length)
        """
        self.K = camera_matrix if camera_matrix is not None else DEFAULT_K
        self.P = projection_matrix if projection_matrix is not None else DEFAULT_P
        self.dims = class_dims if class_dims is not None else DEFAULT_DIMS
        
        # Initialize Kalman filters for tracking 3D boxes
        self.kf_trackers = {}
        
        # Store history of 3D boxes for filtering
        self.box_history = defaultdict(list)
        self.max_history = 5
    
    def estimate_3d_box(self, bbox_2d, depth_value, class_name, object_id=None):
        """
        Estimate 3D bounding box from 2D bounding box and depth
        
        Args:
            bbox_2d (list): 2D bounding box [x1, y1, x2, y2]
            depth_value (float): Depth value at the center of the bounding box
            class_name (str): Class name of the object
            object_id (int): Object ID for tracking (None for no tracking)
            
        Returns:
            dict: 3D bounding box parameters
        """
        # Get 2D box center and dimensions
        x1, y1, x2, y2 = bbox_2d
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width_2d = x2 - x1
        height_2d = y2 - y1
        
        # Get dimensions for the class
        if class_name.lower() in self.dims:
            dimensions = self.dims[class_name.lower()].copy()  # Make a copy to avoid modifying the original
        else:
            # Use default car dimensions if class not found
            dimensions = self.dims['car'].copy()
        
        # Adjust dimensions based on 2D box aspect ratio and size
        aspect_ratio_2d = width_2d / height_2d if height_2d > 0 else 1.0
        
        # For plants, adjust dimensions based on 2D box
        if 'plant' in class_name.lower() or 'potted plant' in class_name.lower():
            # Scale height based on 2D box height
            dimensions[0] = height_2d / 120  # Convert pixels to meters with a scaling factor
            # Make width and length proportional to height
            dimensions[1] = dimensions[0] * 0.6  # width
            dimensions[2] = dimensions[0] * 0.6  # length
        
        # For people, adjust dimensions based on 2D box
        elif 'person' in class_name.lower():
            # Scale height based on 2D box height
            dimensions[0] = height_2d / 100  # Convert pixels to meters with a scaling factor
            # Make width and length proportional to height
            dimensions[1] = dimensions[0] * 0.3  # width
            dimensions[2] = dimensions[0] * 0.3  # length
        
        # Convert depth to distance - use a larger range for better visualization
        # Map depth_value (0-1) to a range of 1-10 meters
        distance = 1.0 + depth_value * 9.0  # Increased from 4.0 to 9.0 for a larger range
        
        # Calculate 3D location
        location = self._backproject_point(center_x, center_y, distance)
        
        # For plants, adjust y-coordinate to place them on a surface
        if 'plant' in class_name.lower() or 'potted plant' in class_name.lower():
            # Assume plants are on a surface (e.g., table, floor)
            # Adjust y-coordinate based on the bottom of the 2D bounding box
            bottom_y = y2  # Bottom of the 2D box
            location[1] = self._backproject_point(center_x, bottom_y, distance)[1]
        
        # Estimate orientation
        orientation = self._estimate_orientation(bbox_2d, location, class_name)
        
        # Create 3D box
        box_3d = {
            'dimensions': dimensions,
            'location': location,
            'orientation': orientation,
            'bbox_2d': bbox_2d,
            'object_id': object_id,
            'class_name': class_name
        }
        
        # Apply Kalman filtering if tracking is enabled
        if object_id is not None:
            box_3d = self._apply_kalman_filter(box_3d, object_id)
            
            # Add to history for temporal filtering
            self.box_history[object_id].append(box_3d)
            if len(self.box_history[object_id]) > self.max_history:
                self.box_history[object_id].pop(0)
            
            # Apply temporal filtering
            box_3d = self._apply_temporal_filter(object_id)
        
        return box_3d
    
    def _backproject_point(self, x, y, depth):
        """
        Backproject a 2D point to 3D space
        
        Args:
            x (float): X coordinate in image space
            y (float): Y coordinate in image space
            depth (float): Depth value
            
        Returns:
            numpy.ndarray: 3D point (x, y, z) in camera coordinates
        """
        # Create homogeneous coordinates
        point_2d = np.array([x, y, 1.0])
        
        # Backproject to 3D
        # The z-coordinate is the depth
        # The x and y coordinates are calculated using the inverse of the camera matrix
        point_3d = np.linalg.inv(self.K) @ point_2d * depth
        
        # For indoor scenes, adjust the y-coordinate to be more realistic
        # In camera coordinates, y is typically pointing down
        # Adjust y to place objects at a reasonable height
        # This is a simplification - in a real system, this would be more sophisticated
        point_3d[1] = point_3d[1] * 0.5  # Scale down y-coordinate
        
        return point_3d
    
    def _estimate_orientation(self, bbox_2d, location, class_name):
        """
        Estimate orientation of the object
        
        Args:
            bbox_2d (list): 2D bounding box [x1, y1, x2, y2]
            location (numpy.ndarray): 3D location of the object
            class_name (str): Class name of the object
            
        Returns:
            float: Orientation angle in radians
        """
        # Calculate ray from camera to object center
        theta_ray = np.arctan2(location[0], location[2])
        
        # For plants and stationary objects, orientation doesn't matter much
        # Just use a fixed orientation aligned with the camera view
        if 'plant' in class_name.lower() or 'potted plant' in class_name.lower():
            # Plants typically don't have a specific orientation
            # Just use the ray angle
            return theta_ray
        
        # For people, they might be facing the camera
        if 'person' in class_name.lower():
            # Assume person is facing the camera
            alpha = 0.0
        else:
            # For other objects, use the 2D box aspect ratio to estimate orientation
            x1, y1, x2, y2 = bbox_2d
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 1.0
            
            # If the object is wide, it might be facing sideways
            if aspect_ratio > 1.5:
                # Object is wide, might be facing sideways
                # Use the position relative to the image center to guess orientation
                image_center_x = self.K[0, 2]  # Principal point x
                if (x1 + x2) / 2 < image_center_x:
                    # Object is on the left side of the image
                    alpha = np.pi / 2  # Facing right
                else:
                    # Object is on the right side of the image
                    alpha = -np.pi / 2  # Facing left
            else:
                # Object has normal proportions, assume it's facing the camera
                alpha = 0.0
        
        # Global orientation
        rot_y = alpha + theta_ray
        
        return rot_y
    
    def _init_kalman_filter(self, box_3d):
        """
        Initialize a Kalman filter for a new object
        
        Args:
            box_3d (dict): 3D bounding box parameters
            
        Returns:
            filterpy.kalman.KalmanFilter: Initialized Kalman filter
        """
        # State: [x, y, z, width, height, length, yaw, vx, vy, vz, vyaw]
        kf = KalmanFilter(dim_x=11, dim_z=7)
        
        # Initial state
        kf.x = np.array([
            box_3d['location'][0],
            box_3d['location'][1],
            box_3d['location'][2],
            box_3d['dimensions'][1],  # width
            box_3d['dimensions'][0],  # height
            box_3d['dimensions'][2],  # length
            box_3d['orientation'],
            0, 0, 0, 0  # Initial velocities
        ])
        
        # State transition matrix (motion model)
        dt = 1.0  # Time step
        kf.F = np.eye(11)
        kf.F[0, 7] = dt  # x += vx * dt
        kf.F[1, 8] = dt  # y += vy * dt
        kf.F[2, 9] = dt  # z += vz * dt
        kf.F[6, 10] = dt  # yaw += vyaw * dt
        
        # Measurement function
        kf.H = np.zeros((7, 11))
        kf.H[0, 0] = 1  # x
        kf.H[1, 1] = 1  # y
        kf.H[2, 2] = 1  # z
        kf.H[3, 3] = 1  # width
        kf.H[4, 4] = 1  # height
        kf.H[5, 5] = 1  # length
        kf.H[6, 6] = 1  # yaw
        
        # Measurement uncertainty
        kf.R = np.eye(7) * 0.1
        kf.R[0:3, 0:3] *= 1.0  # Location uncertainty
        kf.R[3:6, 3:6] *= 0.1  # Dimension uncertainty
        kf.R[6, 6] = 0.3  # Orientation uncertainty
        
        # Process uncertainty
        kf.Q = np.eye(11) * 0.1
        kf.Q[7:11, 7:11] *= 0.5  # Velocity uncertainty
        
        # Initial state uncertainty
        kf.P = np.eye(11) * 1.0
        kf.P[7:11, 7:11] *= 10.0  # Velocity uncertainty
        
        return kf
    
    def _apply_kalman_filter(self, box_3d, object_id):
        """
        Apply Kalman filtering to smooth 3D box parameters
        
        Args:
            box_3d (dict): 3D bounding box parameters
            object_id (int): Object ID for tracking
            
        Returns:
            dict: Filtered 3D bounding box parameters
        """
        # Initialize Kalman filter if this is a new object
        if object_id not in self.kf_trackers:
            self.kf_trackers[object_id] = self._init_kalman_filter(box_3d)
        
        # Get the Kalman filter for this object
        kf = self.kf_trackers[object_id]
        
        # Predict
        kf.predict()
        
        # Update with measurement
        measurement = np.array([
            box_3d['location'][0],
            box_3d['location'][1],
            box_3d['location'][2],
            box_3d['dimensions'][1],  # width
            box_3d['dimensions'][0],  # height
            box_3d['dimensions'][2],  # length
            box_3d['orientation']
        ])
        
        kf.update(measurement)
        
        # Update box_3d with filtered values
        filtered_box = box_3d.copy()
        filtered_box['location'] = np.array([kf.x[0], kf.x[1], kf.x[2]])
        filtered_box['dimensions'] = np.array([kf.x[4], kf.x[3], kf.x[5]])  # height, width, length
        filtered_box['orientation'] = kf.x[6]
        
        return filtered_box
    
    def _apply_temporal_filter(self, object_id):
        """
        Apply temporal filtering to smooth 3D box parameters over time
        
        Args:
            object_id (int): Object ID for tracking
            
        Returns:
            dict: Temporally filtered 3D bounding box parameters
        """
        history = self.box_history[object_id]
        
        if len(history) < 2:
            return history[-1]
        
        # Get the most recent box
        current_box = history[-1]
        
        # Apply exponential moving average to location and orientation
        alpha = 0.7  # Weight for current measurement (higher = less smoothing)
        
        # Initialize with current values
        filtered_box = current_box.copy()
        
        # Apply EMA to location and orientation
        for i in range(len(history) - 2, -1, -1):
            weight = alpha * (1 - alpha) ** (len(history) - i - 2)
            filtered_box['location'] = filtered_box['location'] * (1 - weight) + history[i]['location'] * weight
            
            # Handle orientation wrapping
            angle_diff = history[i]['orientation'] - filtered_box['orientation']
            if angle_diff > np.pi:
                angle_diff -= 2 * np.pi
            elif angle_diff < -np.pi:
                angle_diff += 2 * np.pi
            
            filtered_box['orientation'] += angle_diff * weight
        
        return filtered_box
    
    def project_box_3d_to_2d(self, box_3d):
        """
        Project 3D bounding box corners to 2D image space
        
        Args:
            box_3d (dict): 3D bounding box parameters
            
        Returns:
            numpy.ndarray: 2D points of the 3D box corners (8x2)
        """
        # Extract parameters
        h, w, l = box_3d['dimensions']
        x, y, z = box_3d['location']
        rot_y = box_3d['orientation']
        class_name = box_3d['class_name'].lower()
        
        # Get 2D box for reference
        x1, y1, x2, y2 = box_3d['bbox_2d']
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width_2d = x2 - x1
        height_2d = y2 - y1
        
        # Create rotation matrix
        R_mat = np.array([
            [np.cos(rot_y), 0, np.sin(rot_y)],
            [0, 1, 0],
            [-np.sin(rot_y), 0, np.cos(rot_y)]
        ])
        
        # 3D bounding box corners
        # For plants and stationary objects, make the box more centered
        if 'plant' in class_name or 'potted plant' in class_name:
            # For plants, center the box on the plant
            x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
            y_corners = np.array([h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2])  # Center vertically
            z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
        else:
            # For other objects, use standard box configuration
            x_corners = np.array([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2])
            y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h])  # Bottom at y=0
            z_corners = np.array([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2])
        
        # Rotate and translate corners
        corners_3d = np.vstack([x_corners, y_corners, z_corners])
        corners_3d = R_mat @ corners_3d
        corners_3d[0, :] += x
        corners_3d[1, :] += y
        corners_3d[2, :] += z
        
        # Project to 2D
        corners_3d_homo = np.vstack([corners_3d, np.ones((1, 8))])
        corners_2d_homo = self.P @ corners_3d_homo
        corners_2d = corners_2d_homo[:2, :] / corners_2d_homo[2, :]
        
        # Constrain the 3D box to be within a reasonable distance of the 2D box
        # This helps prevent wildly incorrect projections
        mean_x = np.mean(corners_2d[0, :])
        mean_y = np.mean(corners_2d[1, :])
        
        # If the projected box is too far from the 2D box center, adjust it
        if abs(mean_x - center_x) > width_2d or abs(mean_y - center_y) > height_2d:
            # Shift the projected points to center on the 2D box
            shift_x = center_x - mean_x
            shift_y = center_y - mean_y
            corners_2d[0, :] += shift_x
            corners_2d[1, :] += shift_y
        
        return corners_2d.T
    
    def draw_box_3d(self, image, box_3d, color=(0, 255, 0), thickness=2):
        """
        Draw simplified 3D bounding box on image (only class name).
        """
        x1, y1, x2, y2 = [int(coord) for coord in box_3d['bbox_2d']]
        
        # Get depth value for scaling
        depth_value = box_3d.get('depth_value', 0.5)
        
        # Calculate box dimensions
        width = x2 - x1
        height = y2 - y1
        
        # Calculate the offset for the 3D effect (deeper objects have smaller offset)
        # Inverse relationship with depth - closer objects have larger offset
        offset_factor = 1.0 - depth_value
        offset_x = int(width * 0.3 * offset_factor)
        offset_y = int(height * 0.3 * offset_factor)
        
        # Ensure minimum offset for visibility
        offset_x = max(15, min(offset_x, 50))
        offset_y = max(15, min(offset_y, 50))
        
        # Create points for the 3D box
        # Front face (the 2D bounding box)
        front_tl = (x1, y1)
        front_tr = (x2, y1)
        front_br = (x2, y2)
        front_bl = (x1, y2)
        
        # Back face (offset by depth)
        back_tl = (x1 + offset_x, y1 - offset_y)
        back_tr = (x2 + offset_x, y1 - offset_y)
        back_br = (x2 + offset_x, y2 - offset_y)
        back_bl = (x1 + offset_x, y2 - offset_y)
        
        # Create a slightly transparent copy of the image for the 3D effect
        overlay = image.copy()
        
        # Draw the front face (2D bounding box)
        cv2.rectangle(image, front_tl, front_br, color, thickness)
        
        # Draw the connecting lines between front and back faces
        cv2.line(image, front_tl, back_tl, color, thickness)
        cv2.line(image, front_tr, back_tr, color, thickness)
        cv2.line(image, front_br, back_br, color, thickness)
        cv2.line(image, front_bl, back_bl, color, thickness)
        
        # Draw the back face
        cv2.line(image, back_tl, back_tr, color, thickness)
        cv2.line(image, back_tr, back_br, color, thickness)
        cv2.line(image, back_br, back_bl, color, thickness)
        cv2.line(image, back_bl, back_tl, color, thickness)
        
        # Fill the top face with a semi-transparent color to enhance 3D effect
        pts_top = np.array([front_tl, front_tr, back_tr, back_tl], np.int32)
        pts_top = pts_top.reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_top], color)
        
        # Fill the right face with a semi-transparent color
        pts_right = np.array([front_tr, front_br, back_br, back_tr], np.int32)
        pts_right = pts_right.reshape((-1, 1, 2))
        # Darken the right face color for better 3D effect
        right_color = (int(color[0] * 0.7), int(color[1] * 0.7), int(color[2] * 0.7))
        cv2.fillPoly(overlay, [pts_right], right_color)
        
        # Apply the overlay with transparency
        alpha = 0.3  # Transparency factor
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        
        # Get class name and object ID
        class_name = box_3d['class_name']
        label_y = y1 - 10
        cv2.putText(image, class_name, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return image
    def cleanup_trackers(self, active_ids):
        """
        Clean up Kalman filters and history for objects that are no longer tracked
        
        Args:
            active_ids (list): List of active object IDs
        """
        # Convert to set for faster lookup
        active_ids_set = set(active_ids)
        
        # Clean up Kalman filters
        for obj_id in list(self.kf_trackers.keys()):
            if obj_id not in active_ids_set:
                del self.kf_trackers[obj_id]
        
        # Clean up box history
        for obj_id in list(self.box_history.keys()):
            if obj_id not in active_ids_set:
                del self.box_history[obj_id]
class BirdEyeView:
    def __init__(self, size=(600, 800), scale=50):
        self.width, self.height = size
        self.scale = scale

        self.bev_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.origin_x = self.width // 2
        self.origin_y = self.height - 30

        # Load icons (forward and reverse)
        self.car_icon = cv2.imread('car_icon.png', cv2.IMREAD_UNCHANGED)
        self.car_icon_rev = cv2.imread('car_icon_reverse.png', cv2.IMREAD_UNCHANGED)

        self.truck_icon = cv2.imread('truck_icon.png', cv2.IMREAD_UNCHANGED)
        self.truck_icon_rev = cv2.imread('truck_icon_reverse.png', cv2.IMREAD_UNCHANGED)

        self.bus_icon = cv2.imread('bus_icon.png', cv2.IMREAD_UNCHANGED)
        self.bus_icon_rev = cv2.imread('bus_icon_reverse.png', cv2.IMREAD_UNCHANGED)

        print("Icons loaded.")

    def reset(self):
        self.bev_image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.bev_image[:, :] = (20, 20, 20)

    def overlay_icon(self, icon, position, scale=0.2):
        ih, iw = int(icon.shape[0] * scale), int(icon.shape[1] * scale)
        icon_resized = cv2.resize(icon, (iw, ih))

        x, y = position
        x1, y1 = x - iw // 2, y - ih // 2

        if x1 < 0 or y1 < 0 or x1 + iw > self.width or y1 + ih > self.height:
            return

        roi = self.bev_image[y1:y1+ih, x1:x1+iw]

        if icon_resized.shape[2] == 4:
            alpha_s = icon_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                roi[:, :, c] = (alpha_s * icon_resized[:, :, c] + alpha_l * roi[:, :, c])
        else:
            roi[:] = icon_resized

    def draw_box(self, box_3d):
        try:
            class_name = box_3d['class_name'].lower()
            depth_value = box_3d.get('depth_value', 0.5)
            depth = 1 + depth_value * 9

            x1, y1, x2, y2 = box_3d['bbox_2d']
            center_x_2d = (x1 + x2) / 2
            lane_offset = int((center_x_2d / 1280 - 0.5) * self.scale * 4)

            bev_x = self.origin_x + lane_offset
            bev_y = self.origin_y - int(depth * self.scale)

            # --- Determine Forward or Reverse ---
            orientation_rad = box_3d.get('orientation', 0.0)
            angle_deg = np.degrees(orientation_rad) % 360

            is_reverse = angle_deg > 90 and angle_deg < 270  # Between 90° and 270°

            # --- Select Icon Based on Class and Direction ---
            if 'bus' in class_name:
                icon = self.bus_icon_rev if is_reverse else self.bus_icon
                self.overlay_icon(icon, (bev_x, bev_y), scale=0.15)

            elif 'truck' in class_name:
                icon = self.truck_icon_rev if is_reverse else self.truck_icon
                self.overlay_icon(icon, (bev_x, bev_y), scale=0.20)

            elif 'car' in class_name:
                icon = self.car_icon_rev if is_reverse else self.car_icon
                self.overlay_icon(icon, (bev_x, bev_y), scale=0.18)

            elif 'person' in class_name:
                cv2.circle(self.bev_image, (bev_x, bev_y), 6, (0, 255, 0), -1)
            else:
                cv2.circle(self.bev_image, (bev_x, bev_y), 8, (200, 200, 200), -1)

        except Exception as e:
            print(f"BEV Draw Error: {e}")

    def get_image(self):
        return self.bev_image