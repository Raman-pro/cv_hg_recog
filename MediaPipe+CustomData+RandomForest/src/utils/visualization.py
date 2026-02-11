import cv2
import numpy as np

class Visualization:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height

    def get_rotated_triangle(self, center, size, yaw_deg):
        """Calculates points for an isoceles triangle rotated by yaw."""
        pts = np.array([
            [0, -size],
            [-size//1.5, size],
            [size//1.5, size]
        ])

        angle = np.radians(yaw_deg)
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle),  np.cos(angle)]
        ])
        rotated_pts = pts @ rot_matrix.T
        final_pts = (rotated_pts + center).astype(np.int32)
        return final_pts

    def draw_top_view(self, drone):
        top_view = np.zeros((800, 800, 3), dtype=np.uint8)
        d_center = (400, 400)
        
        # Scale values if needed, but original code just added drone.x/y
        tri_pts = self.get_rotated_triangle((d_center[0] + drone.x, d_center[1] + drone.y), 20, drone.yaw)
        
        cv2.drawContours(top_view, [tri_pts], 0, (0, 255, 255), -1)
        cv2.putText(top_view, f"Top View: X={int(drone.x)}, Y={int(drone.y)}, Yaw={int(drone.yaw)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return top_view

    def draw_side_view(self, drone):
        side_view = np.zeros((400, 200, 3), dtype=np.uint8)
        z_pos = 200 - int(drone.z * 0.5)
        
        # Simple triangle representation
        pts = np.array([[100, z_pos-10], [70, z_pos+10], [130, z_pos+10]])
        cv2.fillPoly(side_view, [pts], (255, 0, 255))
        
        cv2.putText(side_view, f"Side View: Z={int(drone.z)}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return side_view

    def draw_control_zones(self, frame, width, height, right_center, large_radius, small_radius):
        cv2.circle(frame, right_center, large_radius, (100, 100, 100), 2)
        cv2.circle(frame, right_center, small_radius, (100, 100, 100), 2)

    def draw_landmarks(self, frame, landmarks):
         # This is handled by MediaPipe usually, but we can add custom drawing if needed
         pass
