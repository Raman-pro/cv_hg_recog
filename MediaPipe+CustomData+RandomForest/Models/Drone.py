class Drone:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.yaw = 0

    def update_position(self, x, y, z, yaw):
        if self.x+x<400 and self.x+x>-400:
            self.x += x
        if self.y+y<400 and self.y+y>-400:
            self.y += y
        if self.z+z<200 and self.z+z>-200:
            self.z += z
        self.yaw += yaw

    def __str__(self):
        return f"Drone Position: (x={self.x}, y={self.y}, z={self.z}, yaw={self.yaw})"