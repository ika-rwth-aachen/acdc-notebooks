from squaternion import Quaternion

class TrajectoryPoint2D:
    def __init__(self):
        self.x=None
        self.y=None
        self.psi=None
        self.t=None
                             
    def from_odometry(self, odometry):
        self.x = odometry.pose.pose.position.x
        self.y = odometry.pose.pose.position.y
        self.t = odometry.header.stamp.sec+odometry.header.stamp.nanosec*1e-9
        self.psi = self.quaternion_to_yaw(odometry.pose.pose.orientation)
        
    def from_pose(self, pose):
        self.x = pose.pose.position.x
        self.y = pose.pose.position.y
        self.t = pose.header.stamp.sec+pose.header.stamp.nanosec*1e-9
        self.psi = self.quaternion_to_yaw(pose.pose.orientation)
        
    def quaternion_to_yaw(self, quaternion):
        q = Quaternion(quaternion.w,quaternion.x,quaternion.y,quaternion.z)
        e = q.to_euler(degrees=True)
        return e[2]