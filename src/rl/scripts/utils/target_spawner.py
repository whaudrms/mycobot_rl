import os
import rospy
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, DeleteModel, DeleteModelRequest

class TargetSpawner:
    def __init__(self):
        # 여기서 ServiceProxy를 한 번만 만들어 둠
        self.delete_client = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.spawn_client  = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        
        # SDF 파일 경로
        self.sdf_path = "/home/tony/mycobot_rl/src/rl/sdf/cube_target.sdf"
        if not os.path.exists(self.sdf_path):
            rospy.logerr(f"SDF file not found: {self.sdf_path}")

    def spawn_target_model(self, model_name, x, y, z):
        # SDF 파일 읽기
        with open(self.sdf_path, 'r') as f:
            sdf_xml = f.read()
        
        # Delete (기존 model_name이 이미 존재하면 삭제)
        del_req = DeleteModelRequest()
        del_req.model_name = model_name
        try:
            self.delete_client(del_req)
        except rospy.ServiceException as e:
            rospy.logwarn(f"DeleteModel failed: {e}")
        
        # Spawn
        req = SpawnModelRequest()
        req.model_name = model_name
        req.model_xml  = sdf_xml
        req.robot_namespace = ""
        
        req.initial_pose.position.x = x
        req.initial_pose.position.y = y
        req.initial_pose.position.z = z
        req.reference_frame = "world"

        try:
            resp = self.spawn_client(req)
            rospy.loginfo(f"Spawn response: {resp.status_message}")
            rospy.loginfo(f"Target position: {req.initial_pose.position.x,req.initial_pose.position.y,req.initial_pose.position.z }")
        except rospy.ServiceException as e:
            rospy.logerr(f"Spawn service call failed: {e}")
