#!/usr/bin/env python
from urdf_parser_py.urdf import URDF
import PyKDL as kdl
from kdl_parser_py.urdf import treeFromUrdfModel
import numpy as np

class Kinematics:
    def __init__(self, urdf_path=None, base_link="base_link", end_link="link6_flange"):
        """
        초기화 시 URDF 파일을 로드하고 KDL Tree, Chain, FK solver를 생성합니다.
        
        :param urdf_path: URDF 파일 경로 
        :param base_link: 시작 링크 이름 (기본값 "base_link")
        :param end_link: 종료 링크 이름 (기본값 "link6_flange")
        """
        if urdf_path is None:
            urdf_path = "/home/tony/mycobot_rl/src/mycobot_ros/mycobot_description/urdf/mycobot.urdf"
        self.urdf_path = urdf_path
        self.base_link = base_link
        self.end_link = end_link

        # URDF 파일 로드
        self.robot_urdf = URDF.from_xml_file(self.urdf_path)
        
        # URDF -> KDL Tree 변환
        ok, self.kdl_tree = treeFromUrdfModel(self.robot_urdf)
        if not ok:
            raise RuntimeError("URDF -> KDL Tree 변환에 실패했습니다.")

        # KDL Tree에서 체인(chain) 생성
        self.kdl_chain = self.kdl_tree.getChain(self.base_link, self.end_link)

        # Forward Kinematics 계산용 solver 생성
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.kdl_chain)

        # 체인의 관절 개수
        self.num_joints = 6

    def forward_kinematics(self, q):
        """
        입력된 관절 각도(q)를 사용하여 Forward Kinematics를 계산하고, 
        엔드이펙터의 x, y, z 좌표를 NumPy 배열로 반환합니다.
        
        :param q: 관절 각도 리스트 (각도는 라디안 단위, 예: [-pi, pi] 범위의 6개 값)
        :return: 엔드이펙터 위치 [x, y, z] (NumPy 배열)
        """

        # KDL의 JntArray에 관절 각도 대입
        joint_positions = kdl.JntArray(self.num_joints)
        for i, angle_rad in enumerate(q):
            joint_positions[i] = angle_rad

        # FK 계산을 위한 Frame 객체 생성
        end_effector_frame = kdl.Frame()
        self.fk_solver.JntToCart(joint_positions, end_effector_frame)

        # x, y, z 좌표 추출
        x = end_effector_frame.p[0]
        y = end_effector_frame.p[1]
        z = end_effector_frame.p[2]

        # NumPy 배열로 반환
        return np.array([x, y, z])
