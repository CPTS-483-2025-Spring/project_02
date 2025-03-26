import numpy as np
import math
import rclpy
import tf2_ros
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf_transformations import quaternion_matrix


def rotation_matrix_x(theta):
    """
    Compute the 3D rotation matrix around the x-axis.
    
    Parameters:
    theta (float): Rotation angle in radians.
    
    Returns:
    np.array: 3x3 rotation matrix.
    """
    return np.array([
        [1, 0, 0],
        [0, math.cos(theta), -math.sin(theta)],
        [0, math.sin(theta), math.cos(theta)]
    ])

def rotation_matrix_y(theta):
    """
    Compute the 3D rotation matrix around the y-axis.
    
    Parameters:
    theta (float): Rotation angle in radians.
    
    Returns:
    np.array: 3x3 rotation matrix.
    """
    return np.array([
        [math.cos(theta), 0, math.sin(theta)],
        [0, 1, 0],
        [-math.sin(theta), 0, math.cos(theta)]
    ])

def rotation_matrix_z(theta):
    """
    Compute the 3D rotation matrix around the z-axis.
    
    Parameters:
    theta (float): Rotation angle in radians.
    
    Returns:
    np.array: 3x3 rotation matrix.
    """
    return [np.array([
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1]
        ])]

def transformation_matrix(rotation_matrix, translation_vector):
    """
    Create a 4x4 homogeneous transformation matrix from a 3x3 rotation matrix and a 3x1 translation vector.
    
    Parameters:
    rotation_matrix (np.array): 3x3 rotation matrix.
    translation_vector (np.array): 3x1 translation vector.
    
    Returns:
    np.array: 4x4 homogeneous transformation matrix.
    """
    # Ensure the inputs are numpy arrays
    rotation_matrix = np.array(rotation_matrix)
    translation_vector = np.array(translation_vector)
    
    # Create the transformation matrix
    transformation_matrix = np.eye(4)  # Start with a 4x4 identity matrix
    transformation_matrix[:3, :3] = rotation_matrix  # Set the top-left 3x3 block to the rotation matrix
    transformation_matrix[:3, 3] = translation_vector  # Set the top-right 3x1 block to the translation vector
    
    return transformation_matrix

def forward_kinematics_franka(joint_angles):
    """
    Compute the forward kinematics for franka arm (panda_ee frame) using transformation matrices.
    
    Parameters:
    joint_angles (list): List of joint angles in radians [theta1, ..., theta7].
    
    Returns:
    np.array: transformation matrix of the end-effector.
    """
    
    # Rotation matrices for each joint
    theta1, theta2, theta3, theta4, theta5, theta6, theta7 = joint_angles
    R01 = rotation_matrix_z(theta1)  # First joint rotates around z-axis
    R12 = rotation_matrix_x(-math.pi/2) @ rotation_matrix_z(theta2)
    R23 = rotation_matrix_x(math.pi/2) @ rotation_matrix_z(theta3)
    R34 = rotation_matrix_x(math.pi/2) @ rotation_matrix_z(theta4)
    R45 = rotation_matrix_x(-math.pi/2) @ rotation_matrix_z(theta5)
    R56 = rotation_matrix_x(math.pi/2) @ rotation_matrix_z(theta6)
    R67 = rotation_matrix_x(math.pi/2) @ rotation_matrix_z(theta7)
    R78 = np.eye(3)
    R8ee = rotation_matrix_z(-0.785)
    
    # Translation vectors for each link
    p01 = np.array([0, 0, 0.333])  # First link translates along z-axis
    p12 = np.array([0, 0, 0])
    p23 = np.array([0, -0.316, 0])
    p34 = np.array([0.0825, 0, 0])
    p45 = np.array([-0.0825, 0.384, 0])
    p56 = np.array([0, 0, 0])
    p67 = np.array([0.088, 0, 0])
    p78 = np.array([0, 0, 0.107])
    p8ee = np.array([0, 0, 0.103])

    # Transformation matrix for each link
    T01 = transformation_matrix(R01, p01)
    T12 = transformation_matrix(R12, p12)
    T23 = transformation_matrix(R23, p23)
    T34 = transformation_matrix(R34, p34)
    T45 = transformation_matrix(R45, p45)
    T56 = transformation_matrix(R56, p56)
    T67 = transformation_matrix(R67, p67)
    T78 = transformation_matrix(R78, p78)
    T8ee = transformation_matrix(R8ee, p8ee)

    # print("T01:\n", T01)
    # print("T12:\n", T12)
    # print("T23:\n", T23)
    # print("T34:\n", T34)
    # print("T45:\n", T45)
    # print("T56:\n", T56)
    # print("T67:\n", T67)
    # print("T78:\n", T78)
    
    # Compute the end-effector position
    ee_T = T01 @ T12 @ T23 @ T34 @ T45 @ T56 @ T67 @ T78 @ T8ee
    
    return ee_T


def inverse_kinematics_franka(ee_T):
    return np.array([0] * 7)


class FKNode(Node):

    def __init__(self):
        super().__init__('fk_calculation_node')
        self.subscription = self.create_subscription(
            JointState,
            'joint_states',  # Replace with your topic name
            self.listener_callback,
            10)
        self.subscription  # Prevent unused variable warning

    def listener_callback(self, msg):
        # get joint angles
        joint_names = msg.name
        print(joint_names)
        joint_names_in_order = ["panda_joint1",
                                "panda_joint2",
                                "panda_joint3",
                                "panda_joint4",
                                "panda_joint5",
                                "panda_joint6",
                                "panda_joint7"]
        indices = [joint_names.index(name) for name in joint_names_in_order]
        joint_angles = [msg.position[i] for i in indices]
        ee_T = forward_kinematics_franka(joint_angles)  # compute FK
        ee_round = np.round(ee_T, 3)
        self.get_logger().info(f'Joint Angles 1-7 : {joint_angles}')
        self.get_logger().info(f'EE Transformation Matrix:: {ee_round}')


class IKNode(Node):

    def __init__(self):
        super().__init__('ik_calculation_node')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Call on_timer function every second
        self.timer = self.create_timer(1.0, self.on_timer)

    def on_timer(self):
        try:
            # Get the transform
            transform = self.tf_buffer.lookup_transform(
                "panda_link0", "panda_ee", rclpy.time.Time())
            
            # Extract translation and rotation
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            
            # Convert to transformation matrix (using numpy)
            self.get_logger().info(f"Translation: [{translation.x}, {translation.y}, {translation.z}]")
            self.get_logger().info(f"Rotation (quaternion): [{rotation.x}, {rotation.y}, {rotation.z}, {rotation.w}]")
            
            matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
            matrix[0:3, 3] = [translation.x, translation.y, translation.z]

            self.get_logger().info(f'EE Transformation Matrix:: \n {np.round(matrix, 3)}')

            # call franka IK function
            angles = inverse_kinematics_franka(matrix)
            self.get_logger().info(f'Joints angles:: \n {np.round(angles, 3)}')

            # validate with FK function
            ee_FK = forward_kinematics_franka(angles)
            self.get_logger().info(f'EE Transformation Matrix from FK:: \n {np.round(ee_FK, 3)}')

            
        except tf2_ros.LookupException as e:
            self.get_logger().error(f"Lookup failed: {e}")
        except tf2_ros.ConnectivityException as e:
            self.get_logger().error(f"Connectivity issue: {e}")
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().error(f"Extrapolation issue: {e}")


def main(args=None):
    rclpy.init(args=args)
    calculation_node = IKNode()
    rclpy.spin(calculation_node)
    calculation_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()