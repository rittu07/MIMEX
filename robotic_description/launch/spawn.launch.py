from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    # Package paths
    pkg_desc = get_package_share_directory('robotic_description')
    pkg_gazebo = get_package_share_directory('gazebo_ros')

    # URDF path
    urdf_file = os.path.join(pkg_desc, 'urdf', 'robot.urdf')

    return LaunchDescription([

        # 1️ Start Gazebo
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_gazebo, 'launch', 'gazebo.launch.py')
            )
        ),

        # 2️Publish robot description
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': open(urdf_file).read()
            }]
        ),

        # 3 Spawn robot in Gazebo
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=[
                '-topic', 'robot_description',
                '-entity', 'robotic_arm'
            ],
            output='screen'
        )
    ])
