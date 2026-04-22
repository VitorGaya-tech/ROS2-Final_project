"""
full_system.launch.py
---------------------
Launches:
  1. slam_toolbox (online_async)  — SLAM mapping
  2. lane_detection_node          — white/yellow/orange line detection
  3. lane_map_node                — publishes lane trail into /map frame
  4. navigation_node              — PD lane follower
  5. obstacle_avoidance           — LiDAR-based stop/go
  6. behavior_controller          — top-level state machine

RUN (from your ROS2 workspace):
  ros2 launch autonomous_car_pkg full_system.launch.py

OPTIONAL ARGS:
  use_rviz:=true         — launch RViz with the preconfigured display
  slam_reset:=true       — force a fresh map (always true by default)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Caminho absoluto para o os.path.join (usado pelo slam_toolbox)
    pkg_dir = get_package_share_directory('autonomous_car_pkg')
    
    # Objeto de substituição (usado pelo RViz lá embaixo)
    pkg = FindPackageShare('autonomous_car_pkg')

    use_rviz_arg = DeclareLaunchArgument(
        'use_rviz', default_value='false',
        description='Launch RViz')

    # Monta o caminho exato para o YAML
    slam_params = os.path.join(pkg_dir, 'config', 'slam_toolbox.yaml')

    # ── slam_toolbox ─────────────────────────────────────────────
    slam_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[{
            'use_sim_time': False, # Muda para True se estiveres num simulador!
            'mode': 'mapping',
            'scan_topic': '/scan',
            'odom_topic': '/odometry/filtered',
            'map_frame': 'map',
            'base_frame': 'base_link',
            'odom_frame': 'odom',
            'resolution': 0.05,
            'max_laser_range': 5.0,
            'minimum_travel_distance': 0.05,
            'minimum_travel_heading': 0.1,
            'use_scan_matching': True,
            'use_scan_barycenter': True,
            'minimum_time_interval': 0.1,
            'do_loop_closing': True,
            'loop_search_maximum_distance': 3.0
        }
        ],
    )

    # ── Our nodes ─────────────────────────────────────────────────
    lane_detection = Node(
        package='autonomous_car_pkg',
        executable='lane_detection_node',
        name='lane_detection_node',
        output='screen',
    )

    lane_map = Node(
        package='autonomous_car_pkg',
        executable='lane_map_node',
        name='lane_map_node',
        output='screen',
    )

    navigation = Node(
        package='autonomous_car_pkg',
        executable='navigation_node',
        name='navigation_node',
        output='screen',
        parameters=[{
            'kp': 0.8,
            'kd': 0.15,
            'base_speed': 0.15,
            'max_turn': 1.2,
        }],
    )

    obstacle_avoidance = Node(
        package='autonomous_car_pkg',
        executable='obstacle_avoidance',
        name='obstacle_avoidance',
        output='screen',
        parameters=[{
            'stop_distance': 0.40,
            'cone_half_deg': 25.0,
        }],
    )

    behavior = Node(
        package='autonomous_car_pkg',
        executable='behavior_controller',
        name='behavior_controller',
        output='screen',
    )

    # ── RViz (optional) ───────────────────────────────────────────
    rviz_config = PathJoinSubstitution([pkg, 'config', 'autonomous_car.rviz'])
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('use_rviz')),
    )

    return LaunchDescription([
        use_rviz_arg,
        slam_node,
        lane_detection,
        lane_map,
        navigation,
        obstacle_avoidance,
        behavior,
        rviz,
    ])