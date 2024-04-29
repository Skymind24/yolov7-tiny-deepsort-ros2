import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler, EmitEvent, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessStart, OnExecutionComplete, OnProcessExit, OnShutdown
from launch.events import Shutdown
from launch.substitutions import FindExecutable, LaunchConfiguration, EnvironmentVariable, LocalSubstitution

def generate_launch_description():

    package_name='tracking_ros2' # CHANGE ME

    tracking_ros2_share_dir = get_package_share_directory(package_name)
    yolo_param_file = os.path.join(tracking_ros2_share_dir, 'config', 'yolov7-tiny-custom.yaml')

    ### Lanch File ###


    ### Node ###
    camera = Node(
        package='v4l2_camera',
        executable='v4l2_camera_node',
        output='screen',
        parameters=[{
            'image_size': [640,480],
            'camera_frame_id': 'camera_link_optical',
            'video_device': "/dev/video0"
            }]
    )

    detector_node = Node(
        package=package_name,
        executable='detector',
        remappings=[
            ('/image_raw', 'image_raw'),
        ],
        output='screen',
        parameters=[yolo_param_file]
    )

    tracker_node = Node(
        package=package_name,
        executable='tracker',
        remappings=[
            ('/image_raw', 'image_raw'),
            ('/yolo_detection/detector/bounding_boxes', 'yolo_detection/detector/bounding_boxes'),
        ],
        output='screen'
    )


    ### ExecuteProcess ###
    rqt = ExecuteProcess(
        cmd=["rqt"], 
        output="screen",
        shell=True
    )


    ### Event Handlers ###


    return LaunchDescription([

        camera,
        detector_node,
        tracker_node,
        rqt,
        RegisterEventHandler(
            OnShutdown(
                on_shutdown=[LogInfo(
                    msg=['Launch was asked to shutdown: ',
                        LocalSubstitution('event.reason')]
                )]
            )
        ),
    ])