from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='crop_tracker',
            namespace='Streamer',
            executable='video_streamer',
            name='load_images',
        ),
        Node(
            package='crop_tracker',
            namespace='Listener',
            executable='video_listener',
            name='listen_images',
        ),
   ])
