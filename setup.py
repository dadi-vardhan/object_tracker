import os
import sys
import glob 
from setuptools import setup
from setuptools import find_packages

package_name = 'crop_tracker'


setup(
    name=package_name,
    version='0.0.0',
    #packages=[package_name],
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        os.path.join('share', package_name, 'launch'),
        glob.glob('launch/*.launch.py'),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='dadi_vardhan',
    maintainer_email='vardhan.vishnu32@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'video_streamer = crop_tracker.ros_nodes:main',
            'video_listener = crop_tracker.ros_nodes:main'
        ],
    },
)
