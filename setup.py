from setuptools import setup

package_name = 'crop_tracker'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'video_reader = crop_tracker.ros_nodes:main',
            'video_listener = crop_tracker.ros_nodes:main'
        ],
    },
)
