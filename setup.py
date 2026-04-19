from setuptools import setup
import os
from glob import glob

package_name = 'autonomous_car_pkg'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Team',
    maintainer_email='team@example.com',
    description='Autonomous car navigation with lane following and SLAM',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_detection_node   = autonomous_car_pkg.lane_detection_node:main',
            'navigation_node       = autonomous_car_pkg.navigation_node:main',
            'behavior_controller   = autonomous_car_pkg.behavior_controller:main',
            'lane_map_node         = autonomous_car_pkg.lane_map_node:main',
            'obstacle_avoidance    = autonomous_car_pkg.obstacle_avoidance:main',
        ],
    },
)
