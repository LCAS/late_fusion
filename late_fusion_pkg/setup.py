from glob import glob
from setuptools import find_packages, setup

package_name = 'late_fusion_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(
        include=[
            'late_fusion_pkg',
            'late_fusion_pkg.*',
            'scripts'
            ],
        exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'numpy',
        'ultralytics',
        'sensor_msgs_py'
        ],
    zip_safe=True,
    maintainer='user',
    maintainer_email='ernestoroque777@gmail.com',
    description='Implementation of an late fusion algorithm for camera images and lidar point clouds in agricultural enviorments',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'late_fusion_node = late_fusion_pkg.late_fusion_node:main'
        ],
    },
)
