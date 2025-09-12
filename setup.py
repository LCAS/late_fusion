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
            'late_fusion_scripts',
            'late_fusion_scripts.*'],
        exclude=['test']),

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ernstmv',
    maintainer_email='ernestoroque777@gmail.com',
    description='Implementation of a late fusion detector using 2d image-based and 3d lidar-based detections',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'late_fusion_node = late_fusion_pkg.late_fusion_node:main'
        ],
    },
)
