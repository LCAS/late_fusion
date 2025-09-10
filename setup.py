from glob import glob
from setuptools import find_packages, setup

package_name = 'late_fusion_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(
        include=[
            'late_fusion_pkg',
            'late_fusion_pkg.*',
            'scripts',
            'scripts.*'],
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
    maintainer='ros',
    maintainer_email='ros@todo.todo',
    description='TODO: Package description',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'late_fusion_node = late_fusion_pkg.late_fusion_node:main'
        ],
    },
)
