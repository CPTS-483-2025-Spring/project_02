from setuptools import find_packages, setup
from glob import glob

package_name = 'project_02'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/urdf', glob('urdf/*.urdf')),
        ('share/' + package_name + '/urdf/meshes/collision', glob('urdf/meshes/collision/*.stl')),
        ('share/' + package_name + '/urdf/meshes/visual', glob('urdf/meshes/visual/*.dae')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sihui',
    maintainer_email='sihui@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'ik_node = project_02.franka_ik:main'
        ],
    },
)
