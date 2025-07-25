from setuptools import find_packages, setup
import os
from glob import glob

package_name = "ur5_motion_planner"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/rl_config.yaml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "scipy",
        "transformations",
        "gymnasium",
        "stable-baselines3[extra]",
        "tensorboard",
        "torch",
        "PyYAML",
        "matplotlib",
        "seaborn",
        "pandas",
    ],
    zip_safe=True,
    maintainer="carlos",
    maintainer_email="ce.cesc01@gmail.com",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ik_motion_planner = ur5_motion_planner.ik_motion_planner:main",
            "validator = ur5_motion_planner.validator:main",
            "dataset_generator_node = ur5_motion_planner.dataset_generator_node:main",
            "env_node = ur5_motion_planner.env_node:main",
            "evaluate_rl = ur5_motion_planner.evaluate_rl:main",
            "evaluate_ik = ur5_motion_planner.evaluate_ik:main",
        ],
    },
)
