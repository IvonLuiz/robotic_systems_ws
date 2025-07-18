from setuptools import find_packages, setup

package_name = "ur5_motion_planner"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "scipy",
        "transformations",
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
        ],
    },
)
