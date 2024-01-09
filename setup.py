import sys
import os
from distutils.core import setup

setup(
    name="two_class_density_scatterplot",
    version=0.2,
    packages=[""],
    package_dir={"": "."},
    install_requires=["scipy", "numpy", "pandas", "coloraide", "matplotlib"],
    py_modules=["two_class_density_scatterplot"],
)
