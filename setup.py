"""
Serenity Force Field.
"""

import sys

from setuptools import find_packages, setup

import versioneer

needs_pytest = {"pytest", "test", "ptr"}.intersection(sys.argv)
pytest_runner = ["pytest-runner"] if needs_pytest else []

short_description = __doc__.split("\n")

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except IOError:
    long_description = "\n".join(short_description[2:])

setup(
    name="serenityff",
    author="rinikerlab",
    author_email="mlehner@ethz.ch",
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/rinikerlab/serenityff",
    setup_requires=[] + pytest_runner,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license="MIT",
    packages=find_packages(),
    # packages=find_namespace_packages(include=["serenityff/charge/*"]),
    include_package_data=True,
    keywords="molecular dynamics, force field, parametrization, nonbonded parameters, explainable ml",
    python_requires=">=3.7",
    entry_points={
        "openff.toolkit.plugins.handlers": [
            "SerenityFFCharge = serenityff.charge.utils.serenityff_charge_handler:SerenityFFChargeHandler"
        ]
    },
)
print("test setup.py done")
