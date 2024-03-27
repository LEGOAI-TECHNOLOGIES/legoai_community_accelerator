from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_desc = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().strip().split("\n")

setup(
    name="legoai",
    version="0.0.10",
    description="",
    package_dir={"": "legoai"},
    packages=find_packages(where="legoai"),
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="",
    author="LEGOAI Team",
    author_email="",
    license="",
    classifiers=[
        "License :: ::",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
    install_required=requirements,
    extras_require={
        "dev": ["twine >= 4.0.2"]
    },
    python_requires=">=3.10",
)