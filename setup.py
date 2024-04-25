from setuptools import find_packages, setup

with open("README-pypi.md", "r") as f:
    long_desc = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().strip().split("\n")

VERSION = '0.1.0'
setup(
    name="legoai",
    version=VERSION,
    description="An open-source package from LEGOAI for identifying data types",
    packages=find_packages(),
    package_data= {
        '': ['config.yaml','model/dependant/datatype_l1_identification/*','model/model_objects/datatype_l1_identification/*.pkl'],
       },
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/LEGOAI-TECHNOLOGIES/Data-Type-Identifier",
    author="LEGOAI Team",
    author_email="contactus@legoai.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Environment :: Console",
    ],
    install_requires=requirements,
    extras_require={
        "dev": ["twine>=4.0.2"],
    },
    python_requires=">=3.10",
)
