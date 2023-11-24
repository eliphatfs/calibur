import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()
with open("caliborn/version.py", "r") as fh:
    exec(fh.read())
    __version__: str


def packages():
    return setuptools.find_packages(exclude=['tests'])


setuptools.setup(
    name="caliborn",
    version=__version__,
    author="flandre.info",
    author_email="flandre@scarletx.cn",
    description="Toolbox for converting rotations, camera calibration matrices, transforms and spaces.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/eliphatfs/caliborn",
    packages=packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy', 'typing_extensions', 'opencv-contrib-python', "trimesh>=3.23.1"
    ],
    python_requires='~=3.6',
)
