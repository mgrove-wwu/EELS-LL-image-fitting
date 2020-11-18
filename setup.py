import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pl_analysis", # Replace with your own username
    version="0.0.5",
    author="Maximilian Grove",
    author_email="m_grov01@uni-muenster.de",
    description="Plasmon loss analysis by 2D - or 3D - (stack) images of EELS spectra",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mgrove-wwu/EELS-LL-image-fitting",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)