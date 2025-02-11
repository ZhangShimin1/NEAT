from setuptools import setup, find_packages

requirements = ["torch"]

with open("./requirements.txt", "r", encoding="utf-8") as fh:
    install_requires = fh.read()
    
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="acouspike",
    version="0.0.0.2",
    author="Zeyang Song",
    author_email="zeyang_song@u.nus.edu",
    description="A package for acoustic spike analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ZhangShimin1/AcouSpike",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6'
)
