import setuptools
import os

os.system("pip install git+https://github.com/openai/CLIP.git") 
os.system("pip install sentence-transformers")


setuptools.setup(
    name="ZSIC",
    version="1.0",
    author="PithiviDa @ Donkey Stereotype",
    author_email="",
    description="Zeroshot Image Classification",
    long_description="Zero Shot Image Classification equivalent for HuggingFace Zero Shot Text Classification - By Prithivi Da",
    url="https://github.com/PrithivirajDamodaran/ZSIC.git",
    packages=setuptools.find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: MIT",
        "Operating System :: OS Independent",
    ],
)
