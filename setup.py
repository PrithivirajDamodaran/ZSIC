import setuptools

setuptools.setup(
    name="ZSIC",
    version="1.0",
    author="Donkey Stereotype",
    author_email="",
    description="Zeroshot image classification",
    long_description="Zero Shot Image Classification equivalent for HuggingFace Zero Shot Text Classification - By Prithivi Da",
    url="https://github.com/PrithivirajDamodaran/ZSIC.git",
    packages=setuptools.find_packages(),
    install_requires=['ftfy' ,'regex', 'tqdm', 'torch',' torchvision', 'git+https://github.com/openai/CLIP.git']
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: MIT",
        "Operating System :: OS Independent",
    ],
)
