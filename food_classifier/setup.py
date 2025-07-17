from setuptools import setup, find_packages

setup(
    name="foodclassifier",
    version="0.1.0",
    author="Guy Paiss",
    description="A deep learning project to classify images from the Food101 dataset.",
    packages=find_packages(),
    install_requires=[
        "torch==1.11.0+cu113",
        "torchvision==0.12.0+cu113",
        "numpy==1.23.5",
        "scikit-learn",
        "matplotlib",
    ],
    python_requires=">=3.7",
)