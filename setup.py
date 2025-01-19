from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(filepath:str) -> List[str]:
    with open(filepath) as file:
        requirements = file.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)

    return requirements

setup(
    name='ML Project',
    version='0.0.1',
    description='Developing and Deploying an end-to-end machine learning system',
    author='Teja Cherukuri',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)