from setuptools import setup,find_packages
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as fileobj:
        requirements=fileobj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements

setup(
    name="HeartDiseasePrediction",
    version="0.0.0",
    author="kp",
    author_email="kp@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)