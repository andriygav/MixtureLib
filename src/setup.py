import io
from setuptools import setup, find_packages

def read(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


__version__ = 0.0.1
readme = read('README.md')
requirements = read('requirements.txt')


setup(
    # metadata
    name='MixtureLib',
    version=__version__,
    author='Andrey Grabovoy',
    description='Lib for Mixture of Experts and Mixture of Models',
    long_description=readme,
    
    # options
    packages=find_packages(),
    install_requires=requirements,
)