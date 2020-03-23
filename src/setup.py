import io
from setuptools import setup, find_packages

from mixturelib import __version__

def read(file_path):
    with io.open(file_path, 'r', encoding='utf-8') as f:
        return f.read()


readme = read('README.rst')
requirements = read('requirements.txt')


setup(
    # metadata
    name='mixturelib',
    version=__version__,
    license='MIT',
    author='Andrey Grabovoy',
    author_email="grabovoy.av@phystech.edu",
    description='mixture lib, python package',
    long_description=readme,
    url='https://github.com/andriygav/MixtureLib',

    # options
    packages=find_packages(),
    python_requires='==3.6.*',
    install_requires=requirements,
)