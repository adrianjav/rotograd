from setuptools import setup, find_packages
import os
import re

DOCS_REQUIRES = [
    "sphinx",
    "sphinx-autodoc-typehints",
    "sphinx-rtd-theme",
]

classifiers = [
    'Development Status :: 3 - Alpha',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.7',
    'License :: OSI Approved :: MIT License',
    'Topic :: Scientific/Engineering',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'Intended Audience :: Science/Research',
    'Intended Audience :: Developers',
    'Operating System :: OS Independent',
]

# Get the long description from the README file
with open('README.md', 'r', encoding='utf8') as fh:
    long_description = fh.read()

# Get version string from module
init_path = os.path.join(os.path.dirname(__file__), 'rotograd/__init__.py')
with open(init_path, 'r', encoding='utf8') as f:
    version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

setup(
    name='rotograd',
    version=version,
    description='RotoGrad: Gradient Homogenization in Multitask Learning in Pytorch',
    author='Adrián Javaloy',
    author_email='adrian.javaloy@gmail.com',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/adrianjav/rotograd',
    classifiers=classifiers,
    keywords=['Multitask Learning', 'Gradient Alignment', 'Gradient Interference', 'Negative Transfer', 'Pytorch',
              'Positive Transfer', 'Gradient Conflict'],
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=['torch>=1.5', 'geotorch'],
    extras_require={'docs': DOCS_REQUIRES},
)
