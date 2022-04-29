import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.test")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(
    name="spaceweight",
    version="0.0.1",
    license='GNU Lesser General Public License, version 3 (LGPLv3)',
    description="Python tools for geographical spatial weighting",
    author="Wenjie Lei",
    author_email="lei@princeton.edu",
    url="https://github.com/wjlei1990/spaceweight",
    packages=find_packages("src"),
    package_dir={"": "src"},
    tests_require=['pytest'],
    cmdclass={'test': PyTest},
    zip_safe=False,
    classifiers=[
        # complete classifier list:
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)"
    ],
    keywords=[
        "geographic", "space", "weighting strategy"
    ],
    install_requires=[
        "numpy", "obspy>=1.0.0", "flake8>=3.0", "pytest", "nose",
        "future>=0.14.1", "scipy", "matplotlib"
    ],
    extras_require={
        "docs": ["sphinx"]
    }
)
