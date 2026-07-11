from setuptools import setup, find_packages

setup(
    name='factorminer',
    version='4.0.0',
    packages=find_packages(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'factorminer=core.cli:main',
        ],
    },
)
