from setuptools import setup, find_packages


setup(
    name = 'tetris_dqn',
    version= '0.1',
    packages= find_packages(),
    install_requires = [
        'torch',
        'numpy',
        'gym',
        'pygame',
        'matplotlib',
    ],
    entry_points = {
        'console_scripts':[
            'tetris_dqn=tetris_dqn:main'
        ],
    },
)