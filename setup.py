from setuptools import setup
setup(
    name='app-name',
    version='1.0',
    author='Tor, Olivia og Nicolai',
    description='',
    url='https://github.com/NicolaiVArends/BachelorThesis',
    keywords='',
    packages=['src'],
    install_requires=[
        'PyYAML',
        'pandas==0.23.3',
        'numpy>=1.14.5',
        'matplotlib>=2.2.0',
        'jupyter'
    ]
)