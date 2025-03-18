from setuptools import setup, find_packages
import os

# Determine the platform-specific binary directory
bin_dir = os.path.join('bin', 'energibridge')

setup(
    name='pyEnergiBridge',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': [bin_dir],
    },
    install_requires=[
    ],
    extras_require={
        'ipython': ['ipython']
    },
    # Metadata
    author='Luis Cruz',
    author_email='luismirandacruz@gmail.com',
    # description='A description of my_library',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/luiscruz/pyEnergiBridge',
)
