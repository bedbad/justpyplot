# Copyright (c) 2023 bedbad
from setuptools import setup, find_packages

setup(
    name='justpyplot',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'opencv-python',
    ],
    author='bedbad',
    author_email='antonyuk@bu.edu',
    description='Get your plot in you array, plot fast',
    long_description='''Justplot lets you hang your plot on live videofeed,'''
                     ''' display anyhow you would display an array image,'''
                     ''' save as usual plot, or record in a movie that is too fast to see''',
    long_description_content_type='text/markdown',
    url='https://github.com/your_username/your_package',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
