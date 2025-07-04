#!/usr/bin/env python

from setuptools import setup

def requirements_from_file(file_name):
    return open(file_name).read().splitlines()

setup(
    name="maai",
    version="0.0.1",
    description="Realtime Voice Activity Projection (Realtime-VAP)",
    author="inokoj",
    author_email="inoue.koji.3x@kyoto-u.ac.jp",
    url="https://github.com/inokoj/VAP-Realtime",
    packages=["vap_realtime"],
    install_requires=requirements_from_file('requirements.txt'),
)