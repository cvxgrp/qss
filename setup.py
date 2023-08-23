from setuptools import setup
import subprocess

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# get all the git tags from the cmd line that follow our versioning pattern
git_tags = subprocess.Popen(
    ["git", "tag", "--list", "v*[0-9]", "--sort=version:refname"],
    stdout=subprocess.PIPE,
)
tags = git_tags.stdout.read()
git_tags.stdout.close()
tags = tags.decode("utf-8").split("\n")
tags.sort()

# PEP 440 won't accept the v in front, so here we remove it, strip the new line and decode the byte stream
VERSION_FROM_GIT_TAG = tags[-1][1:]

setup(
    name="qss",
    version=VERSION_FROM_GIT_TAG,
    author="Luke Volpatti",
    description="QSS: Quadratic-Separable Solver",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["numpy", "scipy", "qdldl", "cvxpy"],
    url="https://github.com/lukevolpatti/qss",
    project_urls={
        "Bug Tracker": "https://github.com/lukevolpatti/qss/issues",
    },
    license="Apache 2.0",
    packages=["qss"],
    python_requires=">=3.6",
)
