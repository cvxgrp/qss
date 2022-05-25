from setuptools import setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="qss",
    version="0.1.1",
    author="Luke Volpatti",
    description="QSS: the quadratic-separable solver",
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
