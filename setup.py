from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


## edit below variables as per your requirements -
REPO_NAME = "ML-project-template"
AUTHOR_USER_NAME = "akash-soni"


setup(
    name=REPO_NAME,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="A small package for MLflow app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="aakashsoni.official@gmail.com",
    packages=find_packages(),
    license="MIT",
)
