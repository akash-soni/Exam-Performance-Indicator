from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


# edit below variables as per your requirements -
REPO_NAME = "Exam-Performance-Indicator"
AUTHOR_USER_NAME = "akash-soni"


setup(
    name=REPO_NAME,
    version="0.0.1",
    author=AUTHOR_USER_NAME,
    description="Simple ML project to check student performance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="aakashsoni.official@gmail.com",
    packages=find_packages(),
    license="MIT",
)
