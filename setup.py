import pathlib
from setuptools import setup

# Path to main package directory
HERE = pathlib.Path(__file__).parent

# Text of the README file
README = (HERE / "README.md").read_text()

setup(
    name="locomotion",
    version="1.1.0",
    packages=["locomotion"],
    install_requires=["numpy>=1.16.2", "plotly>=4.4.1", "scipy>=1.2.1",
                      "dtw-python>=1.1.4", "igl>=0.4.1"],

    # metadata to display on PyPI
    author="Mechanisms Underlying Behavior Lab",
    author_email="mechunderlyingbehavior@gmail.com",
    description="Computational geometric tools for quantitative comparison of locomotory behavior",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/mechunderlyingbehavior/locomotion",
    project_urls={
        "Lab Website": "https://https://mechunderlyingbehavior.wordpress.com/"
    }
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7"
    ]
)
