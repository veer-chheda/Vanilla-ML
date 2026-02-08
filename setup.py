from setuptools import setup

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="vanillaml",
    version="0.1.0",
    author="Veer Chheda",
    author_email="veerchheda3525@gmail.com",
    description="A machine learning library implemented from scratch using only NumPy, Pandas and Python",
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/veer-chheda/vanillaml",
    package_dir={"vanillaml": "."},
    packages=["vanillaml", "vanillaml.utils", "vanillaml.algorithms"],
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence :: Machine Learning :: Deep Learning",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
)
