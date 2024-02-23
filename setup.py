import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pymccorrelation",
    version="0.2.6",
    author="George C. Privon",
    author_email="gprivon@nrao.edu",
    description="Compute correlation coefficients with uncertainties",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/privong/pymccorrelation",
    packages=setuptools.find_packages(),
    install_requires=['numpy>=1.17',
                      'scipy'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.6',
)
