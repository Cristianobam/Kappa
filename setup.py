import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="kappalib", # Replace with your own username
    version="0.1a",
    author="Cristiano Moraes Bilacchi Azarias",
    author_email="cristiano.bilacchi@aluno.ufabc.edu.br",
    description="Kappa is a Python 3 plotting and statistical library which performs both graphics and Frequentists analyses. Kappa aims to create better-looking graphics using Matplotlib and brings the Jamovi/Jasp analyses to python.",
    long_description=long_description,
    url="https://github.com/Cristianobam/kappalib",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7.5',
    setup_requires=[
        "numpy>=1.17.3",
    ],
    install_requires=[
        "numpy>=1.17.3",
        "matplotlib>=3.1.1",
    ],
    include_package_data=True,
)
