import setuptools

exec(open('kappalib/_version.py').read())

setuptools.setup(
    name="kappalib", # Replace with your own username
    version=__version__,#pylint: disable=E0602
    author="Cristiano Moraes Bilacchi Azarias",
    author_email="cristiano.bilacchi@aluno.ufabc.edu.br",
    description="Python plotting and statistical package",
    long_description='''
    Kappa is a Python 3 plotting and statistical library which performs both graphics
    and Frequentists analyses. Kappa aims to create better-looking graphics using Matplotlib
    and brings the Jamovi/Jasp analyses to python.''',
    url="https://github.com/Cristianobam/kappalib",
    packages=['kappalib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6.9',
    setup_requires=[
        "numpy>=1.17.3",
    ],
    install_requires=[
        "numpy>=1.17.3",
        "matplotlib>=3.1.1",
        "tabulate>=0.8.6"
    ],
    include_package_data=True,
)
