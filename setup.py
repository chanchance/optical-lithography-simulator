from setuptools import setup, find_packages

setup(
    name='optical-litho-sim',
    version='0.1.0',
    description='Optical Lithography Simulator based on Pistor 2001 FDTD/Fourier Optics',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.19.0',
        'scipy>=1.6.0',
        'matplotlib>=3.3.0',
        'PyYAML>=5.4.0',
        'pdfplumber>=0.6.0',
    ],
    extras_require={
        'layout': ['gdstk>=0.8.0'],
        'gui': ['PySide6>=6.2.0'],
    },
)
