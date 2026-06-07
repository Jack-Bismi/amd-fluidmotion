from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [l.strip() for l in f if l.strip() and not l.startswith("#")]

setup(
    name="amd-fluidmotion",
    version="0.2.1",
    author="Indie Developer",
    author_email="dev@fluidmotion.ai",
    description="Real-Time AI Frame Interpolation & Super-Resolution for AMD GPUs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jack-Bismi/amd-fluidmotion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "fluidmotion=scripts.run_demo:main",
        ],
    },
)
