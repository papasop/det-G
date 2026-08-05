from setuptools import setup


setup(
    name="r-to-law1",
    version="0.1.1",
    package_dir={
        "r_to_law1": "src/r_to_law1",
        "realizability": "src/realizability",
        "audits": "audits",
    },
    packages=["r_to_law1", "realizability", "audits"],
    python_requires=">=3.10",
    install_requires=["numpy>=2.0"],
)
