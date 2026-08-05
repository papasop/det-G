from setuptools import setup


setup(
    name="r-to-law1",
    version="0.1.0",
    package_dir={"r_to_law1": "src/r_to_law1", "audits": "audits"},
    packages=["r_to_law1", "audits"],
    install_requires=["numpy>=2.0"],
)
