from setuptools.command.test import test as TestCommand
import sys

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    'pyarrow',
    'h5py',
    'pycodestyle',
    'tqdm',
    'pandas',
    'numpy',
    'scipy',
    'sklearn',
    'torch',
    'matplotlib',
    'tables',
    'marshmallow<3.0',
    'jupyterlab',
    'ipympl',
    'statsmodels',
    'xgboost',
    'cattledb',
    'anthilldb',
    'sxutils',
    'sxapi'
]

test_requirements = [
    "pytest",
    "mock"
]


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


setup(
    name='ml_heat',
    version='v1.0',
    description="heat_detection_sandbox ",
    long_description=readme,
    author="Sebastian Grill",
    author_email='sebastian.grill@smaxtec.com',
    url='https://anthill.smaxtec.com',
    packages=[
        'ml_heat'
    ],
    package_dir={'ml_heat': 'ml_heat'},
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='ml_heat',
    test_suite='tests',
    tests_require=test_requirements,
    cmdclass={'test': PyTest},
)
