from setuptools.command.test import test as TestCommand
import sys

try:  # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError:  # for pip <= 9.0.3
    from pip.req import parse_requirements

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

install_reqs = parse_requirements('requirements.txt', session=False)

# parse_requirements is not really intended to be used like this. we should probably
# find a better way to start our apps correctly and do not demand on installing
# ourselves...

reqs = [str(ir.req) for ir in install_reqs]
# reqs.remove("None")

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
    install_requires=reqs,
    zip_safe=False,
    keywords='ml_heat',
    test_suite='tests',
    tests_require=test_requirements,
    cmdclass={'test': PyTest},
)