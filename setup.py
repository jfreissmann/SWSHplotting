from setuptools import setup

with open('README.rst', 'r') as f:
    long_discription = f.read()

setup(
    name='SWSHplotting',
    version='0.0.1',
    author='Jonas Freißmann',
    author_email='jonas.freissmann@hs-flensburg.de',
    description='Plotting package for SWSH project.',
    long_discription=long_discription,
    long_discription_content_type='text/x-rst',
    url='https://github.com/jfreissmann/SWSHplotting',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License'
        ],
    py_modules=['SWSHplotting'],
    package_dir={'': 'src'}
)
