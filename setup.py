from os import path

from setuptools import setup

setup(name='ma_gym',
      version='0.0.1',
      url='https://github.com/koulanurag/ma-gym',
      py_modules=['ma_gym'],
      packages=['ma_gym'],
      author='Anurag Koul',
      author_email='koulanurag@gmail.com',
      long_description=open(path.join(path.abspath(path.dirname(__file__)), 'README.md')).read(),
      license=open(path.join(path.abspath(path.dirname(__file__)), 'LICENSE')).read(),
      install_requires=[x.strip() for x in
                        open(path.join(path.abspath(path.dirname(__file__)), 'requirements.txt')).readlines()],
      tests_require=['pytest'],
      python_requires='>=3.5',
      )
