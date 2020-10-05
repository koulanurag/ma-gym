from os import path

from setuptools import setup

extras = {
    'test': ['pytest', 'pytest_cases'],
    'develop': ['imageio'],
}

# Meta dependency groups.
extras['all'] = [item for group in extras.values() for item in group]

setup(name='ma_gym',
      version='0.0.1',
      description='A collection of multi agent environments based on OpenAI gym.',
      long_description=open(path.join(path.abspath(path.dirname(__file__)), 'README.md')).read(),
      url='https://github.com/koulanurag/ma-gym',
      author='Anurag Koul',
      author_email='koulanurag@gmail.com',
      license=open(path.join(path.abspath(path.dirname(__file__)), 'LICENSE')).read(),
      packages=['ma_gym'],
      py_modules=['ma_gym'],
      install_requires=[
          'scipy',
          'numpy>=1.16.4',
          'pyglet>=1.4.0,<=1.5.0',
          'cloudpickle>=1.2.0,<1.7.0',
          'gym>=0.14.0',
          'Pillow>=6.2.0',
      ],
      extras_require=extras,
      tests_require=extras['test'],
      python_requires='>=3.5',
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.8',
      ],
      )
