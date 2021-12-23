import setuptools

setuptools.setup(
    name='optimism',
    description='Rapid development platform for solid mechanics research using optimization tools',
    author="Michael Tupek and Brandon Talamini",
    author_email='btalami@sandia.gov', # todo: make an email list
    install_requires=['jax',
                      'jaxlib',
                      'scipy',
                      'matplotlib', # this is not strictly necessary
                      'netcdf4'],
    #tests_require=[], # could put chex and pytest here
    extras_require={'sparse': ['scikit-sparse']},
    python_requires='>=3.7',
    version='0.0.1')
