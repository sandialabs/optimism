import setuptools

setuptools.setup(
    name='optimism',
    description='Rapid development platform for solid mechanics research using optimization tools',
    author="Michael Tupek and Brandon Talamini",
    author_email='talamini1@llnl.gov', # todo: make an email list
    install_requires=['equinox',
                      'jax[cpu]',
                      'jaxtyping',
                      'matplotlib', # this is not strictly necessary
                      'netcdf4',
                      'scipy'],
    #tests_require=[], # could put chex and pytest here
    extras_require={'sparse': ['scikit-sparse'],
                    'test': ['pytest', 'pytest-cov', 'pytest-xdist'],
                    'docs': ['sphinx', 'sphinx-copybutton', 'sphinx-rtd-theme', 'sphinxcontrib-bibtex', 'sphinxcontrib-napoleon']},
    python_requires='>=3.13',
    version='0.0.1',
    license='MIT',
    url='https://github.com/sandialabs/optimism'
)
