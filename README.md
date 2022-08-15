# OptimiSM: Computational solid mechanics made easy with Jax

## What is OptimiSM?

OptimiSM is a library for posing and solving problems in solid
mechanics using the finite element method.
The central theme of this project is exploring how to get better
performance and robustness by taking advantages of the tools of
variational calculus.
OptimiSM uses Lagrangian field theory to pose hard nonlinear solid
mechanics problems as optimization problems, and then uses powerful
optimization methods to solve them efficiently and reliably.  

To do this, OptimiSM relies on Google's
[JAX](https://github.com/google/jax) library for automatic
differentiation and just-in-time compiling for performance.

## Why use OptimiSM?

These days, there are lots of finite element software libraries out
there.  Why would you want to use OptimiSM? 

- OptimiSM is for **rapid development**: OptimiSM is written in Python and
  uses the NumPy/SciPy stack. This means that it's easy to read,
  understand, and extend. If you're like us, and you prefer working in
  Python/NumPy to C++, you'll find OptimiSM a more pleasant place to
  work (and play) than heavily abstracted finite element libraries,
  even the well-designed ones.  
  OptimiSM makes use of Jax's just-in-time compilation to get good
  performance, so the simplicity of Python coding doesn't condemn you
  to toy problems.
- OptimiSM provides **robust solvers**: OptimiSM takes a different
  approach than most finite element libraries.  *All* problems are
  formulated by encoding them in a scalar-valued functional and then
  minimizing that functional. This includes nonlinear phenomena like
  finite deformations and contact, and even irreversible (dissipative)
  phenomena like plasticity and viscoelasticity. A big motivation for
  creating this library was proving to others (and ourselves) that
  real-world, complex problems could be written in this way, and that
  it could pay off for solving hard problems. By imposing a
  minimization structure, the OptimiSM solvers can avoid stagnating in
  hard problems and also avoid converging to spurious unstable
  configurations. In other words, OptimiSM helps you find the solutions
  that *should* be out there and prevents you from finding "solutions"
  that really aren't solutions. Check out the [examples]() to see some
  cases that are difficult or impossible to solve correctly even with
  commerical codes.
- OptimiSM gives **sensitivities** for design optimization, inverse
  analysis, and training of machine learning models. 

## Installation instructions

At the moment, OptimiSM is meant to be used as a development package.
First, fork and clone the code repository from GitHub.
Next, you have a choice: you can pick a basic installation which
requires only a minimal set of dependencies, or the recommended
installation, which requires some additional packages. The main
difference of the recommended installation is that it requires the
`scikit-sparse` package, which provides a sparse Cholesky
preconditioner. This is needed if you want to run large-scale
problems; without it, you'll only be able to use a dense matrix
preconditioner (which is both slower and uses up much more memory).

- Basic installation: If you just want to try some examples out and
test-drive OptimiSM, install the basic installation by navigating into
the base project directory and executing

```bash
pip install -e .
```

- Recommended installation: The `scikit-sparse` package requires the
[SuiteSparse](https://people.engr.tamu.edu/davis/suitesparse.html)
library to be present. If you have access to a package manager on your
system, this is the easiest way to get it. On a Mac platform, 
this would be done with MacPorts by running

```bash
sudo port install SuiteSparse
```

or with Homebrew by

```bash
brew install suite-sparse
```

On a Fedora system, you would run
```bash
sudo dnf install suitesparse-devel
```

Of course, you could compile the source yourself if you wish. Check to
make sure the version you download is supported by the `scikit-sparse`
package. The source is available from the [SuiteSparse
website](https://people.engr.tamu.edu/davis/suitesparse.html) (a
GitHub link is also provided there).

Once the SuiteSparse library is in place, navigate into the
`optimism` directory and execute

```bash
pip install -e ".[sparse]"
```

Note that you can always start with the basic installation, and if you
want to switch to the recommended version later, you can just get
SuiteSpase and run the above recommended installation command to get
the additional functionality. You don't need to remove the basic
package first.

## Sample Installation on OSX using Homebrew

From the `optimism` directory:
```bash
brew install suite-sparse
brew install python-tk 

INC=/usr/local/Cellar/suite-sparse/5.11.0/include
LIB=/usr/local/Cellar/suite-sparse/5.11.0/lib
pip=/usr/local/opt/python/bin/pip3
SUITESPARSE_INCLUDE_DIR=$INC SUITESPARSE_LIBRARY_DIR=$LIB $pip install -e . sparse
```

## Citing OptimiSM

If you use OptimiSM in your research, please cite

```
@software{OptimiSM,
  author = {Michael R. Tupek and Brandon Talamini},
  title = {{OptimiSM}},
  url = {https://github.com/sandialabs/optimism},
  version = {0.0.1},
  year = {2021},
}
```
***TODO***: add citation for contact paper


## Reference documentation

For details about the OptimiSM API, see the [documentation]().

## Contact

OptimiSM was created and is maintained by Michael Tupek
<mrtupek@sandia.gov> and Brandon Talamini <btalami@sandia.gov>.

SCR#: 2709.0
