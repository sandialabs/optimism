Installation
============
To install the vanilla version of optimism, simply run the following commands
from within the optimism git repo directory that you just cloned

``python -m pip install -e .``

To install with documentation run

``python -m pip install -e ".[doc]"``

To install with sparse matrix capabilities run

``python -m pip install -e ".[sparse]"``

To install with test capabilities run 

``python -m pip install -e ".[test]"``

And finally, to install with all of the extras run

``python -m pip install -e ".[doc, sparse, test]"``