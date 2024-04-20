---------------
Contributions
---------------

Contributions to PyHoloscope are very welcome via pull requests.

New functionality should first be introduced as functions. Once tested and working, the functionality can then be added as an option to
the Holo class. New functionality should generally not change the default behaviour of the Holo class, or any existing functions, in order
to maintain backwards compatibility, but should be made available via additional keyword arguments or new functions. The exception to this is a bug fix correcting
an output that is unambigously incorrect. Changes to default behaviour should also not result in slower processing speed; PyHoloscope is the basis
of several GUIs for real-time imaging, and updates must not break this functionality.

All additions must be fully documented, both in the code and in the Sphinx documentation. Additional documentation of existing functions is also welcomed.


^^^^^^^^^^^^
Testing
^^^^^^^^^^^^

There are two sets of tests in the test folder, integration tests and unit tests. Both sets of tests can be run
using the run_all_tests.py script each folder. Tests in the unit_tests folder use the unittest pacakge and rely on asserts, 
while tests in the integration_tests folder are to test correctness of output and rely on manual
inspection of the output images.

After making any changes, ensure the unit tests pass and inspect the integration tests for any visual changes to the output.
If you have written new functions, please consider creating additional unit tests.