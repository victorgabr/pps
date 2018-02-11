import functools

import numba as nb

# add numba global compilation directives
njit = functools.partial(nb.njit, cache=False, nogil=True)
# njit = functools.partial(nb.njit, fastmath=True, cache=False, nogil=True, parallel=True)
