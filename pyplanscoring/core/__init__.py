import functools

import numba as nb

# add numba global compilation directives
njit = functools.partial(nb.njit, cache=False, nogil=True)
# njit = functools.partial(nb.njit, fastmath=True, cache=False, nogil=True, parallel=True)

# add fast-math
# if int(nb.__version__.split('.')[1]) >= 34:
#     njit = functools.partial(nb.njit, fastmath=True, cache=False, nogil=True, parallel=True)
# else:
#     njit = functools.partial(nb.njit, cache=False, nogil=True)

# njit = functools.partial(nb.njit, cache=False, nogil=True)
