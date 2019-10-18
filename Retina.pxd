cimport numpy as np

cdef class Retina:
	cdef int _n
	cdef int[::1] _polarities
	cdef double[:,::1] _axons
	cdef double[:,::1] _dendrites
	cdef double[:,:,::1] _scene

	cdef void _build(self)
	cpdef void set_params(self, int[::1] polarities, double[:,::1] axons,
						double[:,::1] dendrites):
	cpdef void process(self, double[:,:,::1] buffer)