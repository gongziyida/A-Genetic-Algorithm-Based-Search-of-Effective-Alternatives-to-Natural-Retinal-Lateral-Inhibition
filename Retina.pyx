import numpy as np
cimport numpy as np

cdef class Retina:
	def __init__(self, int[::1] polarities, double[:,::1] axons,
				double[:,::1] dendrites, double[:,:,::1] scene):
		self._polarities = polarities
		self._axons = axons
		self._dendrites = dendrites
		self._n = polarities.shape[0]
		self.scene = scene

		self._build()

	cdef void _build(self):
		cdef Py_ssize_t i
		for i in range(self._n):
			pass
			# TODO: build the filters


	cpdef void set_params(self, int[::1] polarities, double[:,::1] axons,
						double[:,::1] dendrites):
		self.polarities = polarities
		self.axons = axons
		self.dendrites = dendrites
		self.n = polarities.shape[0]


	cpdef void process(self, double[:,:,::1] buffer):
		pass
		# TODO: a non-inplace modification of the image onto a buffer