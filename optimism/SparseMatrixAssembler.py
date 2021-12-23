import numpy as onp
from scipy.sparse import coo_matrix

from optimism.JaxConfig import *


def assemble_sparse_stiffness_matrix(kValues, conns, dofManager):
     nElements, nNodesPerElement = conns.shape
     nFields = kValues.shape[2]
     nDofPerElement = nNodesPerElement*nFields
     
     kValues = kValues.reshape((nElements,nDofPerElement,nDofPerElement))

     nUnknowns = dofManager.unknownIndices.size
     
     K = coo_matrix((kValues[dofManager.hessian_bc_mask], (dofManager.HessRowCoords, dofManager.HessColCoords)), shape = (nUnknowns, nUnknowns))
     return K.tocsc()


