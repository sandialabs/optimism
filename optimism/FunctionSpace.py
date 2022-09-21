from collections import namedtuple
import numpy as onp

import jax
import jax.numpy as np
from jax.scipy.linalg import solve

from optimism import Interpolants
from optimism import Mesh


FunctionSpace = namedtuple('FunctionSpace', ['shapes', 'vols', 'shapeGrads', 'mesh', 'quadratureRule'])

EssentialBC = namedtuple('EssentialBC', ['nodeSet', 'component'])


def construct_function_space_from_master_element(mesh, masterElement, mode2D='cartesian'):
    shapes = jax.vmap(lambda elConns, elShape: elShape, (0, None))(mesh.conns, masterElement.shapes)

    shapeGrads = jax.vmap(map_element_shape_grads, (None, 0, None))(mesh.coords, mesh.conns, masterElement)

    vols = jax.vmap(compute_element_volumes, (None, 0, None))(mesh.coords, mesh.conns, masterElement)

    return FunctionSpace(shapes, vols, shapeGrads, mesh, masterElement.quadratureRule)


def map_element_shape_grads(coordField, nodeOrdinals, masterElement):
    Xn = coordField.take(nodeOrdinals,0)
    v = Xn[masterElement.vertexNodes]
    J = np.column_stack((v[0] - v[2], v[1] - v[2]))
    return jax.vmap(lambda dN: solve(J.T, dN.T).T)(masterElement.shapeGradients) 


def compute_element_volumes(coordField, nodeOrdinals, masterElement):
    Xn = coordField.take(nodeOrdinals,0)
    v = Xn[masterElement.vertexNodes]
    jac = np.cross(v[1] - v[0], v[2] - v[0])
    return jac*masterElement.quadratureRule.wgauss


def construct_function_space(mesh, quadratureRule, mode2D='cartesian'):
    shapes = jax.vmap(compute_shape_values_on_element,
                  (None, 0, None, None))(mesh.coords, mesh.conns, mesh.nodalBasis, quadratureRule.xigauss)

    shapeGrads = compute_shape_grads(mesh.coords, mesh.conns, mesh.nodalBasis, quadratureRule)
    
    if mode2D == 'cartesian':
        vols = jax.vmap(compute_volumes_on_element,
                    (None, 0, None, None))(mesh.coords, mesh.conns, mesh.nodalBasis, quadratureRule)
    elif mode2D == 'axisymmetric':
        vols = jax.vmap(compute_axisymmetric_volumes_on_element,
                    (None, 0, 0, None, None))(mesh.coords, mesh.conns, shapes, mesh.nodalBasis, quadratureRule)        
    return FunctionSpace(shapes, vols, shapeGrads, mesh, quadratureRule)


# only supports cartesian and linear shape functions on tris, single integration point per element
def construct_weighted_function_space(mesh, quadratureRule, quadratureWeights=1.0):
    
    def normal_vector(edge):
        ba = edge[1]-edge[0]
        return np.array([ba[1], -ba[0]])
    
    def compute_volume(coords):
        vol = 0.
        for e in range(3):
            edge = [coords[e], coords[(e+1)%3]]
            normal = normal_vector(edge)
            vol += normal[0] * 0.5*(edge[0][0]+edge[1][0])
        return vol

    def compute_elem_volume(coordField, elem):
        return compute_volume(coordField.take(elem,0))

    def compute_elem_linear_shape_gradient(coordField, vol, elem):
        coords = coordField.take(elem,0)
        shape = np.zeros([3,2])
        halfVol = 0.5/vol
        for e in range(3):
            n1 = e
            n2 = (e+1)%3
            edge = [coords[n1], coords[n2]]
            normal = normal_vector(edge)
            shape = shape.at[np.array([n1,n2])].add(normal*halfVol)
        return shape.reshape((1,3,2))
    
    shapes = np.ones( (mesh.conns.shape[0], 1, 3) )

    vols = jax.vmap(compute_elem_volume, (None, 0))(mesh.coords, mesh.conns)
    shapeGrads = jax.vmap(compute_elem_linear_shape_gradient, (None, 0, 0))(mesh.coords, vols, mesh.conns)
    
    vols = np.reshape( vols, (vols.shape[0], 1) )
    vols = vols * quadratureWeights
        
    return FunctionSpace(shapes, vols, shapeGrads, mesh, quadratureRule)


def default_modify_element_gradient(elemGrads, elemShapes, elemVols, elemNodalDisps, elemNodalCoords):
    return elemGrads


def compute_field_gradient(functionSpace, nodalField, modify_element_gradient=default_modify_element_gradient):
    return jax.vmap(compute_element_field_gradient, (None,None,0,0,0,0,None))(nodalField, functionSpace.mesh.coords, functionSpace.shapes, functionSpace.shapeGrads, functionSpace.vols, functionSpace.mesh.conns, modify_element_gradient)


def interpolate_to_points(functionSpace, nodalField):
    return jax.vmap(interpolate_to_element_points, (None, 0, 0))(nodalField, functionSpace.shapes, functionSpace.mesh.conns)


def integrate_over_block(functionSpace, U, stateVars, func, block,
                         *params, modify_element_gradient=default_modify_element_gradient):
    """Integrates a density function over a block of the mesh.

    Args:
      functionSpace: Function space object to do the integration with.
      U: The vector of dofs for the primal field in the functional.
      stateVars: Internal state variable array.
      func: Lagrangian density function to integrate, Must have the signature
        ``func(u, dudx, q, x, *params) -> scalar``, where ``u`` is the primal field, ``q`` is the
        value of the internal variables, ``x`` is the current point coordinates, and ``*params`` is
        a variadic set of additional parameters, which correspond to the ``*params`` argument.
      block: Group of elements to integrate over. This is an array of element indices. For
        performance, the elements within the block should be numbered consecutively.
      *params: Optional parameter fields to pass into Lagrangian density function. These are
        represented as a single value per element.
      modify_element_gradient: Optional function that modifies the gradient at the element level.
        This can be to set the particular 2D mode, and additionally to enforce volume averaging
        on the gradient operator. This is a keyword-only argument.

    Returns:
      A scalar value for the integral of the density functional ``func`` integrated over the
      block of elements.
    """
    
    vals = evaluate_on_block(functionSpace, U, stateVars, func, block, *params, modify_element_gradient=modify_element_gradient)
    return np.dot(vals.ravel(), functionSpace.vols[block].ravel())


def evaluate_on_block(functionSpace, U, stateVars, func, block,
                      *params, modify_element_gradient=default_modify_element_gradient):
    """Evaluates a density function at every quadrature point in a block of the mesh.

    Args:
      functionSpace: Function space object to do the evaluation with.
      U: The vector of dofs for the primal field in the functional.
      stateVars: Internal state variable array.
      func: Lagrangian density function to evaluate, Must have the signature
        ``func(u, dudx, q, x, *params) -> scalar``, where ``u`` is the primal field, ``q`` is the
        value of the internal variables, ``x`` is the current point coordinates, and ``*params`` is
        a variadic set of additional parameters, which correspond to the ``*params`` argument.
      block: Group of elements to evaluate over. This is an array of element indices. For
        performance, the elements within the block should be numbered consecutively.
      *params: Optional parameter fields to pass into Lagrangian density function. These are
        represented as a single value per element.
      modify_element_gradient: Optional function that modifies the gradient at the element level.
        This can be to set the particular 2D mode, and additionally to enforce volume averaging
        on the gradient operator. This is a keyword-only argument.

    Returns:
      An array of shape (numElements, numQuadPtsPerElement) that contains the scalar values of the
      density functional ``func`` at every quadrature point in the block.
    """
    fs = functionSpace
    compute_elem_values = jax.vmap(evaluate_on_element, (None, None, 0, 0, 0, 0, 0, None, None, *tuple(0 for p in params)))
    
    blockValues = compute_elem_values(U, fs.mesh.coords, stateVars[block], fs.shapes[block],
                                      fs.shapeGrads[block], fs.vols[block],
                                      fs.mesh.conns[block], func, modify_element_gradient, *params)
    return blockValues


def integrate_element_from_local_field(elemNodalField, elemNodalCoords, elemStates, elemShapes, elemShapeGrads, elemVols, func, modify_element_gradient=default_modify_element_gradient):
    """Integrate over element with element nodal field as input.
    This allows element residuals and element stiffness matrices to computed.
    """
    elemVals = jax.vmap(interpolate_to_point, (None,0))(elemNodalField, elemShapes)
    elemGrads = jax.vmap(compute_quadrature_point_field_gradient, (None,0))(elemNodalField, elemShapeGrads)
    elemGrads = modify_element_gradient(elemGrads, elemShapes, elemVols, elemNodalField, elemNodalCoords)
    elemPoints = jax.vmap(interpolate_to_point, (None,0))(elemNodalCoords, elemShapes)
    fVals = jax.vmap(func)(elemVals, elemGrads, elemStates, elemPoints)
    return np.dot(fVals, elemVols)


def compute_element_field_gradient(U, coords, elemShapes, elemShapeGrads, elemVols, elemConnectivity, modify_element_gradient):
    elemNodalDisps = U[elemConnectivity]
    elemGrads = jax.vmap(compute_quadrature_point_field_gradient, (None, 0))(elemNodalDisps, elemShapeGrads)
    elemNodalCoords = coords[elemConnectivity]
    elemGrads = modify_element_gradient(elemGrads, elemShapes, elemVols, elemNodalDisps, elemNodalCoords)
    return elemGrads


def compute_quadrature_point_field_gradient(u, shapeGrad):
    dg = np.tensordot(u, shapeGrad, axes=[0,0])
    return dg


def interpolate_to_point(elementNodalValues, shape):
    return np.dot(shape, elementNodalValues)


def interpolate_to_element_points(U, elemShapes, elemConnectivity):
    elemU = U[elemConnectivity]
    return jax.vmap(interpolate_to_point, (None, 0))(elemU, elemShapes)


def integrate_element(U, coords, elemStates, elemShapes, elemShapeGrads, elemVols, elemConn, func, modify_element_gradient):
    elemVals = interpolate_to_element_points(U, elemShapes, elemConn)
    elemGrads = compute_element_field_gradient(U, coords, elemShapes, elemShapeGrads, elemVols, elemConn, modify_element_gradient)
    elemXs = interpolate_to_element_points(coords, elemShapes, elemConn)
    fVals = jax.vmap(func)(elemVals, elemGrads, elemStates, elemXs)
    return np.dot(fVals, elemVols)


def evaluate_on_element(U, coords, elemStates, elemShapes, elemShapeGrads, elemVols, elemConn, kernelFunc, modify_element_gradient, *params):
    elemVals = interpolate_to_element_points(U, elemShapes, elemConn)
    elemGrads = compute_element_field_gradient(U, coords, elemShapes, elemShapeGrads, elemVols, elemConn, modify_element_gradient)
    elemXs = interpolate_to_element_points(coords, elemShapes, elemConn)
    vmapArgs = 0, 0, 0, 0, *tuple(None for p in params)
    fVals = jax.vmap(kernelFunc, vmapArgs)(elemVals, elemGrads, elemStates, elemXs, *params)
    return fVals


def compute_shape_values_on_element(coordField, nodeOrdinals, master, evalPoints):
    return Interpolants.compute_shapes_on_tri(master, evalPoints)


def compute_volumes_on_element(coordField, nodeOrdinals, master, quadRule):
    Xn = coordField.take(nodeOrdinals,0)
    v = Xn[master.vertexNodes]
    jac = np.cross(v[1] - v[0], v[2] - v[0])
    return jac*quadRule.wgauss


def compute_axisymmetric_volumes_on_element(coordField, nodeOrdinals, elemShapes, master, quadRule):
    vols = compute_volumes_on_element(coordField, nodeOrdinals, master, quadRule)
    Xn = coordField.take(nodeOrdinals,0)
    Rs = elemShapes@Xn[:,0]
    return 2*np.pi*Rs*vols



def compute_shape_grads(coordField, conns, master, quadRule):
    masterShapeGrads = Interpolants.compute_shapeGrads_on_tri(master, quadRule.xigauss)
    return jax.vmap(compute_element_shape_grads,
                (None,0,None,None))(coordField, conns, master, masterShapeGrads)


def compute_element_shape_grads(coordField, nodeOrdinals, master, masterShapeGrads):
    Xn = coordField.take(nodeOrdinals,0)
    v = Xn[master.vertexNodes]
    J = np.column_stack((v[0] - v[2], v[1] - v[2]))
    return jax.vmap(lambda dN: solve(J.T, dN.T).T)(masterShapeGrads)   


def project_quadrature_field_to_element_field(functionSpace, quadField):
    return jax.vmap(average_quadrature_field_over_element)(quadField, functionSpace.vols)


def average_quadrature_field_over_element(elemQPData, vols):
    S = np.tensordot(vols, elemQPData, axes=[0,0])
    elVol = np.sum(vols)
    return S/elVol


class DofManager:
    def __init__(self, functionSpace, dim, EssentialBCs):
        self.fieldShape = Mesh.num_nodes(functionSpace.mesh), dim
        self.isBc = onp.full(self.fieldShape, False, dtype=bool)
        for ebc in EssentialBCs:
            self.isBc[functionSpace.mesh.nodeSets[ebc.nodeSet], ebc.component] = True
        self.isUnknown = ~self.isBc

        self.ids = np.arange(self.isBc.size).reshape(self.fieldShape)

        self.unknownIndices = self.ids[self.isUnknown]
        self.bcIndices = self.ids[self.isBc]

        ones = np.ones(self.isBc.size, dtype=int) * -1
        self.dofToUnknown = ones.at[self.unknownIndices].set(np.arange(self.unknownIndices.size)) 

        self.HessRowCoords, self.HessColCoords = self._make_hessian_coordinates(functionSpace.mesh.conns)

        self.hessian_bc_mask = self._make_hessian_bc_mask(functionSpace.mesh.conns)


    def get_bc_size(self):
        return np.sum(self.isBc).item() # item() method casts to Python int


    def get_unknown_size(self):
        return np.sum(self.isUnknown).item() # item() method casts to Python int


    def create_field(self, Uu, Ubc=0.0):
        U = np.zeros(self.isBc.shape).at[self.isBc].set(Ubc)
        return U.at[self.isUnknown].set(Uu)


    def get_bc_values(self, U):
        return U[self.isBc]


    def get_unknown_values(self, U):
        return U[self.isUnknown]


    def slice_unknowns_with_dof_indices(self, Uu, dofIndexSlice):
        i = self.isUnknown[dofIndexSlice]
        j = self.dofToUnknown.reshape(self.fieldShape)[dofIndexSlice]
        return Uu[j[i]]


    def _make_hessian_coordinates(self, conns):
        nElUnknowns = onp.zeros(conns.shape[0], dtype=int)
        nHessianEntries = 0
        for e, eNodes in enumerate(conns):
            elUnknownFlags = self.isUnknown[eNodes,:].ravel()
            nElUnknowns[e] = onp.sum(elUnknownFlags)
            nHessianEntries += onp.square(nElUnknowns[e])

        rowCoords = onp.zeros(nHessianEntries, dtype=int)
        colCoords = rowCoords.copy()
        rangeBegin = 0
        for e,eNodes in enumerate(conns):
            elDofs = self.ids[eNodes,:]
            elUnknownFlags = self.isUnknown[eNodes,:]
            elUnknowns = self.dofToUnknown[elDofs[elUnknownFlags]]
            elHessCoords = onp.tile(elUnknowns, (nElUnknowns[e],1))

            rangeEnd = rangeBegin + onp.square(nElUnknowns[e])

            rowCoords[rangeBegin:rangeEnd] = elHessCoords.ravel()
            colCoords[rangeBegin:rangeEnd] = elHessCoords.T.ravel()

            rangeBegin += np.square(nElUnknowns[e])
        return rowCoords, colCoords


    def _make_hessian_bc_mask(self, conns):
        nElements, nNodesPerElement = conns.shape
        nFields = self.ids.shape[1]
        nDofPerElement = nNodesPerElement*nFields

        hessian_bc_mask = onp.full((nElements,nDofPerElement,nDofPerElement),
                                   True, dtype=bool)
        for e, eNodes in enumerate(conns):
            eFlag = self.isBc[eNodes,:].ravel()
            hessian_bc_mask[e,eFlag,:] = False
            hessian_bc_mask[e,:,eFlag] = False
        return hessian_bc_mask
