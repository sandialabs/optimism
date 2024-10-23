from ..MeshFixture import MeshFixture
from collections import namedtuple
import numpy as onp

class FiniteDifferenceFixture(MeshFixture):
    def assertFiniteDifferenceCheckHasVShape(self, errors, tolerance=1e-6):
        minError = min(errors)
        self.assertLess(minError, tolerance, "Smallest finite difference error not less than tolerance.")
        self.assertLess(minError, errors[0], "Finite difference error does not decrease from initial step size.")
        self.assertLess(minError, errors[-1], "Finite difference error does not increase after reaching minimum. Try more finite difference steps.")

    def build_direction_vector(self, numDesignVars, seed=123):

        onp.random.seed(seed)
        directionVector = onp.random.uniform(-1.0, 1.0, numDesignVars)
        normVector = directionVector / onp.linalg.norm(directionVector)

        return onp.array(normVector)

    def compute_finite_difference_error(self, stepSize, initialParameters):
        storedState = self.forward_solve(initialParameters)
        originalObjective = self.compute_objective_function(storedState, initialParameters) 
        gradient = self.compute_gradient(storedState, initialParameters) 

        directionVector = self.build_direction_vector(initialParameters.shape[0])
        directionalDerivative = onp.tensordot(directionVector, gradient, axes=1)

        perturbedParameters = initialParameters + stepSize * directionVector
        storedState = self.forward_solve(perturbedParameters)
        perturbedObjective = self.compute_objective_function(storedState, perturbedParameters) 

        fd_value = (perturbedObjective - originalObjective) / stepSize
        error = abs(directionalDerivative - fd_value)
        
        return error

    def compute_finite_difference_errors(self, stepSize, steps, initialParameters, printOutput=True):
        storedState = self.forward_solve(initialParameters)
        originalObjective = self.compute_objective_function(storedState, initialParameters) 
        gradient = self.compute_gradient(storedState, initialParameters) 

        directionVector = self.build_direction_vector(initialParameters.shape[0])
        directionalDerivative = onp.tensordot(directionVector, gradient, axes=1)

        fd_values = []
        errors = []
        for i in range(0, steps):
            perturbedParameters = initialParameters + stepSize * directionVector
            storedState = self.forward_solve(perturbedParameters)
            perturbedObjective = self.compute_objective_function(storedState, perturbedParameters) 

            fd_value = (perturbedObjective - originalObjective) / stepSize
            fd_values.append(fd_value)

            error = abs(directionalDerivative - fd_value)
            errors.append(error)

            stepSize *= 1e-1

        if printOutput:
            print("\n       grad'*dir          |         FD approx        |         abs error")
            print("--------------------------------------------------------------------------------")
            for i in range(0, steps):
                print(f" {directionalDerivative}   |   {fd_values[i]}   |   {errors[i]}")
        
        return errors
    
