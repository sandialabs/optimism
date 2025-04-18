module use --append /projects/dirrect_numerical_simulation_of_foam_replacement_structures/Modules/modulefiles/
echo "==> Setting up spack"
source /ascldap/users/lkgallu/optimism_sandbox/spack/share/spack/setup-env.sh
echo "==> Activating optimism environment"
spack env activate /ascldap/users/lkgallu/optimism_sandbox/optimism
echo "==> Loading spack packages"
spack load py-pip py-scikit-sparse
echo "==> Loading seacas module for exodus3.py"
module load seacas
echo "==> Running simulation"
METIS_DLL=/ascldap/users/lkgallu/optimism_sandbox/spack/opt/spack/linux-rhel8-skylake_avx512/gcc-8.5.0/metis-5.1.0-2bhqvsoyunilmohqilbqgnwiw4zl5qzl/lib/libmetis.so python NodalCoordinateOptimization.py
