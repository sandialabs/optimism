# Units:
# Force = N
# Length = mm
# Time = s

#Elastic Properties
E = 69.0e3
nu = 0.3

#Hardening Properties
Y0 = 281.329651
Ysat = 390.507476654
eps0 = 0.0737221022
#H = E/10

# Phase Field Parameters
Gc = 15.
L = 1.0
psiC = 3.*Gc/(16*L)

# void growth Parameters
C0 = 0.3333
C1 = 1.0
f0 = 0.0

props = {'elastic modulus': E,
         'poisson ratio': nu,
         'yield strength': Y0,
         'hardening model': 'voce',
         #'hardening modulus': H,
         'saturation strength': Ysat,
         'reference plastic strain': eps0,
         'critical energy release rate': Gc,
         'critical strain energy density': psiC,
         'regularization length': L,
         'void growth prefactor': C0,
         'void growth exponent': C1,
         'initial void fraction': f0}
