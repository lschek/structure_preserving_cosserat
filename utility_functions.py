from ngsolve import *
import sys
import time

def pprint(*args, **kwargs):
    """
    Print arguments and flush stdout.
    """
    print(*args, **kwargs)
    sys.stdout.flush()


def Anti(vec):
    """
    Create antisymmetric matrix from vector.
    """
    return CF( (0, -vec[2], vec[1],
                vec[2], 0, -vec[0],
                -vec[1], vec[0], 0), dims = (3,3) ) 


def vec(m):
    """
    Extract vector from matrix.
    """
    return CF( (m[2,1],m[0,2],m[1,0]), dims = (3,1) ) 


def axl(m):
    """
    Get axial vector from skew part of matrix.
    """
    return vec(Skew(m))
    

def cub(m):
    """
    Get cubic vector from symmetric part of matrix.
    """
    return vec(Sym(m))
    

def Dyad(v1, v2):
    """
    Create dyadic product of v1 and v2.
    """
    return CF( (v1[0]*v2[0], v1[0]*v2[1], v1[0]*v2[2],
                v1[1]*v2[0], v1[1]*v2[1], v1[1]*v2[2],
                v1[2]*v2[0], v1[2]*v2[1], v1[2]*v2[2]), dims = (3,3) ) 


def max_value(a, b):
    """
    Take maximum of a and b.
    """
    return IfPos(a-b, a, b)


def min_value(a, b):
    """
    Take minimum of a and b.
    """
    return IfPos(b-a, a, b)


def invariants_principal(A):
    """Principal invariants of (real-valued) tensor A.
    https://doi.org/10.1007/978-3-7091-0174-2_3

    Copied and modified to work with ngsolve from https://github.com/fenics-dolfiny/dolfiny
    """
    i1 = Trace(A)
    i2 = (Trace(A) ** 2 - Trace(A * A)) / 2
    i3 = Det(A)
    return i1, i2, i3


def eigenstate2(A, autodiff=False):
    """Eigenvalues and eigenprojectors of the 2x2 (real-valued) tensor A.

    Provides the spectral decomposition A = sum_{a=0}^{1} λ_a * E_a
    with (ordered) eigenvalues λ_a and their associated eigenprojectors E_a = n_a^R x n_a^L.

    Note: Tensor A must not have complex eigenvalues!
    Copied and modified to work with ngsolve from https://github.com/fenics-dolfiny/dolfiny
    """
    if A.shape != (2, 2):
        raise RuntimeError(f"Tensor A of shape {A.shape} != (2, 2) is not supported!")
    #
    eps = 3.0e-16  # slightly above 2**-(53 - 1), see https://en.wikipedia.org/wiki/IEEE_754
    eps = 1.0e-10

    #
    A = A.MakeVariable() # not strictly necessary, but safer
    #
    # --- determine eigenvalues λ0, λ1
    #
    I1, _, _ = invariants_principal(A)
    #
    Δ = (A[0, 0] - A[1, 1]) ** 2 + 4 * A[0, 1] * A[1, 0]  # = I1**2 - 4 * I2
    # Avoid dp = 0 and disc = 0, both are known with absolute error of ~eps**2
    # Required to avoid sqrt(0) derivatives and negative square roots
    Δ += eps**2
    # sorted eigenvalues: λ0 <= λ1
    λ = (I1 - sqrt(Δ)) / 2, (I1 + sqrt(Δ)) / 2
    #
    # --- determine eigenprojectors E0, E1
    #
    if autodiff:
        E = [(λk.Diff(A)).trans for λk in λ]
    else:
        λ0, λ1 = λ
        E0 = (A-λ1*Id(2))/(λ0-λ1)
        E1 = (A-λ0*Id(2))/(λ1-λ0)
        E = [E0, E1]
    return λ, E


def eigenstate3_legacy(A, autodiff=False):
    """Eigenvalues and eigenprojectors of the 3x3 (real-valued) tensor A.
    Provides the spectral decomposition A = sum_{a=0}^{2} λ_a * E_a
    with eigenvalues λ_a and their associated eigenprojectors E_a = n_a^R x n_a^L
    ordered by magnitude.
    The eigenprojectors of eigenvalues with multiplicity n are returned as 1/n-fold projector.

    Note: Tensor A must not have complex eigenvalues!

    Copied and modified to work with ngsolve from https://github.com/fenics-dolfiny/dolfiny
    """
    if A.shape != (3, 3):
        raise RuntimeError(f"Tensor A of shape {A.shape} != (3, 3) is not supported!")
    eps = 3.0e-16  # slightly above 2**-(53 - 1), see https://en.wikipedia.org/wiki/IEEE_754
    eps = 1.0e-10

    
    A = A.MakeVariable() # not strictly necessary, but safer

    # --- determine eigenvalues λ0, λ1, λ2
    #
    # additively decompose: A = tr(A) / 3 * I + dev(A) = q * I + B
    q = Trace(A) / 3
    B = A - q * Id(3)
    # observe: det(λI - A) = 0  with shift  λ = q + ω --> det(ωI - B) = 0 = ω**3 - j * ω - b
    j = Trace(B * B) / 2  # == -I2(B) for trace-free B, j < 0 indicates A has complex eigenvalues
    b = Trace(B * B * B) / 3  # == I3(B) for trace-free B
    # solve: 0 = ω**3 - j * ω - b  by substitution  ω = p * cos(phi)
    #        0 = p**3 * cos**3(phi) - j * p * cos(phi) - b  | * 4 / p**3
    #        0 = 4 * cos**3(phi) - 3 * cos(phi) - 4 * b / p**3  | --> p := sqrt(j * 4 / 3)
    #        0 = cos(3 * phi) - 4 * b / p**3
    #        0 = cos(3 * phi) - r                  with  -1 <= r <= +1
    #    phi_k = [acos(r) + (k + 1) * 2 * pi] / 3  for  k = 0, 1, 2
    p = 2 / sqrt(3) * sqrt(j + eps**2)  # eps: MMM
    r = 4 * b / p**3
    r = max_value(min_value(r, +1 - eps), -1 + eps)  # eps: LMM, MMH
    phi = acos(r) / 3
    # sorted eigenvalues: λ0 <= λ1 <= λ2
    λ0 = q + p * cos(phi + 2 / 3 * pi)  # low
    λ1 = q + p * cos(phi + 4 / 3 * pi)  # middle
    λ2 = q + p * cos(phi)  # high

    # --- determine eigenprojectors E0, E1, E2
    if autodiff:
        E0 = (λ0.Diff(A)).trans
        E1 = (λ1.Diff(A)).trans
        E2 = (λ2.Diff(A)).trans
    else:
        E0 = (A-λ1*Id(3))*(A-λ2*Id(3))/((λ0-λ1)*(λ0-λ2))
        E1 = (A-λ2*Id(3))*(A-λ0*Id(3))/((λ1-λ2)*(λ1-λ0))
        E2 = (A-λ0*Id(3))*(A-λ1*Id(3))/((λ2-λ0)*(λ2-λ1))
    return [λ0, λ1, λ2], [E0, E1, E2]


def eigenstate3(A, autodiff=False, symmetric=False):
    """Eigenvalues and eigenprojectors of the 3x3 (real-valued) tensor A.
    Provides the spectral decomposition A = sum_{a=0}^{2} λ_a * E_a
    with (ordered) eigenvalues λ_a and their associated eigenprojectors E_a = n_a^R x n_a^L.

    Note: Tensor A must not have complex eigenvalues!

    Copied and modified to work with ngsolve from https://github.com/fenics-dolfiny/dolfiny
    """
    if A.shape != (3, 3):
        raise RuntimeError(f"Tensor A of shape {A.shape} != (3, 3) is not supported!")
    #eps = 3.0e-16 #16  # slightly above 2**-(53 - 1), see https://en.wikipedia.org/wiki/IEEE_754
    eps = 1.0e-10
    A = A.MakeVariable() # not strictly necessary, but safer

    I1, _, _ = invariants_principal(A)
    
    # New variables for matrix components
    if symmetric:
        # Note: Discriminants could be further simplified here!
        A00 = A[0,0]
        A01 = A[0,1]
        A02 = A[0,2]
        A10 = A01
        A11 = A[1,1]
        A12 = A[1,2]
        A20 = A02
        A21 = A12
        A22 = A[2,2]
    else:
        A00 = A[0,0]
        A01 = A[0,1]
        A02 = A[0,2]
        A10 = A[1,0]
        A11 = A[1,1]
        A12 = A[1,2]
        A20 = A[2,0]
        A21 = A[2,1]
        A22 = A[2,2]

    # Discriminant as sum-of-products
    Δx = [
        A01 * A12 * A20 - A02 * A10 * A21,
        A01 ** 2 * A12
        - A01 * A02 * A11
        + A01 * A02 * A22
        - A02 ** 2 * A21,
        A00 * A01 * A21
        - A01 ** 2 * A20
        - A01 * A21 * A22
        + A02 * A21 ** 2,
        A00 * A02 * A12
        + A01 * A12 ** 2
        - A02 ** 2 * A10
        - A02 * A11 * A12,
        A00 * A01 * A12
        - A01 * A02 * A10
        - A01 * A12 * A22
        + A02 * A12 * A21,
        A00 * A02 * A21
        - A01 * A02 * A20
        + A01 * A12 * A21
        - A02 * A11 * A21,
        A01 * A10 * A12
        - A02 * A10 * A11
        + A02 * A10 * A22
        - A02 * A12 * A20,
        A00 ** 2 * A12
        - A00 * A02 * A10
        - A00 * A11 * A12
        - A00 * A12 * A22
        + A01 * A10 * A12
        + A02 * A10 * A22
        + A11 * A12 * A22
        - A12 ** 2 * A21,
        A00 ** 2 * A12
        - A00 * A02 * A10
        - A00 * A11 * A12
        - A00 * A12 * A22
        + A02 * A10 * A11
        + A02 * A12 * A20
        + A11 * A12 * A22
        - A12 ** 2 * A21,
        A00 * A01 * A11
        - A00 * A01 * A22
        - A01 ** 2 * A10
        + A01 * A02 * A20
        - A01 * A11 * A22
        + A01 * A22 ** 2
        + A02 * A11 * A21
        - A02 * A21 * A22,
        A00 * A01 * A11
        - A00 * A01 * A22
        + A00 * A02 * A21
        - A01 ** 2 * A10
        - A01 * A11 * A22
        + A01 * A12 * A21
        + A01 * A22 ** 2
        - A02 * A21 * A22,
        A00 * A01 * A12
        - A00 * A02 * A11
        + A00 * A02 * A22
        - A01 * A11 * A12
        - A02 ** 2 * A20
        + A02 * A11 ** 2
        - A02 * A11 * A22
        + A02 * A12 * A21,
        A00 * A02 * A11
        - A00 * A02 * A22
        - A01 * A02 * A10
        + A01 * A11 * A12
        - A01 * A12 * A22
        + A02 ** 2 * A20
        - A02 * A11 ** 2
        + A02 * A11 * A22,
        A00 ** 2 * A11
        - A00 ** 2 * A22
        - A00 * A01 * A10
        + A00 * A02 * A20
        - A00 * A11 ** 2
        + A00 * A22 ** 2
        + A01 * A10 * A11
        - A02 * A20 * A22
        + A11 ** 2 * A22
        - A11 * A12 * A21
        - A11 * A22 ** 2
        + A12 * A21 * A22,
    ]
    Δy = [
        A02 * A10 * A21 - A01 * A12 * A20,
        A10 ** 2 * A21
        - A10 * A11 * A20
        + A10 * A20 * A22
        - A12 * A20 ** 2,
        A00 * A10 * A12
        - A02 * A10 ** 2
        - A10 * A12 * A22
        + A12 ** 2 * A20,
        A00 * A20 * A21
        - A01 * A20 ** 2
        + A10 * A21 ** 2
        - A11 * A20 * A21,
        A00 * A10 * A21
        - A01 * A10 * A20
        - A10 * A21 * A22
        + A12 * A20 * A21,
        A00 * A12 * A20
        - A02 * A10 * A20
        + A10 * A12 * A21
        - A11 * A12 * A20,
        A01 * A10 * A21
        - A01 * A11 * A20
        + A01 * A20 * A22
        - A02 * A20 * A21,
        A00 ** 2 * A21
        - A00 * A01 * A20
        - A00 * A11 * A21
        - A00 * A21 * A22
        + A01 * A10 * A21
        + A01 * A20 * A22
        + A11 * A21 * A22
        - A12 * A21 ** 2,
        A00 ** 2 * A21
        - A00 * A01 * A20
        - A00 * A11 * A21
        - A00 * A21 * A22
        + A01 * A11 * A20
        + A02 * A20 * A21
        + A11 * A21 * A22
        - A12 * A21 ** 2,
        A00 * A10 * A11
        - A00 * A10 * A22
        - A01 * A10 ** 2
        + A02 * A10 * A20
        - A10 * A11 * A22
        + A10 * A22 ** 2
        + A11 * A12 * A20
        - A12 * A20 * A22,
        A00 * A10 * A11
        - A00 * A10 * A22
        + A00 * A12 * A20
        - A01 * A10 ** 2
        - A10 * A11 * A22
        + A10 * A12 * A21
        + A10 * A22 ** 2
        - A12 * A20 * A22,
        A00 * A10 * A21
        - A00 * A11 * A20
        + A00 * A20 * A22
        - A02 * A20 ** 2
        - A10 * A11 * A21
        + A11 ** 2 * A20
        - A11 * A20 * A22
        + A12 * A20 * A21,
        A00 * A11 * A20
        - A00 * A20 * A22
        - A01 * A10 * A20
        + A02 * A20 ** 2
        + A10 * A11 * A21
        - A10 * A21 * A22
        - A11 ** 2 * A20
        + A11 * A20 * A22,
        A00 ** 2 * A11
        - A00 ** 2 * A22
        - A00 * A01 * A10
        + A00 * A02 * A20
        - A00 * A11 ** 2
        + A00 * A22 ** 2
        + A01 * A10 * A11
        - A02 * A20 * A22
        + A11 ** 2 * A22
        - A11 * A12 * A21
        - A11 * A22 ** 2
        + A12 * A21 * A22,
    ]
    Δd = [9, 6, 6, 6, 8, 8, 8, 2, 2, 2, 2, 2, 2, 1]
    Δ = sum(Δxk * Δdk * Δyk for Δxk, Δdk, Δyk in zip(Δx, Δd, Δy))  # discriminant as sop
    
    # Invariant dp as sum-of-products
    Δxp = [A10, A20, A21, -A00 + A11, -A00 + A22, -A11 + A22]
    Δyp = [A01, A02, A12, -A00 + A11, -A00 + A22, -A11 + A22]
    Δdp = [6, 6, 6, 1, 1, 1]
    dp = sum(Δxpk * Δdpk * Δypk for Δxpk, Δdpk, Δypk in zip(Δxp, Δdp, Δyp)) / 2  # dp as sop

    # Invariant dq as sum-of-products
    Δxq = [
        A12,
        A21,
        A01 * A10,
        A02 * A20,
        A12 * A21,
        A11 + A22 - 2 * A00,
    ]
    Δyq = [
        A01 * A20,
        A02 * A10,
        A00 + A11 - 2 * A22,
        A00 + A22 - 2 * A11,
        A11 + A22 - 2 * A00,
        (A00 + A22 - 2 * A11) * (A00 + A11 - 2 * A22),
    ]
    Δdq = [27, 27, 9, 9, 9, -1]
    dq = sum(Δxqk * Δdqk * Δyqk for Δxqk, Δdqk, Δyqk in zip(Δxq, Δdq, Δyq))

    # Avoid dp = 0 and disc = 0, both are known with absolute error of ~eps**2
    # Required to avoid sqrt(0) derivatives and negative square roots
    dp += eps**2
    Δ += eps**2

    phi3 = atan2(sqrt(27) * sqrt(Δ) , dq) # atan2(x,y)? UFL documentation is weird
    
    # sorted eigenvalues: λ0 <= λ1 <= λ2
    λ = [(I1 + 2 * sqrt(dp) * cos((phi3 + 2 * pi * k) / 3)) / 3 for k in range(1, 4)]

    #
    # --- determine eigenprojectors E0, E1, E2
    #
    if autodiff:
        E = [(λk.Diff(A)).trans for λk in λ]
        # print ("len(E) = ", len(E))
        # print ("lam0 = ", λ[0].Compile())
        # print ("lam0.diff = ", E[0].Compile())
        
    else:
        λ0, λ1, λ2 = λ
        E0 = (A-λ1*Id(3))*(A-λ2*Id(3))/((λ0-λ1)*(λ0-λ2))
        E1 = (A-λ2*Id(3))*(A-λ0*Id(3))/((λ1-λ2)*(λ1-λ0))
        E2 = (A-λ0*Id(3))*(A-λ1*Id(3))/((λ2-λ0)*(λ2-λ1))
        E = [E0, E1, E2]
    
    return λ, E


def matrix_function(A, fn=lambda A: A, use_legacy=False, autodiff=False):
    """Evaluates A -> fn(A) : R^(m x m) -> R^(m x m) for the given (real-valued) tensor A and fn.
    Uses spectral decomposition and spectral synthesis fn(A) = sum_{a=0}^{m} fn(λ_a) * E_a.

    Parameters
    ----------
    A
        Coefficient function
    fn
        Functor providing the analytic function

        Examples: `fn=exp`, `fn=lambda A: A**2`
        Note: If differentiation through the matrix function is needed, consider
              eps-ification of expressions which are not differentiable at critical points,
              e.g. `fn=lambda A: sqrt(A + eps)`.

    Copied and modified to work with ngsolve from https://github.com/fenics-dolfiny/dolfiny
    """
    if A.shape == (3, 3):
        # instantiate zero matrix
        fn_A = CF(((0,0,0),(0,0,0),(0,0,0)),dims=(3,3))
        if not use_legacy:
            λ, E = eigenstate3(A, autodiff=autodiff)
        else:
            pprint("--- Using legacy formulation for eigenstate ---")
            λ, E = eigenstate3_legacy(A, autodiff=autodiff)
    elif A.shape == (2, 2):
        # instantiate zero matrix
        fn_A = CF(((0,0),(0,0)),dims=(2,2))
        λ, E = eigenstate2(A, autodiff=autodiff)

    # apply UFL function on eigenvalue and synthesise matrix function
    for λ_, E_ in zip(λ, E):
        fn_A += fn(λ_) * E_
    return fn_A


def get_polar_decomposition(P, use_legacy=True, autodiff=False):
    """
    Returns the polar decomposition of P with:
    
    P   = R * U with R: V->SO(3) and U V->Sym++(3)
    U^2 = P^T * P
    U   = sqrt(U^2) (spectral decomposition needed)
    R   = P * U^-1  (spectral decomposition helpful, compiles much faster)
    """
    U_squared = P.trans * P
    eps = 3e-16
    U = matrix_function(U_squared, fn=lambda x: sqrt(x+eps),use_legacy=use_legacy,autodiff=autodiff)
    Uinv = matrix_function(U_squared, fn=lambda x: 1/sqrt(x+eps),use_legacy=use_legacy,autodiff=autodiff)
    R = P * Uinv
    return R, U