/*Species Transport Equation:
∂(Yk)/∂t + ∂(ui * Yk)/∂xi + ∂(V_ic*Y_k)/∂xi = ∂(Dk * Wk/W * ∂Xk/∂xi)/∂xi + ωkdot/ρ
where Xk = Yk.W / Wk :> mole fraction

Breaking Down the Equation:
1. Advection Term:
∂(ui * Yk)/∂xi :> This term represents the rate of change of the Species concentration due to the Flow of the fluid. It is the product of the velocity of the fluid and the concentration of the Species.
Handled by the function: FD_Euler based on the Euler method.

2. Diffusion Term:
∂(Dk * Wk/W * ∂Xk/∂xi)/∂xi :> This term represents the rate of change of the Species concentration due to the diffusion of the Species. It is the product of the diffusion coefficient of the Species, the molecular weight of the Species, and the rate of change of the mole fraction of the Species.
Handled by the function: FD_HC_Euler based on the Hirschfelder-Curtiss method.

3. Reaction Term:
ωkdot/ρ :> This term represents the rate of change of the Species concentration due to the chemical reactions in the system. It is the ratio of the rate of production or consumption of the Species to the density of the fluid.
Handled by the function: FD_HC_Euler.

4. Source Term:
∂(V_k)/∂xi :> This term represents the rate of change of the Species concentration due to the source or sink terms in the system. It is the product of the velocity of the Species and the concentration of the Species.
Handled by the function: FD_Euler.

5. Correction Term:
∂(V_ic.Y_k)/∂xi :> This term represents the rate of change of the Species concentration due to the correction terms in the system. It is the product of the velocity of the Species and the concentration of the Species. 
Handled by the function: FD_Euler_Correction_Velocity.
*/
/*
The Species mass balance equation in conservative form with correction velocity is given by:
    ∂(Yk)/∂t + ∂((ui + Vi_c)Yk) / ∂xi = ∂(Dk.Wk/W.∂Xk/∂xi)/∂xi + ωkdot/ρ
    Source: TNC book page 15. eq: 1.45
where:
    ρ = density : Flow->rho_0
    Yk = mass fraction of Species k : mass_fraction[{X, Y, Z, k}]
    ui = velocity in the i-direction : Flow->velocity[{X, Y, Z, i}]
    Dk = diffusion coefficient of Species k : diffusion_coefficient[{X, Y, Z, k}]
    Wk = molar mass of Species k : Molar_mass[k]
    W is the mean molecular weight : molar_mass_av[{X, Y, Z}]
    Xk = mole fraction of Species k : mass_fraction[{X, Y, Z, k}] * (molar_mass_av[{X, Y, Z}] / Molar_mass[k])
    ωkdot = production rate of Species k : Production[{X, Y, Z, k}]
    Vi_c = correction velocity

    ∂(ρYk)/∂t = conservation of mass of Species k
    ∂(ρ(ui + Vi_c)Yk) / ∂xi = advection of Species k
    ∂(ρDk.Wk/W.∂Xk/∂xi)/∂xi = diffusion of Species k
    ωkdot = production rate of Species k
    Vi_c = Σ_k=1^N_sp(Dk.Wk/W.∂Xk/∂xi)
    The Species mass balance equation is a partial differential equation that describes the evolution of the mass fraction of each Species in time and space.

    This equation solves:
    1. Mass Fraction Evolution: how the mass fraction of each Species evolves in time and space
    2. Advection: ∂(ρYkui)/∂xi : represents the convective transport of Species Yk due to the fluid Flow with velocity ui.
    This accounts for the movement of the Species along the spatial directions.
    3. Diffusion: ∂(ρDk ∂Yk/∂xi)/∂xi : represents the diffusive transport of Species Yk due to the concentration gradient of the Species.
    It accounts for how the Species spreads or diffuses within the fluid.
    4. Production: ωkdot : accounts for any volumetric production or consumption of Species k due to chemical reactions occurring within the fluid.
    This term represents the net effect of chemical reactions on the concentration of Species Yk.
    5. Mean Molecular Weight: W = Σk (Xk/Wk) : represents the mean molecular weight of the mixture.
    It adjusts the diffusion coefficient accordingly to reflect the mass transfer properties of the Species.
    6. Conservation: ∂(ρYk)/∂t : represents the conservation of mass of Species Yk, ensuring that the total mass of Species within the system is conserved as it evolves over time and space.
    7. Correction Velocity: Vi_c = Σ_k=1^N_sp(Dk.Wk/W.∂Xk/∂xi) : represents the correction velocity due to the diffusion of Species k.
    It is evaluated to ensure that the Species mass fraction is updated correctly.
    * Boundary Conditions: The boundary conditions for the Species mass fraction are set at the boundaries of the domain.
    These boundary conditions can be of different types such as Dirichlet, Neumann, or Robin boundary conditions.
    The Species mass fraction is updated using the finite difference method.
    The finite difference method is used to discretize the spatial derivatives in the Species mass balance equation.
*/
/* "FD_Euler" solves (∂(Yk)/∂t = - ∂((ui)Yk)/∂xi))

        From the equation:  ∂(Yk)/∂t = -∂(ui + Vi_c)Yk/∂xi + ∂(Dk.Wk/W.∂Xk/∂xi)/∂xi + ωkdot/ρ
        Modified: (FD_Euler) = -∂(Vi_c.Yk)/∂xi + ∂(Dk.Wk/W.∂Xk/∂xi)/∂xi + ωkdot/ρ

        This is the advection term in the Species mass balance equation.
        The advection term represents the convective transport of Species Yk due to the fluid Flow with velocity ui.
        This accounts for the movement of the Species along the spatial directions.
        ∂(ρYk)/∂t is implicitly solved as part of the overall solution process for the Species transport equation.
        The time derivative term reflects the change in mass fraction of Species Yk over time, and it is typically updated
        along with other terms in an iterative time-stepping scheme.

    1.1: dY_x = FD::WENO3NONCONS(Flow->velocity[{X, Y, Z, 0}], previous_mass_fraction[{X - 2, Y, Z, k}], previous_mass_fraction[{X - 1, Y, Z, k}],
                                previous_mass_fraction[{X, Y, Z, k}], previous_mass_fraction[{X + 1, Y, Z, k}], previous_mass_fraction[{X + 2, Y, Z, k}]);
                WENO3NONCONS(u, f1, f2, f3, f4, f5)
                here f1, f2, f3, f4, f5 is x-2, x-1, x, x+1, x+2 mass fraction of Species k.
                therefore;
                        dY_x = u * (fp - fn)
                        where, fp, fn are the positive and negative fluxes of Species k and dY_x is the advection term in the x-direction.

    1.2: mass_fraction[{X, Y, Z, k}] = Y0 - (dY_x + dY_y + dY_z);
    */
/* "FD_HC_Euler" solves ∂(Dk.Wk/W.∂Xk/∂xi)/∂xi + ωkdot/ρ
    
	where, Dk = diffusion coefficient of Species k
    Wk = molar mass of Species k	: Molar_mass[k]
    W is the mean molecular weight	: molar_mass_av[{X, Y, Z}]
    Xk = mole fraction of Species k : mass_fraction[{X, Y, Z, k}] * (molar_mass_av[{X, Y, Z}] / Molar_mass[k])
    ωkdot = production rate of Species k : Production[{X, Y, Z, k}]
    ρ = density : Flow->rho_0

    From the equation: ∂(Yk)/∂t + ∂((ui + Vi_c)Yk) / ∂xi = ∂(Dk.Wk/W.∂Xk/∂xi)/∂xi + ωkdot/ρ
    Modified: (FD_Euler) + ∂(Vi_c.Yk)/∂xi = (FD_HC_Euler)

    This is the diffusion term in the Species mass balance equation.
    The diffusion term represents the diffusive transport of Species Yk due to the concentration gradient of the Species.
    It accounts for how the Species spreads or diffuses within the fluid.
    ∂(Dk.Wk/W.∂Xk/∂xi)/∂xi is implicitly solved as part of the overall solution process for the Species transport equation.
    here, the diffusion coefficient is a function of the Species mass fraction and the mean molecular weight of the mixture.

    Here Diffusion coefficient D = λ / (ρ * Cp_k) : λ = Thermal conductivity, ρ = density, Cp_k = specific heat capacity of Species k

2.1.1: If both neighboring grid points in the positive and negative x-directions are not solid.
        Vd_right = FD::CENTRAL4FLUX(molar_mass_av[{X - 1, Y, Z}] * previous_mass_fraction[{X - 1, Y, Z, k}], molar_mass_av[{X, Y, Z}] * previous_mass_fraction[{X, Y, Z, k}],
                                molar_mass_av[{X + 1, Y, Z}] * previous_mass_fraction[{X + 1, Y, Z, k}], molar_mass_av[{X + 2, Y, Z}] * previous_mass_fraction[{X + 2, Y, Z, k}],
                                diffusion_coefficient[{X - 1, Y, Z, k}] / molar_mass_av[{X - 1, Y, Z}], diffusion_coefficient[{X, Y, Z, k}] / molar_mass_av[{X, Y, Z}],
                                diffusion_coefficient[{X + 1, Y, Z, k}] / molar_mass_av[{X + 1, Y, Z}], diffusion_coefficient[{X + 2, Y, Z, k}] / molar_mass_av[{X + 2, Y, Z}]);
        CENTRAL4FLUX: f = D1 * (f1 / 8. - f2 / 6. + f3 / 24.) + D2 * (-f1 / 6. - 3. * f2 / 8. + 2. * f3 / 3. - f4 / 8.)
            + D3 * (f1 / 8. - 2. * f2 / 3. + 3. * f3 / 8. + f4 / 6.) + D4 * (-1. * f2 / 24. + f3 / 6. - f4 / 8.);
        where f1, f2, f3, f4 are the values of the function at x-1, x, x+1, x+2 and D1, D2, D3, D4 are the values of the diffusion coefficient at x-1/2, x, x+1/2, x+3/2
        So, Vd_right = W_{X-1,Y,Z} (1/8 * Y_{X-2,Y,Z} - 1/6 * Y_{X-1,Y,Z} + 1/24 * Y_{X,Y,Z}) + W_{X,Y,Z} (-1/6 * Y_{X-2,Y,Z} - 3/8 * Y_{X-1,Y,Z} + 2/3 * Y_{X,Y,Z} - 1/8 * Y_{X+1,Y,Z})
            Vd_left = W_{X-2,Y,Z} (1/8 * Y_{X-3,Y,Z} - 1/6 * Y_{X-2,Y,Z} + 1/24 * Y_{X-1,Y,Z}) + W_{X-1,Y,Z} (-1/6 * Y_{X-3,Y,Z} - 3/8 * Y_{X-2,Y,Z} + 2/3 * Y_{X-1,Y,Z} - 1/8 * Y_{X,Y,Z})
            Flux = W_{X-1,Y,Z} * (1/24 * Y_{X-3,Y,Z} - 1/6 * Y_{X-2,Y,Z} + 1/8 * Y_{X-1,Y,Z}) + W_{X,Y,Z} (-1/6 * Y_{X-3,Y,Z} - 3/8 * Y_{X-2,Y,Z} + 2/3 * Y_{X-1,Y,Z} - 1/8 * Y_{X,Y,Z})
                + W_{X+1,Y,Z} * (-1/6 * Y_{X-1,Y,Z} - 3/8 * Y_{X,Y,Z} + 2/3 * Y_{X+1,Y,Z} - 1/8 * Y_{X+2,Y,Z}) + W_{X+2,Y,Z} * (1/8 * Y_{X,Y,Z} - 1/6 * Y_{X+1,Y,Z} + 1/24 * Y_{X+2,Y,Z})


2.1.2: If one of the neighboring grid points in the positive or negative x-direction is solid.
        Vd_right is calculated using CENTRAL2FLUX: f = 0.5 * (D2 + D1) * (f2 - f1); therefore
        Vd_right = 0.5 * (W_{X,Y,Z} + W_{X+1,Y,Z}) * (Y_{X+1,Y,Z} - Y_{X,Y,Z})
        Vd_left = 0.5 * (W_{X-1,Y,Z} + W_{X,Y,Z}) * (Y_{X,Y,Z} - Y_{X-1,Y,Z})

        Flux[X][Y][Z][k][0] += (Vd_right - Vd_left);so
        Flux = 1/2 * (W_{X,Y,Z} + W_{X+1,Y,Z}) * (Y_{X+1,Y,Z} - Y_{X,Y,Z}) - 1/2 * (W_{X-1,Y,Z} + W_{X,Y,Z}) * (Y_{X,Y,Z} - Y_{X-1,Y,Z})

2.2: Flux[X][Y][Z][kk][0] -= (previous_mass_fraction[{X + 1, Y, Z, kk}] * Vd_right - previous_mass_fraction[{X, Y, Z, kk}] * Vd_left);
    in simple terms, Flux = Vd_right * Y_{X+1,Y,Z} - Vd_left * Y_{X,Y,Z}, this is done because the flux is the product of the velocity and the mass fraction of the Species k.

2.3: mass_fraction[{X, Y, Z, k}] += (Flux[X][Y][Z][k][0] + Flux[X][Y][Z][k][1] + Flux[X][Y][Z][k][2] + Production[{X, Y, Z, k}]) / (Flow->rho_0 * Flow->density[{X, Y, Z}]);
        Here, mass_fraction[{X, Y, Z, k}] is updated using the fluxes and the production term.
        Flow->rho_0 * Flow->density[{X, Y, Z}]) is the total mass of the fluid at the point (X, Y, Z).
        They are divided by the density of the fluid at the point (X, Y, Z) to get the mass fraction because the mass fraction is the mass of the Species divided by the total mass of the fluid at the point (X, Y, Z).

So this function solves the diffusion term or (a second-order spatial derivative of the mass fraction (∇^2.Yk) of the Species mass balance equation.
*/
/*Correction Velocity Term:
The correction velocity term can be expressed as:
∂(V_ic.Y_k)/∂xi
    
    where:
    V_ic : Corrected velocity of the Species.
    given by: V_ic = ∑(k=1 to N) ((Dk/W) * ∂(Yk * W)/∂xi)
    where:
    N : Number of Species
    W : Molecular weight of the Species
    Dk : Diffusion coefficient of the Species
    Yk : Concentration of the Species
    ∂(Yk * W)/∂xi : Rate of change of the product of the concentration of the Species and the molecular weight of the Species

Substituting the value of V_ic in the equation:
∂(V_ic.Y_k)/∂xi = ∂(∑(k=1 to N) ((Dk/W) * ∂(Yk * W)/∂xi) * Y_k)/∂xi

Modified Equation:
∂(Yk)/∂t = ∂(ui * Yk)/∂xi + ∂(∑(k=1 to N) ((Dk/W) * ∂(Yk * W)/∂xi) * Y_k)/∂xi + ∂(Dk * Wk/W * ∂Xk/∂xi)/∂xi + ωkdot/ρ
Where the second term on the right-hand side corresponds to the correction velocity term that is implemented in the FD_Euler_Correction_Velocity function.

*/
/*FD_Euler_diffusion
Overview:
The FD_Euler_diffusion function performs a forward-time centered-space (FTCS) Euler update for the diffusion of a Species within a computational grid.
This function is used to advance the simulation of Species concentration over time by updating the mass fraction at each grid point based on diffusion processes and potential production terms.
It is designed for use in parallel computing environments, as indicated by its interaction with MPI (Message Passing Interface) for distributed processing.

Equations and Methods Used:
1. Diffusion Equation:
The core of the function relies on solving the diffusion equation using finite difference methods. The diffusion equation in three dimensions can be expressed as:
∂ϕ/∂t = D (∂^2 ϕ/∂x^2) + (∂^2 ϕ/∂y^2) + (∂^2 ϕ/z^2) + Production Term

where:
ϕ represents the mass fraction of the Species, 
D is the diffusion coefficient, and the production term accounts for any generation or loss of the Species.

2.Finite Difference Schemes:
Central Difference Method:
The function uses the central difference method to approximate the second spatial derivative of the mass fraction. The central difference formula for a 1D case is given by:
∂^2 ϕ/∂x^2 ≈ (ϕ[i+1] - 2 * ϕ[i] + ϕ[i-1]) / (Δx * Δx)
This method is extended to 2D and 3D using similar principles.

3. Higher-Order Schemes:
For improved accuracy, the function can also use higher-order finite difference schemes like the fourth-order central difference (FD_CENTRAL4) and WENO (Weighted Essentially Non-Oscillatory) methods.
These methods provide a more accurate approximation of the second derivative by incorporating more neighboring points.
    Upwind Schemes:
Depending on the compilation flags, the function may also use upwind schemes (FD_UPWIND, FD_UPWIND2) to handle diffusion in cases where there are significant Flow gradients or discontinuities.

4. Usage:
4.1: Applications:
The FD_Euler_diffusion function can be applied in scenarios such as pollutant dispersion, heat conduction, or Species mixing in computational fluid dynamics (CFD) models.
*/
/*FD_Fick_Euler
This part calculates the diffusion flux using a simple Fick's Law expression.
Fluxes for Species diffusion using Fick's law of diffusion and an Euler explicit scheme.
Updates the fluxes based on the concentration gradients and diffusion coefficients.
Updates the mass fractions based on the fluxes, production terms, and density.
Vd_right represents the flux entering the cell from the right,
and Vd_left represents the flux leaving the cell to the left.

∇X_p = ∑k=1_N {Xp.Xk (Vk−Vp)/D_pk} + ((Yp−Xp)∇P)/P + ((ρ/p)∑k=1_N (Yp*Yk)(fp−fk) for p=1,N (1.39)
where:
∇Xp = the gradient of mole fraction of Species p.
Xp = the mole fraction of Species p.
Dpk = Dkp is the binary mass diffusion coefficient of Species p into Species k.
Xk = mole fraction of Species k: Xk = Y_k.W/W_k.
Vk = the velocity of Species k.Vp vel of Species p.
Yp = the mass fraction of Species p.
∇P = the gradient of pressure.
P = the pressure.
ρ = the density.
fp = the fugacity of Species p. fk = the fugacity of Species k.

∇Xp = (XpXk(Vk−Vp)/Dpk)			(1.40)
given that (Y1+Y2 = 1 and Y1V1 = -D12∇Y1)
VpYp = -Dpk∇Yp (1.41)
V = -Dpk.∇Yp/Yp :
V_right = -Dpk.∇Yp/Yp : diff_coefficient_right * (mass_fraction_right - mass_fraction_left)
V_left = -Dpk.∇Yp/Yp : diff_coefficient_left * (mass_fraction_right - mass_fraction_left)

*/
/* smoothen fields:
The smoothen fields function is used to apply a smoothing operation to the fields of a computational grid.
This operation involves averaging the values of neighboring grid points to reduce noise or fluctuations in the data and improve the overall stability of the simulation.
The function iterates over the grid points and updates the field values based on the average of neighboring points, effectively smoothing out sharp transitions or irregularities in the data.
The last two parameters (Nt and D) control the number of iterations and the strength of the smoothing operation, allowing users to adjust the level of smoothing applied to the fields.

Nt: Number of iterations:  This parameter specifies the number of time steps the function will run the smoothing or relaxation process.
Impact: A higher value of Nt means the function will perform more iterations, leading to more smoothing of the fields. This can help in stabilizing the initial conditions before the main simulation starts.
Stability and Smoothness: If the initial conditions are rough or highly variable, a higher Nt may be needed to achieve a smooth and stable starting state.
Computational Resources: More iterations mean higher computational cost. Balance the need for smoothness with available computational resources.
Typical values might range from a few tens to a few hundred time steps, depending on the specific requirements and characteristics of the simulation.
Example for Nt:
Small Simulation: Nt = 50 might be sufficient for a small, simple simulation with relatively uniform initial conditions.
Large, Complex Simulation: Nt = 200 could be necessary for a larger domain with more complex initial conditions to ensure smooth and stable starting fields.

D:  This parameter represents a diffusion coefficient used for initializing the Thermal and Species solvers.
Impact: The value of D affects the rate at which the Thermal and Species properties diffuse across the simulation domain during the smoothing process. A higher diffusion coefficient results in faster diffusion and potentially more uniform fields.
The smoothen fields function is commonly used in computational fluid dynamics (CFD) simulations to enhance the accuracy and stability of numerical solutions by reducing numerical artifacts and improving convergence properties.
The value should be physically meaningful for the materials and conditions being simulated.
A higher D leads to faster and more extensive smoothing. If the fields need to be quickly homogenized, a higher D might be appropriate.
Example for D:
Thermal Simulation: D could be set to a value based on the Thermal diffusivity of the material, e.g., D = 0.01 m²/s for a typical metal.
Species Transport: D might be set to a value based on the diffusivity of the Species in the medium, e.g., D = 1e-5 m²/s for a gas diffusing in air.
*/
/*FD_Euler_Correction_Velocity
Overview:
The FD_Euler_Correction_Velocity function calculates the correction velocity of a Species based on the diffusion coefficients, molecular weights, and concentration gradients of multiple Species in a computational grid.
This correction velocity term is used to improve the accuracy of Species transport simulations by accounting for the effects of Species diffusion on the overall Flow field.

Equations and Methods Used:
1. Correction Velocity Term:
The correction velocity term V_ic is defined as the sum of the product of the diffusion coefficient Dk/Wk and the gradient of the product of Species concentration Yk and molecular weight Wk for each Species k:
V_ic = ∑(k=1 to N) ((Dk/Wk) * ∂(Yk * Wk)/∂xi)
where:
N is the total number of Species in the system,
Dk is the diffusion coefficient of Species k,
Wk is the molecular weight of Species k,
Yk is the concentration of Species k,
and ∂(Yk * Wk)/∂xi represents the gradient of the product of Species concentration and molecular weight.

2. Finite Difference Approximation:
The function uses finite difference methods to approximate the gradient of the product Yk * Wk with respect to the spatial coordinate xi.
The central difference method is commonly employed to calculate the spatial derivatives, providing a balance between accuracy and computational efficiency.

3. Implementation:
The FD_Euler_Correction_Velocity function iterates over the grid points and Species to calculate the correction velocity term V_ic for each Species based on the diffusion coefficients and concentration gradients.
By incorporating the correction velocity into the Species transport equations, the function enhances the accuracy of the simulation by accounting for the effects of Species diffusion on the Flow field.

4. Usage:
The FD_Euler_Correction_Velocity function is typically used in conjunction with Species transport solvers in computational fluid dynamics (CFD) simulations to improve the fidelity of concentration fields and capture the interactions between Species diffusion and Flow dynamics.
It plays a crucial role in accurately modeling Species transport phenomena in complex systems, such as chemical reactors, combustion chambers, and environmental flows.
*/


/*
The Species mass balance equation in non-conservative form is given by:
∂Yk/∂t + u · ∇Yk + (1/ρ) ∇·(ρVkYk) = ωk/ρ

- `Yk` is the mass fraction of the k-th Species.
- `∂Yk/∂t` represents the rate of change of `Yk` with respect to time.
- `u` is the mixture velocity.
- `∇` denotes the gradient operator.
- `ρ` is the local density.
- `Vk` is the mass flux due to diffusion for the k-th Species.
- `ωk` is the source term due to chemical reactions.
This equation describes how the mass fraction of each Species changes over time due to advection (Flow), diffusion, and chemical reactions in the reacting flows.

The equation for mass flux due to diffusion:
YkVk = - (DkWk / W) ∇Xk + Yk (1/Nsp) Σ (Dk'Wk' / W) ∇Xk'

- `YkVk` is the mass flux for the k-th Species.
- `Dk`, `Wk` are the mole fraction and molar mass of the k-th Species, respectively.
- `W` is the mixture molar mass.
- `∇Xk` is the gradient of the mole fraction of the k-th Species.
- `Nsp` is the total number of Species.
- The term `(1/Nsp) Σ (Dk'Wk' / W) ∇Xk'` represents the sum over all Species of the diffusion term.
- The negative sign on the left side accounts for the direction of diffusion.

*/

/*Production rates
The production rate of a Species in a chemical reaction system refers to the
rate at which that particular Species is being generated or produced over a given period of time.
The production rate of a Species can be defined differently depending on whether the Species is
in a bulk phase (e.g., gas or liquid) or a surface phase (e.g., a catalyst surface).

Bulk Phases (e.g., gas, liquid):
In bulk phases, the production rate of a Species is usually expressed in terms of the amount of that
Species produced per unit volume per unit time. It's often measured in units like kmol/m^3/s, where:

"kmol" stands for kilomoles, which is a unit of amount of substance.
"m^3" represents cubic meters, which is a unit of volume.
"s" denotes seconds, which is a unit of time.

Mathematically, the production rate of a Species in a bulk phase can be expressed as the derivative of
its concentration with respect to time. For example:
Production Rate = [Species]/dt
Where:
d[Species] is the concentration of the Species in the bulk phase. t is time.
*/

/*	The Lewis number is a dimensionless number that relates the Thermal diffusivity of a Species to its mass diffusivity
    It compares the diffusion speeds of heat and mass in a fluid, here mass is the Species k.
    This paraameter is important for laminar flames and is used to determine the relative importance of heat conduction to mass diffusion.
    D_k = diffusion coefficient of Species k
    λ = heat diffusion coefficient aka Thermal conductivity
    c_p = specific heat capacity

    Le_k =  λ / (rho * c_p * D_k) = D_th / (D_k)
        where D_th =  λ / (rho * c_p) = Thermal diffusivity or heat diffusivity coefficient

    for unity Le: D_k = D_th = λ / (rho * c_p)
        λ = Thermal_solver* Thermaldiffusion_coefficient[{X, Y, Z}]
        rho = Flow->density[{X, Y, Z}]
        c_p = Thermal_solver* Thermalc_p[{X, Y, Z}]
        D_k = Species->diffusion_coefficient[{X, Y, Z, k}]
        Le_k = Le[k]

        here we have 3 options for calculating the Species diffusion coefficient:
        1. UnitLewis_Species_diffusion: Le_k = 1
            => D_k = λ / (rho * c_p)
            => Species->diffusion_coefficient[{X, Y, Z, k}] = Thermal_solver* Thermaldiffusion_coefficient[{X, Y, Z}] / (Thermal_solver* Thermalc_p[{X, Y, Z}])
        2. ConstLewis_Species_diffusion: Le_k = const
            => D_k = λ / (rho * c_p * const)
            => Species->diffusion_coefficient[{X, Y, Z, k}] = Thermal_solver* Thermaldiffusion_coefficient[{X, Y, Z}] / (Le[k] * Thermal_solver* Thermalc_p[{X, Y, Z}])
        3. MixtureAveraged_Species_diffusion: Le_k = λ / (rho * c_p * D_k)
            => D_k = λ / (rho * c_p * D_k)
            => Species->diffusion_coefficient[{X, Y, Z, k}] = Thermal_solver* Thermaldiffusion_coefficient[{X, Y, Z}] / (Le[k] * Thermal_solver* Thermalc_p[{X, Y, Z})

The Prandtl number is a dimensionless number that defines the ratio of momentum diffusivity (kinematic viscosity) and Thermal diffusivity.
        It is used to characterize the relative importance of momentum and heat transfer in fluid Flow.
        Pr = ν / α = ν / (rho * c_p)
        where ν = kinematic viscosity
        α = Thermal diffusivity

        here we have 3 options for calculating the Thermal diffusivity:
        1. UnitPrandtl_Thermal_diffusion: Pr = 1
            => α = ν = kinematic_viscosity
            => thermal_diffusivity = Thermal_solver* Thermalkinematic_viscosity[{X, Y, Z}]
        2. ConstPrandtl_Thermal_diffusion: Pr = const
            => α = ν / const
            => thermal_diffusivity = Thermal_solver* Thermalkinematic_viscosity[{X, Y, Z}] / const
        3. MixtureAveraged_Thermal_diffusion: Pr = ν / α
            => α = ν / Pr
            => thermal_diffusivity = Thermal_solver* Thermalkinematic_viscosity[{X, Y, Z}] / Pr

        Pr = 0.7 for air at 20°C
        Pr = 7 for water at 20°C
        Pr = 0.7 for water at 100°C



The Schmidt number is a dimensionless number that defines the ratio of momentum diffusivity (kinematic viscosity) and mass diffusivity
        It is used to characterize the relative importance of momentum and mass transfer in
        fluid Flow, especially in the context of mass transfer from one fluid to another.
        Sc = ν / D_k
        where ν = kinematic viscosity
        D_k = mass diffusivity

        here we have 3 options for calculating the Species diffusion coefficient:
        1. UnitSchmidt_Species_diffusion: Sc = 1
            => D_k = ν = kinematic_viscosity
            => Species->diffusion_coefficient[{X, Y, Z, k}] = Thermal_solver* Thermalkinematic_viscosity[{X, Y, Z}]
        2. ConstSchmidt_Species_diffusion: Sc = const
            => D_k = ν / const
            => Species->diffusion_coefficient[{X, Y, Z, k}] = Thermal_solver* Thermalkinematic_viscosity[{X, Y, Z}] / const
        3. MixtureAveraged_Species_diffusion: Sc = ν / D_k
            => D_k = ν / Sc
            => Species->diffusion_coefficient[{X, Y, Z, k}] = Thermal_solver* Thermalkinematic_viscosity[{X, Y, Z}] / Sc

        Sc = 1 for air at 20°C
        Sc = 0.7 for water at 20°C
        Sc = 0.6 for water at 100°C

The three numbers are related to each other by:
        Sc_k = Pr * Le_k
        Le_k = Sc_k / Pr
        Pr = Sc_k / Le_k
*/

