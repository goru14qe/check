/* Equations: Hybrid Lattice Boltzmann-finite difference model for low mach number combustion simulation Governing equations and numerical model:
1. Classical Lattice Boltzmann (LB) formulation:
∂f_α/∂t + c_α,i ∂f_α/∂x = (1/τ) (f^(eq)_α - f_α) - F_i ∂f/∂ξ // Eq. (5)

2. First-order approximation of the forcing term:
F_i ∂f/∂ξ_i ≈ F_i ∂f^(eq)/∂ξ_i = -ξ_i - u_i / (ρc_s^2) F_i f^(eq) // Eq. (6)

3. Time-evolution equation for g_α:
∂g_α/∂t + c_α,i ∂g_α/∂x_i = (1/τ) (g^(eq)_α - g_α) - w_α c_s^2 (∂ρ/∂t + c_α,i ∂ρ/∂x_i) + w_α (∂p_h/∂t + c_α,i ∂p_h/∂x_i) + (1/ρ) (c_α,i - u_i) F_i f^(eq)_ // Eq. (8)

4. Material derivative of density using the continuity equation:
∂ρ/∂t + c_α,i ∂ρ/∂x_i = (c_α,i - u_i) ∂ρ/∂x_i - ρ ∂u_i/∂x_i // Eq. (9)

5. Forcing term to include hydrodynamic pressure:
F_i = c_s^2 ∂ρ/∂x_i - ∂p_h/∂x_i + F_b,i // Eq. (10)

6. Species transport equation:
∂Y_k/∂t + u_i ∂Y_k/∂x_i + ∂(ρY_k V_k,i)/∂x_i = ω̇_k // Eq. (25)

7. Diffusion velocity (Hirschfelder-Curtiss approximation):
V_k,i = -D_k Y_k / X_k ∂X_k/∂x_i + Y_k V_c,i // Eq. (26)

8. Relation between mass fraction and mole fraction:
Y_k / X_k = M_k / M̄ // Eq. (27) 

9. Diffusion velocity rewritten:
V_k,i = -D_k M_k / M̄ ∂X_k/∂x_i + Y_k V_c,i // Eq. (28)

10. Mass conservation condition:
∑_k Y_k V_k,i = 0 // Eq. (29)

11. Diffusion velocity correction:
V_c,i = ∑_k D_k M_k / M̄ ∂X_k/∂x_i // Eq. (30)

12. Fick approximation for diffusion velocity:
V_k,i = -D_k ∂Y_k/∂x_i // Eq. (31)

13. Navier-Stokes equation (momentum balance):
∂(ρu_i)/∂t + ∂(ρu_i u_j)/∂x_j + ∂p/∂x_i - ∂[μ(∂u_i/∂x_j + ∂u_j/∂x_i)]/∂x_j - ∂[(ζ - 2/3 μ)∂u_j/∂x_j]/∂x_i = 0 // Eq. (1)
*/

/* Equations for the LMNA Solver:
 * 1. Distribution function evolution equation:
 *    ∂f_α / ∂t + c_α,i ∂f_α / ∂x = (1 / τ) (f_eq_α - f_α) - F_i ∂f / ∂ξ
 *    where f_α is the distribution function, c_α,i is the particle velocity,
 *    τ is the relaxation time, f_eq_α is the equilibrium distribution function,
 *    and F_i is the external force term.
 * 
 * 2. Forcing term approximation:
 *    ∂f / ∂ξ_i ≈ F_i ∂f_eq / ∂ξ_i = -(ξ_i - u_i) / (ρc_s^2) F_i f_eq
 *    where ξ_i is the particle velocity, u_i is the macroscopic velocity,
 *    ρ is the density, and c_s is the speed of sound.
 * 
 * 3. Species transport equation:
 *    ∂Y_k / ∂t + u_i ∂Y_k / ∂x_i + ∂(ρY_k V_k,i) / ∂x_i = ω̇_k
 *    where Y_k is the mass fraction of Species k, V_k,i is the diffusion velocity,
 *    and ω̇_k is the production rate of Species k.
 * 
 * 4. Diffusion velocity in the Hirschfelder-Curtiss approximation:
 *    V_k,i = -D_k Y_k / X_k ∂X_k / ∂x_i + Y_k V_c,i
 *    where D_k is the effective diffusion coefficient, X_k is the mole fraction,
 *    and V_c is the diffusion velocity correction.
 * 
 * 5. Mole fraction relation:
 *    Y_k / X_k = M_k / M̄
 *    where M_k is the molar mass of Species k and M̄ is the average molar mass.
 * 
 * 6. Mass conservation condition for diffusion velocity correction:
 *    Σ_k Y_k V_k,i = 0
 * 
 * 7. Diffusion velocity correction:
 *    V_c,i = Σ_k (D_k M_k / M̄) ∂X_k / ∂x_i
 * 
 * 8. Fick approximation for diffusion velocity:
 *    V_k,i = -D_k ∂Y_k / ∂x_i
 * 
 * Description of Divergence of Velocity:
 * 
 * The divergence of velocity is computed to ensure the low Mach number
 * assumption is maintained. The divergence is evaluated as part of the
 * iterative process to update the Flow field. It is used to adjust the
 * pressure and maintain the incompressibility condition, which is
 * crucial for accurate low Mach number simulations.
 */

/* Validation Steps for the Hybrid Solver

- The validation of the hybrid solver is conducted through two main classes of test-cases: pseudo 1-D freely propagating flames and multi-dimensional configurations.
  
  1. Pseudo 1-D Freely Propagating Flame-Front
    - Simple Thermo-Chemistry (Propane/Air 1-D Freely Propagating Flame-Front)
      - A 1-D freely-propagating air/propane premixed flame is simulated in a 2-D domain bounded by an inlet and outlet boundary condition in the x-direction and periodic boundary conditions in the y-direction.
      - Inlet: Imposed mass Flow-rate with fixed temperature and composition, modeled with fixed-velocity boundary conditions and Dirichlet boundary conditions on temperature and Species solvers.
      - Outlet: Emulated open-boundary conditions with constant hydrodynamic pressure and Neumann zero-gradient boundary conditions for Species and temperature fields.
      - Initialization: Imposing fresh and burnt gas composition, temperature, and density on the left and right halves of the domain.
      - Time-step and grid-size are set to ensure stability and accuracy based on Fourier number, Courant–Friedrichs–Lewy (CFL) condition, and chemical reaction term stiffness.
    - Detailed Thermo-Chemistry (Methane/Air 1-D Freely Propagating Flame)
      - A 1-D freely propagating Methane/Air flame modeled using the BFER 2-step chemistry model.
      - Initial conditions with a domain divided into fresh and burnt gas sections.
      - Simulation of flame-front propagation speeds using five different grid-sizes and corresponding time-steps, demonstrating second-order convergence behavior in space.

  2. Multi-Dimensional Configurations
    - 2-D Configuration I: Premixed Propane/Air Counter-Flow Flame
      - A 2-D premixed counter-Flow burner setup with fresh gas mixture at inlets and constant hydrodynamic pressure at outlets.
      - Simulation initialized with fresh and burnt gas in respective regions, considering only the upper right quadrant due to symmetrical configuration.
      - Comparison of results between the proposed hybrid scheme and the commercial CFD solver ANSYS-FLUENT.

    - 2-D Configuration II: Co-Current Jet Methane/Air Diffusion Flame
      - Not detailed in the provided text, but likely involves similar boundary conditions and initialization procedures as the 2-D counter-Flow flame.
      
    - 3-D Configuration: Premixed Propane/Air Counter-Flow Flame
      - Extension of the 2-D counter-Flow configuration into a 3-D domain.
      - Boundary and initial conditions similar to the 2-D case, adapted for the third dimension.

- Each validation step involves comparisons with reference simulations (e.g., REGATH) to measure errors and ensure accuracy and stability of the proposed hybrid solver.
- The validation steps aim to showcase the solver's performance in handling different thermo-chemical scenarios and geometrical configurations, ensuring robustness and reliability.
*/

/* Summary of the Paper:
 * 
 * Title: Hybrid Lattice Boltzmann-finite difference model for low mach number combustion simulation
 * 
 * Abstract:
 * The paper presents a consistent Lattice Boltzmann (LB) model tailored for
 * simulating combustion processes at low Mach numbers. The proposed model
 * addresses the challenges associated with compressibility effects in LB methods
 * when applied to combustion. The model incorporates a novel approach to 
 * account for pressure and temperature variations, ensuring accurate representation
 * of the thermodynamic properties and chemical kinetics involved in combustion.
 * 
 * Key Contributions:
 * 1. Development of an LB formulation that accurately models low Mach number flows
 *    with combustion, maintaining consistency with the compressible Navier-Stokes
 *    equations.
 * 
 * 2. Introduction of a new equilibrium distribution function that includes the
 *    effects of pressure and temperature variations, enhancing the model's ability
 *    to simulate realistic combustion scenarios.
 * 
 * 3. Implementation of a forcing term that captures the influence of external
 *    forces and chemical reactions on the Flow field, improving the accuracy of
 *    the simulations.
 * 
 * 4. Validation of the model through a series of benchmark tests and comparisons
 *    with experimental data, demonstrating its capability to handle a wide range
 *    of combustion conditions.
 * 
 * 5. Application of the model to practical combustion problems, showcasing its
 *    potential for use in engineering and research applications.
 * 
 * Methodology:
 * The paper details the derivation of the LB model, starting from the Boltzmann
 * equation and introducing the necessary modifications to handle low Mach number
 * flows with combustion. The authors describe the numerical implementation,
 * including the discretization of the velocity space and the integration of
 * chemical kinetics. The model's performance is assessed through various test
 * cases, including laminar and turbulent flames, premixed and non-premixed
 * combustion, and flame-wall interactions.
 * 

 */