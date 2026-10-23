"""
Structured-grid variational-DYREL model of the Eifel mantle plume beneath a 2000 km
south–north section, set up to test how a glaciation to the north affects the Eifel region.

x runs from south (x = 0) to north (x = `lx`). The column is a 30 km crust, an 80 km mantle
lithosphere (base at 110 km), and upper mantle down to 660 km. Near the southern end, a buoyant,
low-viscosity circular plume body of 100 km diameter sits beneath the lithosphere; its width
follows the low-velocity anomaly imaged beneath the Eifel (Ritter et al., 2001, EPSL 186), which
corresponds to an excess temperature of ~100–150 K. All rheologies are linear viscous. An optional Vialov-profile ice
sheet with its dome at the northern boundary loads the surface; `ice_thickness = 0` gives the
unglaciated reference. A marker chain tracks the free surface (on top of the ice where present)
and cuts the sticky-air cells out of the variational Stokes system.

Set `adjoint = true` to evaluate the sensitivity of the vertical velocity in a box beneath the
Eifel to the density and viscosity of every phase.
"""

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

const isCUDA = false

@static if isCUDA
    using CUDA
    CUDA.allowscalar(false)
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
using JustPIC
using GeoParams
using ParallelStencil, ParallelStencil.FiniteDifferences2D

const backend = @static if isCUDA
    JustRelax.CUDABackend
else
    JustRelax.CPUBackend
end

const particle_backend = @static if isCUDA
    CUDA.CUDABackend
else
    JustPIC.CPU
end

const year = 365.25 * 24 * 3600.0

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

const phase_names = (:crust, :lithosphere, :mantle, :plume, :ice, :air)
const air_phase = 6

"""
Vialov ice-sheet thickness at `x` for a dome of height `H` at `x_dome` and the margin at
`x_margin`; zero outside the sheet.
"""
@inline function ice_height(x, x_margin, x_dome, H)
    s = abs(x - x_dome) / abs(x_dome - x_margin)
    return s < 1 ? H * (1 - s^(4 / 3))^(3 / 8) : zero(x)
end

if !isdefined(@__MODULE__, :set_eifel_particle_phases!)
    # Deferring the macro expansion avoids redefining its auxiliary constants when this
    # script is included again in one REPL.
    Core.eval(@__MODULE__, quote
        @parallel_indices (i, j) function set_eifel_particle_phases!(
                phases, px, py, index, crust_base, lithosphere_base, plume_center, plume_radius,
                x_margin, x_dome, ice_thickness,
            )
            @inbounds for ip in cellaxes(phases)
                @index(index[ip, i, j]) == 0 && continue
                x = @index px[ip, i, j]
                y = @index py[ip, i, j]
                phase = if y > ice_height(x, x_margin, x_dome, ice_thickness)
                    6.0 # air
                elseif y > 0
                    5.0 # ice
                elseif y > crust_base
                    1.0
                elseif y > lithosphere_base
                    2.0
                elseif (x - plume_center[1])^2 + (y - plume_center[2])^2 ≤ plume_radius^2
                    4.0
                else
                    3.0
                end
                @index phases[ip, i, j] = phase
            end
            return nothing
        end
    end)
end

function initialise_eifel_particle_phases!(phases, particles, geometry)
    (; crust_base, lithosphere_base, plume_center, plume_radius, x_margin, x_dome, ice_thickness) = geometry
    return @parallel (@idx size(particles.index)) set_eifel_particle_phases!(
        phases, particles.coords..., particles.index, crust_base, lithosphere_base,
        plume_center, plume_radius, x_margin, x_dome, ice_thickness,
    )
end

function eifel_rheology(;
        ρ_mantle = 3300.0,
        plume_density_contrast = 15.0,
        η_crust = 1.0e23,
        η_lithosphere = 1.0e23,
        η_mantle = 5.0e20,
        η_plume = 1.0e19,
        η_ice = 1.0e20,
    )
    gravity = ConstantGravity(; g = 9.81)
    material(phase, name, ρ, η) = SetMaterialParams(;
        Name = name,
        Phase = phase,
        Density = ConstantDensity(; ρ),
        Gravity = gravity,
        CompositeRheology = CompositeRheology((LinearViscous(; η),)),
    )
    return (
        material(1, "crust", 2800.0, η_crust),
        material(2, "mantle lithosphere", ρ_mantle, η_lithosphere),
        material(3, "upper mantle", ρ_mantle, η_mantle),
        # a ~100–150 K excess temperature gives Δρ = ρ α ΔT ≈ 10–15 kg/m³ (α = 3e-5 1/K)
        material(4, "Eifel plume", ρ_mantle - plume_density_contrast, η_plume),
        # shortcut: ice is far stiffer than in nature so it acts as a load and does not spread
        material(5, "ice sheet", 917.0, η_ice),
        SetMaterialParams(;
            Name = "sticky air",
            Phase = 6,
            Density = ConstantDensity(; ρ = 0.0),
            Gravity = ConstantGravity(; g = 0.0),
            CompositeRheology = CompositeRheology((LinearViscous(; η = 1.0e18),)),
        ),
    )
end

"""Return a 0/1 array on the observed field's grid, 1 on the nodes the objective sums."""
function observation_indicator(stokes_ad, grid, observation)
    mask = JustRelax2D.observation_mask(stokes_ad, grid, observation)
    box = zeros(Int, size(mask.target))
    box[mask.i, mask.j] .= 1
    return box
end

function save_eifel_vtk!(
        vtk_dir, step, time, grid, stokes, ρg, ϕ, chain, phase_ratios, observation_box;
        stokes_ad = nothing, gradients = nothing,
    )
    mkpath(vtk_dir)
    time_kyr = time / (1.0e3 * year)
    Vx_v = @zeros(grid.ni .+ 1...)
    Vy_v = @zeros(grid.ni .+ 1...)
    velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
    data_v = (; Vx = Array(Vx_v), Vy = Array(Vy_v))
    phase_center = Array(phase_ratios.center)
    fractions = NamedTuple(
        Symbol(name, "_fraction") => getindex.(phase_center, p) for (p, name) in enumerate(phase_names)
    )
    data_c = (
        ; pressure = Array(stokes.P),
        viscosity = Array(stokes.viscosity.η),
        density = Array(ρg[2]) ./ 9.81,
        strain_rate_invariant = Array(stokes.ε.II),
        stress_invariant = Array(stokes.τ.II),
        fractions...,
        rock_fraction = Array(ϕ.center),
        # a cell is in the box when the Vy nodes on both its bottom and top faces are
        observation_box = observation_box[2:(end - 1), 1:(end - 1)] .* observation_box[2:(end - 1), 2:end],
    )
    if !isnothing(stokes_ad)
        adjoint_Vx_v = @zeros(grid.ni .+ 1...)
        adjoint_Vy_v = @zeros(grid.ni .+ 1...)
        # stokes_ad.V and stokes_ad.P hold the adjoint residuals; λV and λP are the adjoint fields
        velocity2vertex!(adjoint_Vx_v, adjoint_Vy_v, stokes_ad.λV.Vx, stokes_ad.λV.Vy)
        data_v = (; data_v..., adjoint_Vx = Array(adjoint_Vx_v), adjoint_Vy = Array(adjoint_Vy_v))
        # per unit area: the per-cell gradients scale with the cell area; air has none
        gradient_fields = NamedTuple(
            Symbol("gradient_", parameter, "_", phase_names[p], "_per_area") => Array(@view gradient.per_area[p, :, :])
                for (parameter, gradient) in pairs(gradients) for p in 1:(air_phase - 1)
        )
        data_c = (; data_c..., adjoint_pressure = Array(stokes_ad.λP), gradient_fields...)
    end
    filename = joinpath(vtk_dir, "eifel_" * lpad(step, 6, '0'))
    save_vtk(
        filename, grid.xvi, grid.xci, data_v, data_c, (Array(Vx_v), Array(Vy_v));
        t = time_kyr, pvd = joinpath(vtk_dir, "eifel"),
    )
    save_marker_chain(
        joinpath(vtk_dir, "marker_chain_" * lpad(step, 6, '0')), chain;
        conversion = 1.0, t = time_kyr, pvd = joinpath(vtk_dir, "marker_chain"),
    )
    return nothing
end

"""
    eifel_plume_glaciation(igg; kwargs...)

Run the Eifel plume model with an optional northern ice sheet of dome thickness
`ice_thickness` (m) extending from `ice_margin_x` to the northern boundary. Every `nout` steps
and on the final step, the state (and, with `adjoint = true`, the adjoint fields and
per-unit-area material gradients) is written to VTK in `vtk_dir`. The objective is the sum of
`Vy` over a box of half-width `observation_half_width` centred above the plume at
`observation_depth`. The return value contains the forward solver output, the observation
cost, and the phase-wise material gradients.
"""
function eifel_plume_glaciation(
        igg;
        nx = 200,
        ny = 68,
        nsteps = 20,
        nout = 10,
        dt = 1.0e3 * year,
        lx = 2000.0e3,
        depth = 660.0e3,
        air_thickness = 20.0e3,
        crust_thickness = 30.0e3,
        lithosphere_thickness = 80.0e3,
        plume_x = 300.0e3,
        plume_radius = 50.0e3,
        plume_center_depth = 200.0e3,
        ice_thickness = 0.0,
        ice_margin_x = 550.0e3,
        observation_depth = -10.0e3,
        observation_half_width = 50.0e3,
        rheology_kwargs = (;),
        adjoint = false,
        verbose = true,
        vtk_dir = joinpath(@__DIR__, "vtk_eifel"),
    )
    igg.nprocs == 1 || throw(ArgumentError("this example currently supports one MPI rank"))
    nx ≥ 2 && ny ≥ 2 || throw(ArgumentError("nx and ny must both exceed one"))
    nsteps ≥ 1 || throw(ArgumentError("nsteps must be positive"))
    nout ≥ 1 || throw(ArgumentError("nout must be positive"))
    0 ≤ ice_thickness < air_thickness || throw(ArgumentError("the ice must fit below the top of the air layer"))
    0 < plume_x - plume_radius && plume_x + plume_radius < ice_margin_x < lx ||
        throw(ArgumentError("the plume must lie inside the domain, south of the ice margin"))
    crust_thickness + lithosphere_thickness ≤ plume_center_depth - plume_radius &&
        plume_center_depth + plume_radius < depth ||
        throw(ArgumentError("the plume must lie between the base of the lithosphere and the model bottom"))

    air_top = 0.0
    geometry = (;
        crust_base = -crust_thickness,
        lithosphere_base = -(crust_thickness + lithosphere_thickness),
        plume_center = (plume_x, -plume_center_depth), plume_radius,
        x_margin = ice_margin_x, x_dome = lx, ice_thickness,
    )
    origin = (0.0, -depth)
    xv = collect(range(0.0, lx; length = nx + 1))
    yv = collect(range(-depth, air_thickness; length = ny + 1))
    grid = Geometry(PTArray(backend), xv, yv)
    ni = grid.ni
    rheology = eifel_rheology(; rheology_kwargs...)
    particles = init_particles(particle_backend, 24, 40, 12, grid.xi_vel...)
    pPhases, = init_cell_arrays(particles, Val(1))
    initialise_eifel_particle_phases!(pPhases, particles, geometry)
    phase_ratios = PhaseRatios(particle_backend, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    # The free surface starts on top of the ice sheet and at sea level elsewhere.
    surface = [air_top + ice_height(x, ice_margin_x, lx, ice_thickness) for x in Array(grid.xvi[1])]
    chain = init_markerchain(particle_backend, 100, 15, 200, grid.xvi[1], surface)
    ϕ = RockRatio(backend, ni)
    compute_rock_fraction!(ϕ, chain, grid.xvi, grid.di.vertex)
    flow_bcs = VelocityBoundaryConditions(; free_slip = (left = true, right = true, top = true, bot = true))
    stokes = StokesArrays(backend, ni, flow_bcs)
    stokes_ad = AdjointStokesArrays(backend, ni, flow_bcs)

    # shortcut: densities are constant, so temperature only fills the rheology arguments
    temperature = @fill(273.0, (ni .+ 2)...)
    args = (; T = temperature, P = stokes.P, dt)
    ρg = @zeros(ni...), @zeros(ni...)
    compute_ρg!(ρg, phase_ratios, rheology, args; air_phase)
    ρg_rock = ρg[2] .* ϕ.center
    compute_lithostatic_pressure!(stokes.P, ρg_rock, grid.di.vertex[2], igg)
    @. stokes.P = ifelse(ϕ.center > 0, stokes.P / ϕ.center, 0.0)
    copyto!(stokes.P0, stokes.P)
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    viscosity_cutoff = (1.0e18, 1.0e24)
    compute_viscosity!(stokes, phase_ratios, ϕ, args, rheology, viscosity_cutoff; air_phase)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; γfact = 20.0, ϵ = 1.0e-5)
    # One Vy row at `observation_depth`, which lies on a grid row for the defaults.
    dy = minimum(Array(grid.di.center[2]))
    observation = (; field = :Vy, center = (plume_x, observation_depth),
        half_width = (observation_half_width, 0.49 * dy))
    observation_box = observation_indicator(stokes_ad, grid, observation)
    gradients = adjoint ? material_controls(
        backend, ni, (:ρ, :η); nphases = length(rheology),
    ) : (;)

    output = nothing
    for step in 1:nsteps
        write_output = step % nout == 0 || step == nsteps
        run_adjoint = adjoint && write_output
        if verbose && igg.me == 0
            println("\nStarting a new variational DYREL solve, step $step")
        end
        output = solve_VariationalDYREL!(
            stokes,
            run_adjoint ? stokes_ad : nothing,
            ρg,
            dyrel,
            flow_bcs,
            phase_ratios,
            ϕ,
            rheology,
            args,
            grid,
            dt,
            igg;
            verbose_PH = verbose,
            verbose_DR = false,
            iterMax = 50_000,
            total_iterMax = 100_000,
            nout = 100,
            rel_drop = 1.0e-2,
            viscosity_cutoff,
            linear_viscosity = true,
            air_phase,
            free_surface = true,
            adjoint = run_adjoint,
            observation,
            gradients = run_adjoint ? gradients : (;),
        )
        tensor_invariant!(stokes.τ)
        tensor_invariant!(stokes.ε)
        write_output && save_eifel_vtk!(
            vtk_dir, step, step * dt, grid, stokes, ρg, ϕ, chain, phase_ratios, observation_box;
            stokes_ad = run_adjoint ? stokes_ad : nothing,
            gradients = run_adjoint ? gradients : nothing,
        )

        # The marker chain follows the flow and rebuilds the reduced rock domain.
        step == nsteps && continue
        semilagrangian_advection_markerchain!(
            chain, RungeKutta2(), @velocity(stokes), grid.xi_vel, grid.xvi, dt,
        )
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), dt)
        move_particles!(particles, (pPhases,))
        update_phases_given_markerchain!(
            pPhases, chain, particles, origin, grid.di.center, air_phase,
        )
        inject_particles_phase!(particles, pPhases, (), ())
        update_phase_ratios!(phase_ratios, particles, pPhases)
        compute_rock_fraction!(ϕ, chain, grid.xvi, grid.di.vertex)
    end

    cost = if adjoint
        mask = JustRelax2D.observation_mask(stokes_ad, grid, observation)
        sum(@view stokes.V.Vy[mask.i, mask.j])
    else
        nothing
    end
    return (; output, cost, stokes, stokes_ad, grid, particles, particle_phases = pPhases,
        phase_ratios, rock_fraction = ϕ, chain, ρg, gradients, observation)
end

if !@isdefined(NO_AUTORUN)
    let nx = 200, ny = 68
        ImplicitGlobalGrid.grid_is_initialized() && finalize_global_grid(; finalize_MPI = false)
        igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
        # unglaciated reference; pass e.g. `ice_thickness = 2.0e3` for the glaciated case
        Base.invokelatest(eifel_plume_glaciation, igg; nx, ny)
    end
end
