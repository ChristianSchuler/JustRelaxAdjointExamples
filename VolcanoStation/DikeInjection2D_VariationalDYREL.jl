"""
Structured-grid variational-DYREL version of `dike_injection2D_adj_coverage.jl`.

The model is a 40 km × 20 km crustal section containing a finite-width, rotated dike.
The crust is visco-elasto-plastic (Drucker-Prager with a tensile cap); the dike is a
visco-elastic, thermally buoyant magma.  Magma injection is a plane-strain area source
in the continuity equation.  Set `adjoint = true` to evaluate the sensitivity of a
near-surface vertical-velocity observation to density, viscosity and bulk modulus on
the final step.

This is deliberately a structured-grid Stokes example.  A particle marker chain tracks
the free surface and cuts the sticky-air cells out of the variational Stokes system.  The
original FEM model's thermal diffusion and station-network optimisation are not reproduced.
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
using PoissonGrids
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

"""Return horizontal vertices with prescribed minimum and maximum cell widths (in metres)."""
function poisson_horizontal_vertices(nx, lx, min_dx, max_dx; center = 0.0, κ = 0.6)
    0 < min_dx < max_dx || throw(ArgumentError("require 0 < min_dx < max_dx"))
    κ > 0 || throw(ArgumentError("κ must be positive"))
    α = max_dx / min_dx - 1
    half_width = (nx * max_dx - lx) / (2 * α)
    0 < half_width < lx / 2 || throw(ArgumentError(
        "min_dx and max_dx are incompatible with nx=$nx and lx=$lx",
    ))
    center - half_width ≥ -lx / 2 && center + half_width ≤ lx / 2 || throw(ArgumentError(
        "the refined window must lie within the horizontal domain",
    ))
    M = window_monitor(
        α,
        κ,
        half_width / 1e3,
        center / 1e3,
    )
    return solve_grid(-lx / 2e3, lx / 2e3, M, nx) .* 1e3
end

function centers_with_ghosts(xc)
    x = Array(xc)
    return [2 * x[1] - x[2]; x; 2 * x[end] - x[end - 1]]
end

"""Replace a 2-D pressure field with its horizontal mean at every depth."""
function depth_average_pressure!(P)
    P_host = Array(P)
    P_host .= sum(P_host; dims = 1) ./ size(P_host, 1)
    copyto!(P, PTArray(backend)(P_host))
    return nothing
end

if !all(isdefined.(Ref(@__MODULE__), (:set_dike_particle_phases!, :set_dike_source!, :classify_dpcap_failure_mode!)))
    Core.eval(@__MODULE__, quote
        @parallel_indices (i, j) function set_dike_particle_phases!(
                phases, px, py, index, center, across, along, width, dike_length, air_top,
            )
            @inbounds for ip in cellaxes(phases)
                @index(index[ip, i, j]) == 0 && continue
                x = @index px[ip, i, j]
                y = @index py[ip, i, j]
                ξx, ξy = x - center[1], y - center[2]
                across_dike = ξx * across[1] + ξy * across[2]
                along_dike = ξx * along[1] + ξy * along[2]
                in_dike = abs(across_dike) ≤ width / 2 && abs(along_dike) ≤ dike_length / 2
                @index phases[ip, i, j] = y > air_top ? 3.0 : (in_dike ? 2.0 : 1.0)
            end
            return nothing
        end

        @parallel_indices (i, j) function set_dike_source!(
                Q, x, y, center, across, along, width, dike_length, increment
            )
            ξx = x[i] - center[1]
            ξy = y[j] - center[2]
            across_dike = ξx * across[1] + ξy * across[2]
            along_dike = ξx * along[1] + ξy * along[2]
            in_dike = abs(across_dike) ≤ width / 2 && abs(along_dike) ≤ dike_length / 2
            Q[i, j] = in_dike ? increment : 0.0
            return nothing
        end

        @parallel_indices (i, j) function classify_dpcap_failure_mode!(
                failure_mode, λ, τII, P, phase_center, dpcap
            )
            phase = @cell phase_center[i, j]
            if phase[1] > 0 && λ[i, j] > 0
                cap = GeoParams.compute_tensile_cap(
                    dpcap.sinϕ.val, dpcap.cosϕ.val, dpcap.sinΨ.val,
                    dpcap.C.val, dpcap.pT.val,
                )
                # same test as GeoParams' internal (unexported) `ismode2_yield`
                ismode2 = τII[i, j] * (cap.py - cap.pd) ≥ cap.τd * (cap.py - P[i, j])
                failure_mode[i, j] = ismode2 ? Int8(2) : Int8(1)
            else
                failure_mode[i, j] = Int8(0)
            end
            return nothing
        end
    end)
end

"""Return whether `(x, y)` lies inside the rotated dike rectangle."""
@inline function in_dike(x, y, center, across, along, width, dike_length)
    ξx, ξy = x - center[1], y - center[2]
    return abs(ξx * across[1] + ξy * across[2]) ≤ width / 2 &&
           abs(ξx * along[1] + ξy * along[2]) ≤ dike_length / 2
end

function initialise_dike_particle_phases!(
        phases, particles, center, across, along, width, dike_length, air_top,
    )
    return @parallel (@idx size(particles.index)) set_dike_particle_phases!(
        phases, particles.coords..., particles.index, center, across, along, width, dike_length, air_top,
    )
end

function dike_temperature(
        grid, center, across, along, width, dike_length;
        dTdz, T_surface, T_magma, air_top,
    )
    x, y = centers_with_ghosts.(grid.xci)
    T = [
        yj > air_top ? T_surface :
        (in_dike(xi, yj, center, across, along, width, dike_length) ? T_magma : T_surface - dTdz * yj)
        for xi in x, yj in y
    ]
    return PTArray(backend)(T)
end

function dike_rheology(dt)
    crust_viscosity = LinearViscous(; η = 1.0e21)
    magma_viscosity = LinearViscous(; η = 1.0e19)
    crust_elasticity = ConstantElasticity(; G = 3.0e10, Kb = 5.0e10)
    magma_elasticity = ConstantElasticity(; G = 1.0e9, Kb = 1.0e10)
    crust_plasticity = DruckerPragerCap(; C = 1.0e7 / cosd(30.0), ϕ = 30.0, Ψ = 0.0,
        η_vp = 3.0e10 * dt, pT = -5.0e6)

    return (
        SetMaterialParams(;
            Name = "crust",
            Phase = 1,
            Density = PT_Density(; ρ0 = 2700.0, α = 3.0e-5, β = inv(5.0e10), T0 = 273.0),
            Gravity = ConstantGravity(; g = 9.81),
            CompositeRheology = CompositeRheology((crust_viscosity, crust_elasticity, crust_plasticity)),
            Elasticity = crust_elasticity,
            Plasticity = crust_plasticity,
        ),
        SetMaterialParams(;
            Name = "magma dike",
            Phase = 2,
            Density = PT_Density(; ρ0 = 2400.0, α = 5.0e-5, β = inv(1.0e10), T0 = 273.0),
            Gravity = ConstantGravity(; g = 9.81),
            CompositeRheology = CompositeRheology((magma_viscosity, magma_elasticity)),
            Elasticity = magma_elasticity,
        ),
        SetMaterialParams(;
            Name = "sticky air",
            Phase = 3,
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

"""Write a Vy-grid 0/1 box as text, surface row first, with node coordinates in the margins."""
function write_observation_box(filename, box, grid)
    mkpath(dirname(filename))
    x = centers_with_ghosts(grid.xci[1])
    y = Array(grid.xvi[2])
    open(filename, "w") do io
        println(io, "# Vy nodes in the objective box (1) and outside (0); rows: y [m], columns: x [m]")
        println(io, "y\\x ", join(round.(Int, x), ' '))
        for j in reverse(axes(box, 2))
            println(io, round(Int, y[j]), ' ', join(box[:, j], ' '))
        end
    end
    return nothing
end

"""Return DPCap failure codes: 0 = no failure, 1 = tensile cap, 2 = Drucker-Prager shear."""
function dpcap_failure_modes(stokes, phase_ratios, rheology)
    failure_mode = similar(stokes.P, Int8)
    dpcap = rheology[1].Plasticity[1]
    @parallel (@idx size(stokes.P)) classify_dpcap_failure_mode!(
        failure_mode, stokes.λ, stokes.τ.II, stokes.P, phase_ratios.center, dpcap,
    )
    return Array(failure_mode)
end

function save_dike_vtk!(
        vtk_dir, step, time, grid, stokes, ρg, ϕ, chain, phase_ratios, rheology, temperature, dt,
        observation_box;
        stokes_ad = nothing, gradients = nothing,
    )
    mkpath(vtk_dir)
    time_kyr = time / (1.0e3 * year)
    Vx_v = @zeros(grid.ni .+ 1...)
    Vy_v = @zeros(grid.ni .+ 1...)
    velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
    data_v = (; Vx = Array(Vx_v), Vy = Array(Vy_v))
    phase_center = Array(phase_ratios.center)
    data_c = (
        ; pressure = Array(stokes.P),
        material_viscosity = Array(stokes.viscosity.η),
        effective_viscosity = Array(stokes.viscosity.η_vep),
        density = Array(ρg[2]) ./ 9.81,
        temperature = Array(@view temperature[2:(end - 1), 2:(end - 1)]),
        stress_invariant = Array(stokes.τ.II),
        strain_rate_invariant = Array(stokes.ε.II),
        plastic_strain_rate_invariant = Array(stokes.ε_pl.II),
        plastic_failure_mode = dpcap_failure_modes(stokes, phase_ratios, rheology),
        source_rate = Array(stokes.Q) ./ dt,
        crust_fraction = getindex.(phase_center, 1),
        magma_fraction = getindex.(phase_center, 2),
        air_fraction = getindex.(phase_center, 3),
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
        data_c = (
            ; data_c...,
            adjoint_pressure = Array(stokes_ad.λP),
            gradient_ρ0_crust = Array(@view gradients.ρ0.center[1, :, :]),
            gradient_ρ0_magma = Array(@view gradients.ρ0.center[2, :, :]),
            gradient_η_crust = Array(@view gradients.η.center[1, :, :]),
            gradient_η_magma = Array(@view gradients.η.center[2, :, :]),
            gradient_Kb_crust = Array(@view gradients.Kb.center[1, :, :]),
            gradient_Kb_magma = Array(@view gradients.Kb.center[2, :, :]),
        )
    end
    filename = joinpath(vtk_dir, "dike_injection_" * lpad(step, 6, '0'))
    save_vtk(
        filename, grid.xvi, grid.xci, data_v, data_c, (Array(Vx_v), Array(Vy_v));
        t = time_kyr, pvd = joinpath(vtk_dir, "dike_injection"),
    )
    save_marker_chain(
        joinpath(vtk_dir, "marker_chain_" * lpad(step, 6, '0')), chain;
        conversion = 1.0, t = time_kyr, pvd = joinpath(vtk_dir, "marker_chain"),
    )
    return nothing
end

"""
    dike_injection_variational_dyrel(igg; kwargs...)

Run the structured-grid dike-injection model and an adjoint for a near-surface `Vy`
observation on the final time step.  `Q_dike` is the injected plane-strain area rate
in m²/s.  The return value contains the forward solver output, the observation cost,
and phase-wise material gradients.
"""
function dike_injection_variational_dyrel(
        igg;
        nx = 320,
        ny = 176,
        nsteps = 200,
        nout = 10,
        dt = year,
        Q_dike = 1.0e-4,
        strain_rate = 1.0e-30,
        dike_width = 200.0,
        dike_length = 6.0e3,
        dike_center_depth = -5.0e3,
        dip = 90.0,
        air_thickness = 2.0e3,
        observation_rows = 4,
        min_horizontal_cell_width = dike_width / 8,
        max_horizontal_cell_width = 300.0,
        adjoint = true,
        verbose = true,
        vtk_dir = joinpath(@__DIR__, "vtk_dike_injection"),
    )
    igg.nprocs == 1 || throw(ArgumentError("this example currently supports one MPI rank"))
    0 ≤ dip < 180 || throw(ArgumentError("dip must lie in [0, 180) degrees"))
    nx ≥ 2 && ny ≥ 2 || throw(ArgumentError("nx and ny must both exceed one"))
    nsteps ≥ 1 || throw(ArgumentError("nsteps must be positive"))
    nout ≥ 1 || throw(ArgumentError("nout must be positive"))

    lx, depth, air_top = 40.0e3, 20.0e3, 0.0
    air_thickness > 0 || throw(ArgumentError("air_thickness must be positive"))
    dike_center = (0.0, dike_center_depth)
    rotation = deg2rad(90.0 - dip)
    across = (cos(rotation), sin(rotation))
    along = (-sin(rotation), cos(rotation))
    half_extent_y = dike_length / 2 * abs(along[2]) + dike_width / 2 * abs(across[2])
    dike_center_depth + half_extent_y < 0 && dike_center_depth - half_extent_y > -depth ||
        throw(ArgumentError("the dike must lie inside the crust"))

    origin = (-lx / 2, -depth)
    xv = poisson_horizontal_vertices(
        nx, lx, min_horizontal_cell_width, max_horizontal_cell_width;
        center = dike_center[1],
    )
    yv = collect(range(-depth, air_thickness; length = ny + 1))
    grid = Geometry(PTArray(backend), xv, yv)
    ni = grid.ni
    rheology = dike_rheology(dt)
    particles = init_particles(particle_backend, 24, 40, 12, grid.xi_vel...)
    pPhases, = init_cell_arrays(particles, Val(1))
    initialise_dike_particle_phases!(
        pPhases, particles, dike_center, across, along, dike_width, dike_length, air_top,
    )
    phase_ratios = PhaseRatios(particle_backend, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)

    ϕ = RockRatio(backend, ni)
    chain = init_markerchain(particle_backend, 100, 15, 200, grid.xvi[1], air_top)
    compute_rock_fraction!(ϕ, chain, grid.xvi, grid.di.vertex)
    grid_vxi = velocity_grids(grid.xci, grid.xvi, grid.di.center)
    flow_bcs = VelocityBoundaryConditions(; free_slip = (left = true, right = true, top = true, bot = true))
    stokes = StokesArrays(backend, ni, flow_bcs)
    stokes_ad = AdjointStokesArrays(backend, ni, flow_bcs)

    T_surface, T_magma, dTdz = 273.0, 1373.0, 25.0e-3
    temperature = dike_temperature(
        grid, dike_center, across, along, dike_width, dike_length;
        dTdz, T_surface, T_magma, air_top,
    )
    args = (; T = temperature, P = stokes.P, dt)
    ρg = @zeros(ni...), @zeros(ni...)
    air_phase = 3
    compute_ρg!(ρg, phase_ratios, rheology, args; air_phase)
    ρg_rock = ρg[2] .* ϕ.center
    compute_lithostatic_pressure!(stokes.P, ρg_rock, grid.di.vertex[2], igg)
    @. stokes.P = ifelse(ϕ.center > 0, stokes.P / ϕ.center, 0.0)
    depth_average_pressure!(stokes.P)
    stokes
    copyto!(stokes.P0, stokes.P)

    # The original problem prescribes vx = ε̇x at the side walls and vy = 0 at the base.
    stokes.V.Vx .= PTArray(backend)([strain_rate * x for x in Array(grid.xvi[1]), _ in 1:(ny + 2)])
    stokes.V.Vy .= PTArray(backend)([-strain_rate * (y + depth) for _ in 1:(nx + 2), y in Array(grid.xvi[2])])
    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    # shortcut: the dike stays fixed in the Eulerian grid, so the source geometry does not advect.
    # `stokes.Q` stores the volume increment over a step.  Dividing the physical rate by
    # the dike area and multiplying by `dt` makes Q/dt in the pressure residual equal to Q_dike/A.
    source_increment = Q_dike * dt / (dike_width * dike_length)
    @parallel (@idx ni) set_dike_source!(
        stokes.Q, grid.xci..., dike_center, across, along, dike_width, dike_length, source_increment,
    )

    viscosity_cutoff = (1.0e18, 1.0e22)
    compute_viscosity!(stokes, phase_ratios, ϕ, args, rheology, viscosity_cutoff; air_phase)
    dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt; γfact = 20.0, ϵ = 1.0e-5)
    # The objective box lies in the column directly above the dike's top tip and covers the
    # `observation_rows` Vy rows centred between the tip and the free surface, so it stays
    # clear of the air even when the surface moves.
    dy = minimum(Array(grid.di.center[2]))
    dike_top = (dike_center[1] + dike_length / 2 * along[1], dike_center_depth + half_extent_y)
    yv_rows = Array(grid.xvi[2])
    middle = (dike_top[2] + air_top) / 2
    j = argmin(abs.(yv_rows .- (middle - (observation_rows - 1) / 2 * dy)))
    box_bottom, box_top = yv_rows[j], yv_rows[j] + (observation_rows - 1) * dy
    box_bottom ≥ dike_top[2] && box_top < air_top || throw(ArgumentError(
        "the observation box must lie between the dike and the surface",
    ))
    observation = (; field = :Vy, center = (dike_top[1], (box_bottom + box_top) / 2),
        half_width = (dike_width / 2, (box_top - box_bottom) / 2 + 1.0e-6 * dy))
    observation_box = observation_indicator(stokes_ad, grid, observation)
    write_observation_box(joinpath(vtk_dir, "observation_box.txt"), observation_box, grid)
    gradients = adjoint ? material_controls(
        backend, ni, (:ρ0, :η, :Kb); nphases = length(rheology),
    ) : (;)

    output = nothing
    for step in 1:nsteps
        write_output = step % nout == 0 || step == nsteps
        run_adjoint = adjoint && write_output
        if run_adjoint
            for gradient in values(gradients)
                gradient.center .= 0
                gradient.vertex .= 0
            end
        end
        if verbose && igg.me == 0
            println("\nStarting a new variational DYREL solve")
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
            viscosity_relaxation = 1.0e-2,
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
        tensor_invariant!(stokes.ε_pl)
        write_output && save_dike_vtk!(
            vtk_dir, step, step * dt, grid, stokes, ρg, ϕ, chain, phase_ratios, rheology, temperature, dt,
            observation_box,
            stokes_ad = run_adjoint ? stokes_ad : nothing,
            gradients = run_adjoint ? gradients : nothing,
        )

        # The source remains fixed in the Eulerian dike, while the marker chain follows
        # the velocity field and rebuilds the reduced rock domain for the next step.
        step == nsteps && continue
        semilagrangian_advection_markerchain!(
            chain, RungeKutta2(), @velocity(stokes), grid_vxi, grid.xvi, dt,
        )
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
    result = (; output, cost, stokes, stokes_ad, grid, particles, particle_phases = pPhases,
        phase_ratios, rock_fraction = ϕ, chain, ρg, gradients, observation)
    return result
end

# if abspath(PROGRAM_FILE) == @__FILE__
    ImplicitGlobalGrid.grid_is_initialized() && finalize_global_grid(; finalize_MPI = false)
    igg = IGG(init_global_grid(128*1, 128*1, 1; init_MPI = !JustRelax.MPI.Initialized())...)
    @time dike_injection_variational_dyrel(igg)
# end
