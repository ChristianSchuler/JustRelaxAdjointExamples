using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

const isCUDA = false

@static if isCUDA
    using CUDA
    CUDA.allowscalar(false)
end

using JustRelax, JustRelax.JustRelax2D
using JustPIC
using GeoParams
using PoissonGrids
using CairoMakie
using ParallelStencil, ParallelStencil.FiniteDifferences2D

CairoMakie.activate!()

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

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

if !isdefined(@__MODULE__, :set_dora_maira_phases!)
    # `@parallel_indices` creates auxiliary constants while it expands. Deferring its expansion
    # avoids redefining those constants when this editable script is included again in one REPL.
    Core.eval(@__MODULE__, quote
        @parallel_indices (i, j) function set_dora_maira_phases!(
                phases, px, py, index, weak_half_width, x0, y0, a, b
            )
            @inbounds for ip in cellaxes(phases)
                @index(index[ip, i, j]) == 0 && continue
                x = @index(px[ip, i, j])
                y = @index(py[ip, i, j])
                phase = abs(x) ≤ weak_half_width ? 2.0 : 1.0
                ((x - x0) / a)^2 + ((y - y0) / b)^2 ≤ 1 && (phase = 3.0)
                @index phases[ip, i, j] = phase
            end
            return nothing
        end
    end)
end

if !isdefined(@__MODULE__, :uhp_observation_weights!)
    Core.eval(@__MODULE__, quote
        @parallel_indices (i, j) function uhp_observation_weights!(
                weights, phase_ratios_Vy, uhp_phase
            )
            # Vy has one ghost column on each horizontal side.
            @inbounds weights[i + 1, j] = @index phase_ratios_Vy[uhp_phase, i, j]
            return nothing
        end
    end)
end

"""
    initialize_dora_maira_phases!(phases, particles, weak_half_width, ellipse_center, ellipse_axes)

Set mantle phase 1 everywhere, a vertical low-viscosity mantle zone (phase 2) at the domain
centre, and a buoyant UHP ellipse (phase 3) inside that zone.
"""
function initialize_dora_maira_phases!(
        phases, particles, weak_half_width, ellipse_center, ellipse_axes
    )
    # `particles.index` defines the valid particle-cell layout, including boundary padding.
    ni = size(particles.index)
    x0, y0 = ellipse_center
    a, b = ellipse_axes
    return @parallel (@idx ni) set_dora_maira_phases!(
        phases, particles.coords..., particles.index, weak_half_width, x0, y0, a, b,
    )
end

function dora_maira_gravity()
    g = 9.81
    # JustRelax's momentum residual contains `-ρg`, so positive components drive material
    # toward decreasing x and y: from the top right to the bottom left.
    return DippingGravity(; g, gx = g / sqrt(2), gy = 0.0, gz = g / sqrt(2))
end

function dora_maira_rheology(; gravity = dora_maira_gravity())
    # Schmalholz & Schenker (2016), their eq. (1): the reported flow laws are
    # axial-compression laws, so GeoParams must apply its tensor correction.
    # The reference model uses f = 1/50 outside the layer, f = 1/1000 in the
    # layer, and f = 1 for the ellipse. Since η ∝ A^(-1/n), a viscosity factor
    # f is represented by A / f^n and remains valid as strain rate evolves.
    matrix_factor = 1.0 / 50.0
    weak_layer_factor = 1.0e-3
    peridotite = DislocationCreep(;
        A = 2.0e-21 / matrix_factor^4,
        n = 4.0,
        E = 471.0e3,
        V = 0.0,
        r = 0.0,
        R = 8.3145,
        Apparatus = AxialCompression,
    )
    weak_peridotite = DislocationCreep(;
        A = 2.0e-21 / weak_layer_factor^4,
        n = 4.0,
        E = 471.0e3,
        V = 0.0,
        r = 0.0,
        R = 8.3145,
        Apparatus = AxialCompression,
    )
    diabase = DislocationCreep(;
        A = 7.9621e-25,
        n = 3.4,
        E = 260.0e3,
        V = 0.0,
        r = 0.0,
        R = 8.3145,
        Apparatus = AxialCompression,
    )
    return (
        SetMaterialParams(;
            Name = "mantle",
            Phase = 1,
            Density = ConstantDensity(; ρ = 3300.0),
            CompositeRheology = CompositeRheology((peridotite,)),
            Gravity = gravity,
        ),
        SetMaterialParams(;
            Name = "weak mantle shear zone",
            Phase = 2,
            Density = ConstantDensity(; ρ = 3300.0),
            CompositeRheology = CompositeRheology((weak_peridotite,)),
            Gravity = gravity,
        ),
        SetMaterialParams(;
            Name = "buoyant UHP crust",
            Phase = 3,
            Density = ConstantDensity(; ρ = 2900.0),
            CompositeRheology = CompositeRheology((diabase,)),
            Gravity = gravity,
        ),
    )
end

function dora_maira_temperature(grid)
    # Paper initial condition: 500 °C at the top, increasing linearly to 770 °C at 110 km.
    # Build the small initial field on the host, then transfer it to the selected backend.
    temperature = zeros(Float64, (grid.ni .+ 2)...)
    x, y = Array.(grid.xci)
    dTdy = (770.0 - 500.0) / 110e3
    @views begin
        temperature[2:(end - 1), 2:(end - 1)] .= [
            273.15 + 500.0 - dTdy * yj for _ in x, yj in y
        ]
        temperature[2:(end - 1), 1] .= 2 .* temperature[2:(end - 1), 2] .-
                                        temperature[2:(end - 1), 3]
        temperature[2:(end - 1), end] .= 2 .* temperature[2:(end - 1), end - 1] .-
                                          temperature[2:(end - 1), end - 2]
        temperature[1, :] .= temperature[2, :]
        temperature[end, :] .= temperature[end - 1, :]
    end
    return PTArray(backend)(temperature)
end

function initialize_dora_maira_pressure!(P, grid, gravity; ρref = 3300.0)
    # ∇P = -ρref g balances the uniform mantle in JustRelax's `-∇P - ρg` residual.
    gx, _, gz = compute_gravity(gravity)
    x, y = Array.(grid.xci)
    @assert size(P) == (length(x), length(y))
    copyto!(P, [ρref * (gx * (x[end] - xi) + gz * (y[end] - yj)) for xi in x, yj in y])
    return P
end

"""
    uhp_observation(phase_ratios, uhp_phase, Vy)

Build an adjoint objective for the mean vertical velocity of the current UHP body. The weights
are the UHP phase fraction at the `Vy` nodes, so the observation follows the particles as the
initial ellipse translates and deforms.
"""
function uhp_observation(phase_ratios, uhp_phase, Vy)
    weights = similar(Vy)
    fill!(weights, 0)
    ratios = phase_ratios.Vy
    @parallel (@idx size(ratios)) uhp_observation_weights!(weights, ratios, uhp_phase)
    total_weight = sum(weights)
    iszero(total_weight) && error("the advected UHP phase has no observation weight")
    weights ./= total_weight
    return (; field = :Vy, weights)
end

function poisson_horizontal_vertices(nx, lx, weak_zone_width)
    x_window = (-weak_zone_width / 2, weak_zone_width / 2) ./ 1e3
    α_ref = 5.0
    κ_ref = 0.25
    M = window_monitor(
        α_ref / 2, κ_ref * 2, (x_window[2] - x_window[1]) / 2, sum(x_window) / 2,
    )
    return solve_grid(-lx / 2e3, lx / 2e3, M, nx) .* 1e3
end

function plot_grid_spacing!(figdir, grid, weak_half_width)
    mkpath(figdir)
    x, y = Array.(grid.xci) ./ 1e3
    xv, yv = Array.(grid.xvi) ./ 1e3
    Δx, Δy = Array.(grid.di.vertex) ./ 1e3
    fig = Figure(size = (900, 300))
    xaxis = Axis(
        fig[1, 1]; xlabel = "x [km]", ylabel = "Δx [km]",
        title = "Horizontal grid spacing",
    )
    yaxis = Axis(
        fig[1, 2]; xlabel = "y [km]", ylabel = "Δy [km]",
        title = "Vertical grid spacing",
    )
    lines!(xaxis, x, Δx; color = :black, linewidth = 2)
    vlines!(xaxis, [-weak_half_width, weak_half_width] ./ 1e3; color = :crimson, linestyle = :dash)
    lines!(yaxis, y, Δy; color = :black, linewidth = 2)
    xlims!(xaxis, extrema(xv))
    xlims!(yaxis, extrema(yv))
    save(joinpath(figdir, "grid_spacing.png"), fig)
    return nothing
end

"""
    dora_maira_vertical_weak_zone(igg; kwargs...)

Run a 2-D buoyancy-driven UHP exhumation model in a vertical central low-viscosity mantle shear
zone. At the requested cadence, the adjoint computes the sensitivity of the current UHP body's
mean vertical velocity to each phase's density, dislocation-creep prefactor `A`, and exponent `n`.
`adjoint = false` skips those calculations.
"""
function dora_maira_vertical_weak_zone(
        igg;
        nx = 256,
        ny = 250,
        lx = 55e3,
        nsteps = 100,
        figdir = "DoraMairaVerticalWeakZone",
        adjoint = true,
        verbose_PH = true,
        adjoint_every = 10,
    )
    # Schmalholz & Schenker (2016) reference geometry: 125 km × 110 km domain,
    # 5 km-thick weak layer, and a 20 km × 3 km UHP ellipse. The default 55 km width is
    # a compact exploratory domain; use `lx = 125e3` for the paper-domain width.
    igg.nprocs == 1 || throw(ArgumentError(
        "the Dora Maira grid currently supports one MPI rank"
    ))
    ly = 110e3
    weak_zone_width = 5e3
    lx > weak_zone_width || throw(ArgumentError("lx must exceed the 5 km weak-zone width"))
    adjoint_every ≥ 1 || throw(ArgumentError("adjoint_every must be at least one step"))
    xv = poisson_horizontal_vertices(nx, lx, weak_zone_width)
    yv = collect(range(-ly, 0.0; length = ny + 1))
    grid = Geometry(PTArray(backend), xv, yv)
    ni = grid.ni
    min_cell_widths = minimum.(grid.di.vertex)
    # GeoParams maps (gx, gz) to this 2-D model; gravity points top-right to bottom-left.
    gravity = dora_maira_gravity()
    rheology = dora_maira_rheology(; gravity)
    weak_half_width = weak_zone_width / 2
    ellipse_center, ellipse_axes = (0.0, -98e3), (1.5e3, 10e3)
    plot_grid_spacing!(figdir, grid, weak_half_width)

    particles = init_particles(particle_backend, 30, 40, 12, grid.xi_vel...)
    pPhases, = init_cell_arrays(particles, Val(1))
    initialize_dora_maira_phases!(pPhases, particles, weak_half_width, ellipse_center, ellipse_axes)
    phase_ratios = PhaseRatios(particle_backend, length(rheology), ni)
    update_phase_ratios!(phase_ratios, particles, pPhases)
    ϕ = RockRatio(backend, ni)
    update_rock_ratio!(ϕ, phase_ratios, 0)

    # Buoyancy drives the flow; all boundaries are free-slip.
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
        no_slip = (left = false, right = false, top = false, bot = false),
        periodic = (left = false, right = false, top = false, bot = false),
    )
    stokes = StokesArrays(backend, ni, flow_bcs)
    stokes_ad = AdjointStokesArrays(backend, ni, flow_bcs)
    ρg = @zeros(ni...), @zeros(ni...)
    # shortcut: the reference model evolves heat; this compact example retains its initial geotherm.
    temperature = dora_maira_temperature(grid)
    dt = 1.0e11
    args = (; T = temperature, P = stokes.P, dt)
    viscosity_cutoff = 1e16, 1e22
    compute_ρg!(ρg, phase_ratios, rheology, args)
    initialize_dora_maira_pressure!(stokes.P, grid, gravity)
    stokes.P0 .= stokes.P
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff)

    flow_bcs!(stokes, flow_bcs)
    update_halo!(@velocity(stokes)...)

    dyrel = DYREL(backend, stokes, rheology, phase_ratios, ϕ, grid.di, dt, γfact = 20.0; ϵ = 1e-4)
    observation = nothing
    gradients = material_controls(backend, ni, (:ρ, :A, :n); nphases = length(rheology))
    output = nothing

    for step in 1:nsteps
        run_adjoint = adjoint && (step % adjoint_every == 0 || step == nsteps)
        # Rebuild the objective from the current phase ratios after particle advection.
        run_adjoint && (observation = uhp_observation(phase_ratios, 3, stokes.V.Vy))
        # Continuation reaches the paper's 10¹⁶ Pa s lower bound from a well-conditioned
        # initial Stokes solve.  Only the final stage is used for the physical time step.
        cutoff_stages = step == 1 ? (
            (1e18, viscosity_cutoff[2]), (1e17, viscosity_cutoff[2]), viscosity_cutoff,
        ) : (viscosity_cutoff,)
        for (stage, active_cutoff) in pairs(cutoff_stages)
            final_stage = stage == lastindex(cutoff_stages)
            args = (; T = temperature, P = stokes.P, dt)
            output = solve_VariationalDYREL!(
                stokes,
                run_adjoint && final_stage ? stokes_ad : nothing,
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
                verbose_PH,
                verbose_DR = false,
                iterMax = 50_000,
                total_iterMax = 200_000,
                nout = 10,
                rel_drop = 1e-1,
                viscosity_relaxation = 1.0e-2,
                viscosity_cutoff = active_cutoff,
                linear_viscosity = false,
                air_phase = 0,
                adjoint = run_adjoint && final_stage,
                observation,
                gradients = run_adjoint && final_stage ? gradients : (;),
            )
            final_stage || copyto!(stokes.P0, stokes.P)
        end

        step == nsteps && break

        dt = compute_dt(stokes, min_cell_widths, igg)
        advection_MQS!(particles, RungeKutta4(), @velocity(stokes), dt)
        move_particles!(particles, (pPhases,))
        inject_particles_phase!(particles, pPhases, (), ())
        update_phase_ratios!(phase_ratios, particles, pPhases)
        update_rock_ratio!(ϕ, phase_ratios, 0)
    end

    return (; output, stokes, phase_ratios, gradients)
end

if !@isdefined(NO_AUTORUN)
    let nx = 128, ny = 128
        ImplicitGlobalGrid.grid_is_initialized() && finalize_global_grid(; finalize_MPI = false)
        igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
        Base.invokelatest(dora_maira_vertical_weak_zone, igg; nx, ny)
    end
end
