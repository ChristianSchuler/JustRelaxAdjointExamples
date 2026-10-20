using Pkg
Pkg.activate(joinpath(@__DIR__, "Project.toml"))

const isCUDA = true

@static if isCUDA
    using CUDA
    CUDA.allowscalar(false)
end

using JustRelax, JustRelax.JustRelax2D
using JustPIC
using GeoParams
using CairoMakie
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

function center_refined_vertices(nx, lx; weak_width = 5e3, weak_spacing = 100.0)
    nweak = round(Int, weak_width / weak_spacing)
    nside_total = nx - nweak
    nside_total ≥ 0 && iseven(nside_total) || throw(ArgumentError(
        "nx must exceed the weak-zone cell count by an even number"
    ))
    nside = nside_total ÷ 2
    side_length = (lx - weak_width) / 2
    side_length ≥ nside * weak_spacing || throw(ArgumentError(
        "nx is too large for the requested weak-zone spacing"
    ))

    # Geometrically increase the cell widths away from the uniformly resolved weak zone.
    side_sum(r) = isone(r) ? nside : (r^nside - 1) / (r - 1)
    target = side_length / weak_spacing
    lo, hi = 1.0, 2.0
    while side_sum(hi) < target
        hi *= 2
    end
    for _ in 1:80
        r = (lo + hi) / 2
        side_sum(r) < target ? (lo = r) : (hi = r)
    end
    outward_widths = weak_spacing .* ((lo + hi) / 2) .^ (0:(nside - 1))
    @assert isapprox(sum(outward_widths), side_length; rtol = 1e-12)

    left = -weak_width / 2 .- reverse(cumsum(outward_widths))
    weak_zone = collect(range(-weak_width / 2, weak_width / 2; length = nweak + 1))
    right = weak_width / 2 .+ cumsum(outward_widths)
    vertices = vcat(left, weak_zone, right)
    vertices[1], vertices[end] = -lx / 2, lx / 2
    return vertices
end

function uhp_refined_vertical_vertices(
        ny, ly; refined_top = -80e3, fine_spacing = 250.0, coarse_spacing = 1e3,
        growth = 1.05,
    )
    -ly < refined_top < 0 || throw(ArgumentError("refined_top must lie inside the domain"))
    fine_height = refined_top + ly
    nfine = round(Int, fine_height / fine_spacing)
    isapprox(nfine * fine_spacing, fine_height; rtol = 1e-12) || throw(ArgumentError(
        "the fine vertical band must be an integer number of fine cells"
    ))
    ntransition = ceil(Int, log(coarse_spacing / fine_spacing) / log(growth))
    transition_widths = fine_spacing .* growth .^ (0:(ntransition - 1))
    remaining_height = -refined_top - sum(transition_widths)
    ncoarse = round(Int, remaining_height / coarse_spacing)
    ncoarse > 0 || throw(ArgumentError("vertical domain is too short for the requested refinement"))
    coarse_spacing_actual = remaining_height / ncoarse
    coarse_spacing_actual ≤ growth * last(transition_widths) || throw(ArgumentError(
        "ny is incompatible with the requested smooth vertical refinement"
    ))

    widths = vcat(
        fill(fine_spacing, nfine), transition_widths, fill(coarse_spacing_actual, ncoarse),
    )
    length(widths) == ny || throw(ArgumentError(
        "this vertical refinement has $(length(widths)) cells; set ny = $(length(widths))"
    ))
    vertices = -ly .+ vcat(0.0, cumsum(widths))
    vertices[end] = 0.0
    return vertices
end

function plot_grid_refinement!(figdir, xv, yv, weak_half_width; refined_grid = true)
    take(figdir)
    x, y = (xv[1:(end - 1)] .+ xv[2:end]) ./ 2e3, (yv[1:(end - 1)] .+ yv[2:end]) ./ 2e3
    Δx, Δy = diff(xv) ./ 1e3, diff(yv) ./ 1e3
    fig = Figure(size = (900, 300))
    xaxis = Axis(
        fig[1, 1]; xlabel = "x [km]", ylabel = "Δx [km]",
        title = refined_grid ? "Horizontal grid refinement" : "Uniform horizontal grid",
    )
    yaxis = Axis(
        fig[1, 2]; xlabel = "y [km]", ylabel = "Δy [km]",
        title = refined_grid ? "Vertical grid refinement" : "Uniform vertical grid",
    )
    lines!(xaxis, x, Δx; color = :black, linewidth = 2)
    vlines!(xaxis, [-weak_half_width, weak_half_width] ./ 1e3; color = :crimson, linestyle = :dash)
    lines!(yaxis, y, Δy; color = :black, linewidth = 2)
    refined_grid && vlines!(yaxis, [-110.0, -80.0]; color = :crimson, linestyle = :dash)
    save(joinpath(figdir, "grid_refinement.png"), fig)
    return nothing
end

function plot_initial_grid!(figdir, xv, yv, weak_half_width, ellipse_center, ellipse_axes)
    take(figdir)
    x, y = xv ./ 1e3, yv ./ 1e3
    θ = range(0, 2π; length = 129)
    ellipse_x = ellipse_center[1] / 1e3 .+ ellipse_axes[1] / 1e3 .* cos.(θ)
    ellipse_y = ellipse_center[2] / 1e3 .+ ellipse_axes[2] / 1e3 .* sin.(θ)
    fig = Figure(size = (900, 750))
    axis = Axis(
        fig[1, 1]; aspect = DataAspect(), xlabel = "x [km]", ylabel = "y [km]",
        title = "Initial computational grid",
    )
    poly!(
        axis,
        Point2f[
            (-weak_half_width / 1e3, minimum(y)), (weak_half_width / 1e3, minimum(y)),
            (weak_half_width / 1e3, maximum(y)), (-weak_half_width / 1e3, maximum(y)),
        ]; color = (:orange, 0.18), strokewidth = 0,
    )
    vlines!(axis, x; color = (:black, 0.35), linewidth = 0.6)
    hlines!(axis, y; color = (:black, 0.35), linewidth = 0.6)
    vlines!(axis, [-weak_half_width, weak_half_width] ./ 1e3; color = :crimson, linewidth = 2)
    lines!(axis, ellipse_x, ellipse_y; color = :dodgerblue, linewidth = 3)
    text!(axis, 0.0, -8.0; text = "weak zone", align = (:center, :top), color = :crimson, fontsize = 18)
    save(joinpath(figdir, "initial_grid.png"), fig)
    return nothing
end

function gravity_key!(axis, x, y, gravity)
    gx, _, gz = compute_gravity(gravity)
    # GeoParams stores components used in `-ρg`; the key shows physical acceleration.
    gx, gz = -gx, -gz
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    xspan, yspan = xmax - xmin, ymax - ymin
    # A compact, opaque legend keeps the vector and label readable over any phase colour.
    # DataAspect() preserves the physical direction of the arrow.
    magnitude = hypot(gx, gz)
    iszero(magnitude) && return nothing
    left, right = xmax - 0.49 * xspan, xmax - 0.04 * xspan
    bottom, top = ymax - 0.15 * yspan, ymax - 0.035 * yspan
    midy = (bottom + top) / 2
    arrow_length = 0.12 * min(xspan, yspan)
    dx, dy = arrow_length .* (gx, gz) ./ magnitude
    midx = left + 0.10 * xspan

    poly!(
        axis,
        Point2f[(left, bottom), (right, bottom), (right, top), (left, top)];
        color = :white, strokecolor = :gray60, strokewidth = 0.7,
    )
    arrows2d!(
        axis, [midx - dx / 2], [midy - dy / 2], [dx], [dy];
        color = :black, shaftwidth = 2, tipwidth = 9, tiplength = 10,
    )
    labelx = left + 0.20 * xspan
    text!(
        axis, labelx, midy + 0.018 * yspan; text = "Gravity",
        align = (:left, :center), color = :black, fontsize = 15, font = :bold,
    )
    text!(
        axis, labelx, midy - 0.018 * yspan;
        text = "$(round(magnitude; digits = 2)) m s⁻²",
        align = (:left, :center), color = :gray25, fontsize = 12,
    )
    return nothing
end

function plot_dora_maira!(
        figdir, step, grid, phase_ratios, stokes, gravity,
    )
    take(figdir)
    Vx, Vy = @velocity(stokes)
    Vx_v, Vy_v = @zeros(size(stokes.P) .+ 1...), @zeros(size(stokes.P) .+ 1...)
    velocity2vertex!(Vx_v, Vy_v, Vx, Vy)

    xci, xvi = Array.(grid.xci) ./ 1e3, Array.(grid.xvi) ./ 1e3
    # Transfer the backing storage first; Array(::CellArray) otherwise scalar-indexes it.
    uhp_fraction = getindex.(JustPIC.CPUCellArray(phase_ratios.center), 3)
    velocity_magnitude = hypot.(Array(Vx_v), Array(Vy_v))
    fig = Figure(size = (1500, 1050))
    panels = (
        ("UHP phase fraction", xci, uhp_fraction, :batlow, nothing),
        ("log₁₀ effective viscosity [Pa s]", xci, log10.(Array(stokes.viscosity.η)), :batlow, nothing),
        ("velocity magnitude [m s⁻¹]", xvi, velocity_magnitude, :viridis, nothing),
    )
    for (panel, (title, coordinates, field, colormap, colorrange)) in enumerate(panels)
        row = fld(panel - 1, 2) + 1
        column = 2 * (mod(panel - 1, 2) + 1) - 1
        axis = Axis(
            fig[row, column];
            aspect = DataAspect(), title, xlabel = "x [km]", ylabel = column == 1 ? "y [km]" : "",
        )
        image = isnothing(colorrange) ?
                heatmap!(axis, coordinates..., field; colormap) :
                heatmap!(axis, coordinates..., field; colormap, colorrange)
        contour!(
            axis, xci..., uhp_fraction;
            levels = [0.5], color = :white, linewidth = 2,
        )
        if panel == 1
            gravity_key!(axis, coordinates..., gravity)
        end
        Colorbar(fig[row, column + 1], image)
    end
    save(joinpath(figdir, "dora_maira_$(step).png"), fig; px_per_unit = 2)
    save(joinpath(figdir, "dora_maira_$(step).pdf"), fig)
    return nothing
end

@inline parameter_value(x) = hasproperty(x, :val) ? x.val : x

function dora_maira_parameter_values(rheology)
    return (
        ρ = ntuple(
            p -> parameter_value(rheology[p].Density[1].ρ), length(rheology)
        ),
        A = ntuple(
            p -> parameter_value(rheology[p].CompositeRheology[1].elements[1].A),
            length(rheology),
        ),
        n = ntuple(
            p -> parameter_value(rheology[p].CompositeRheology[1].elements[1].n),
            length(rheology),
        ),
    )
end

function symmetric_colorrange(field)
    finite_values = filter(isfinite, vec(field))
    maximum_abs = isempty(finite_values) ? 0.0 : maximum(abs, finite_values)
    return iszero(maximum_abs) ? (-1.0, 1.0) : (-maximum_abs, maximum_abs)
end

function plot_dora_maira_sensitivities!(
        figdir, step, grid, phase_ratios, gradients, rheology,
    )
    take(figdir)
    xci = Array.(grid.xci) ./ 1e3
    uhp_fraction = getindex.(JustPIC.CPUCellArray(phase_ratios.center), 3)
    parameter_values = dora_maira_parameter_values(rheology)
    phase_names = ("mantle", "weak zone", "UHP crust")
    parameter_labels = (ρ = "ρ", A = "A", n = "n")

    fig = Figure(size = (1650, 1450))
    Label(
        fig[0, 1:6],
        "Relative sensitivity of mean UHP vertical velocity,  p ∂J/∂p";
        fontsize = 24, font = :bold,
    )
    for (row, phase) in enumerate(eachindex(rheology))
        for (column_index, parameter) in enumerate((:ρ, :A, :n))
            field = Array(@view gradients[parameter].center[phase, :, :])
            field .*= parameter_values[parameter][phase]
            column = 2column_index - 1
            axis = Axis(
                fig[row, column]; aspect = DataAspect(),
                title = "$(parameter_labels[parameter]) ∂J/∂$(parameter_labels[parameter])  —  $(phase_names[phase])",
                xlabel = row == length(rheology) ? "x [km]" : "",
                ylabel = column_index == 1 ? "y [km]" : "",
            )
            image = heatmap!(
                axis, xci..., field; colormap = :vik,
                colorrange = symmetric_colorrange(field),
            )
            contour!(
                axis, xci..., uhp_fraction;
                levels = [0.5], color = :black, linewidth = 1.5,
            )
            Colorbar(fig[row, column + 1], image; width = 14)
        end
    end
    save(joinpath(figdir, "dora_maira_sensitivities_$(step).png"), fig; px_per_unit = 2)
    save(joinpath(figdir, "dora_maira_sensitivities_$(step).pdf"), fig)
    return nothing
end

"""
    dora_maira_vertical_weak_zone(igg; kwargs...)

Run a 2-D buoyancy-driven UHP exhumation model in a vertical central low-viscosity mantle shear
zone. At every plotted step, the adjoint computes the sensitivity of the current UHP body's
mean vertical velocity to each phase's density, dislocation-creep prefactor `A`, and exponent
`n`. `adjoint = false` skips those calculations.
"""
function dora_maira_vertical_weak_zone(
        igg;
        nx = 256,
        ny = 512,
        lx = 55e3,
        nsteps = 100,
        figdir = "DoraMairaVerticalWeakZone",
        plot_results = true,
        adjoint = true,
        verbose_PH = true,
        refined_grid = false,
        plot_every = 10,
    )
    # Schmalholz & Schenker (2016) reference geometry: 125 km × 110 km domain,
    # 5 km-thick weak layer, and a 20 km × 3 km UHP ellipse. The default 55 km width is
    # a compact exploratory domain; use `lx = 125e3` for the paper-domain width.
    igg.nprocs == 1 || throw(ArgumentError(
        "the refined Dora Maira grid currently supports one MPI rank"
    ))
    ly = 110e3
    weak_zone_width = 5e3
    weak_zone_spacing = 100.0
    lx > weak_zone_width || throw(ArgumentError("lx must exceed the 5 km weak-zone width"))
    plot_every ≥ 1 || throw(ArgumentError("plot_every must be at least one step"))
    # The production grid resolves the 5 km channel at 100 m, grades symmetrically outward,
    # and vertically refines the UHP ellipse's depth range. Set `refined_grid = false` for
    # a uniform grid using the requested nx and ny.
    xv = refined_grid ?
         center_refined_vertices(nx, lx; weak_width = weak_zone_width, weak_spacing = weak_zone_spacing) :
         collect(range(-lx / 2, lx / 2; length = nx + 1))
    yv = refined_grid ?
         uhp_refined_vertical_vertices(ny, ly) :
         collect(range(-ly, 0.0; length = ny + 1))
    grid = Geometry(PTArray(backend), xv, yv)
    ni = grid.ni
    min_cell_widths = minimum.(grid.di.vertex)
    # GeoParams maps (gx, gz) to this 2-D model; gravity points top-right to bottom-left.
    gravity = dora_maira_gravity()
    rheology = dora_maira_rheology(; gravity)
    weak_half_width = weak_zone_width / 2
    ellipse_center, ellipse_axes = (0.0, -98e3), (1.5e3, 10e3)
    if plot_results
        plot_grid_refinement!(figdir, xv, yv, weak_half_width; refined_grid)
        plot_initial_grid!(figdir, xv, yv, weak_half_width, ellipse_center, ellipse_axes)
    end

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
    # viscosity_cutoff = 1e18, 1e22
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
        plot_step = plot_results && (step % plot_every == 0 || step == nsteps)
        run_adjoint = adjoint && plot_step
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
                # A 10% inner reduction is sufficient before each nonlinear PH update;
                # forcing 1% here exhausts the inner budget on the 100 m weak-zone grid.
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

        if plot_step
            plot_dora_maira!(figdir, step, grid, phase_ratios, stokes, gravity)
            run_adjoint && plot_dora_maira_sensitivities!(
                figdir, step, grid, phase_ratios, gradients, rheology,
            )
        end
        step == nsteps && break

        # dt = 0.4 * compute_dt(stokes, min_cell_widths, igg)
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
    let nx = 256, ny = 512
        ImplicitGlobalGrid.grid_is_initialized() && finalize_global_grid(; finalize_MPI = false)
        igg = IGG(init_global_grid(nx, ny, 1; init_MPI = !JustRelax.MPI.Initialized())...)
        Base.invokelatest(dora_maira_vertical_weak_zone, igg; nx, ny)
    end
end
