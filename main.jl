
using LinearAlgebra, Luxor, Colors

dt = 0.0002
CRIT_COMPRESS = 1-1.9e-2
CRIT_STRETCH = 1+7.5e-3
HARDENING = 5.0
YOUNGS_MODULUS = 1.5e5
POISSONS_RATIO = 0.2
LAMBDA = YOUNGS_MODULUS*POISSONS_RATIO/((1+POISSONS_RATIO)*(1-2*POISSONS_RATIO))
MU = YOUNGS_MODULUS/(2+2*POISSONS_RATIO)
BSPLINE_EPSILON = 1e-4
BSPLINE_RADIUS = 2
PARTICLE_DIAM = .0072
DENSITY = 100.0
GRAVITY = -9.8


function bspline(x::Float64)::Float64
    x = abs(x)
    w = 0.0
    if x < 1.0
        w = x*x*(x/2.0 - 1.0) + 2.0/3.0;
    elseif x < 2.0
        w = x*(x*(-x/6.0 + 1.0) - 2.0) + 4.0/3.0;
    else
        return 0.0
    end
    if w < BSPLINE_EPSILON
        return 0.0
    end
    return w
end

function bsplineSlope(x::Float64)::Float64
    absX = abs(x)
    if absX < 1.0
        return 1.5*x*absX - 2.0*x;
    elseif x < 2.0
        return -x*absX/2.0 + 2.0*x - 2.0*x/absX;
    else
        return 0.0
    end
end

mutable struct Particle
    volume::Float64
    mass::Float64
    position::Vector{Float64}
    velocity::Vector{Float64}
    def_elastic::Matrix{Float64}
    def_plastic::Matrix{Float64}
    weight_gradientsX::Matrix{Float64}
    weight_gradientsY::Matrix{Float64}
    weights::Matrix{Float64}
    velocity_gradient::Matrix{Float64}
end

function Particle(pos::Vector{Float64}, vel::Vector{Float64}, mass::Float64)::Particle
    volume = 0.0
    def_elastic = [1.0 0.0; 0.0 1.0]
    def_plastic = [1.0 0.0; 0.0 1.0]
    weight_gradientsX = zeros(Float64, 4, 4)
    weight_gradientsY = zeros(Float64, 4, 4)
    weights = zeros(Float64, 4, 4)
    velocity_gradient = zeros(Float64, 2, 2)
    return Particle(volume, mass, pos, vel, def_elastic, def_plastic, weight_gradientsX, weight_gradientsY, weights, velocity_gradient)
end



function energyDerivative(self::Particle)::Matrix{Float64}
    svdResult = svd(copy(self.def_elastic))
    w::Matrix{Float64} = copy(svdResult.U)
    v::Matrix{Float64} = copy(svdResult.V)
    e::Matrix{Float64} = diagm(copy(svdResult.S))
    harden::Float64 = exp(HARDENING * (1.0 - det(self.def_plastic)))
    Je::Float64 = e[1, 1] * e[2, 2]
    temp::Matrix{Float64} = (2.0 * MU) .* (self.def_elastic - w * v') * self.def_elastic'
    temp = temp + diagm([LAMBDA * Je * (Je - 1.0), LAMBDA * Je * (Je - 1.0)])
    return (self.volume * harden) .* temp
end


mutable struct GridNode
    mass::Float64
    active::Bool
    velocity::Vector{Float64}
    velocity_new::Vector{Float64}
end

mutable struct Grid
    cellsize::Vector{Float64}
    node_area::Float64
    nodes::Matrix{GridNode}
    cellcount::Int64
end

function Grid(dims::Vector{Float64}, cells::Vector{Int64})::Grid
    cellSize = [dims[1] / cells[1], dims[2] / cells[2]]
    grid = Grid(cellSize, cellSize[1] * cellSize[2], Matrix{GridNode}(undef, cells[1], cells[2]), cells[1])
    for y in 1:cells[2]
        for x in 1:cells[1]
            grid.nodes[x, y] = GridNode(0.0, false, [0.0, 0.0], [0.0, 0.0])
        end
    end
    return grid
end


particle_area = PARTICLE_DIAM^2
particle_mass = particle_area * DENSITY
segmentSize = 0.3
area = segmentSize * segmentSize * 0.8
particleCount = round(Int64, area / particle_area)
tempArea = 0.2 * 0.2
particles2 = round(Int64, tempArea / particle_area)


max_velocity = 0.0
particles = Array{Particle}([])
for p in 1:particleCount
    push!(particles, Particle([0.3 + segmentSize * rand(), 0.3 + segmentSize * rand()], [2.0, 0.0],particle_mass))
end

for p in 1:particles2
    push!(particles, Particle([0.7 + 0.2 * rand(), 0.7 + 0.2 * rand()], [0.0, -2.0],particle_mass))
end

max_velocity = 2.0^2 + 0.0^2

self = Grid([1.0, 1.0], [64, 64])

function initializeMass()
    for i in 1:self.cellcount, j in 1:self.cellcount
        self.nodes[i, j].mass = 0.0
        self.nodes[i, j].active = false
        self.nodes[i, j].velocity = [0.0, 0.0]
        self.nodes[i, j].velocity_new = [0.0, 0.0]
    end
    for p in particles
        for i in 1:4, j in 1:4
            p.weights[i, j] = 0.0
            p.weight_gradientsX[i, j] = 0.0
            p.weight_gradientsY[i, j] = 0.0
        end
    end
    for p in particles
        gridIndex::Vector{Int64} = floor.(Int64, p.position ./ self.cellsize[1])
        for i in 0:3, j in 0:3
            distance = (p.position ./ self.cellsize[1]) - (gridIndex + [i-1, j-1])
            # println(distance)
            indexI = gridIndex[1] + i
            indexJ = gridIndex[2] + j
            wy = bspline(distance[2])
            dy = bsplineSlope(distance[2])
            wx = bspline(distance[1])
            dx = bsplineSlope(distance[1])
            weight::Float64 = wx * wy
            p.weights[i+1, j+1] = weight
            p.weight_gradientsX[i+1, j+1] = (dx * wy) / self.cellsize[1]
            p.weight_gradientsY[i+1, j+1] = (wx * dy) / self.cellsize[2]
            self.nodes[indexI, indexJ].mass += weight * p.mass
        end
    end
end

function collisionGrid()
    for i in 1:self.cellcount, j in 1:self.cellcount
        if self.nodes[i, j].active
            new_pos = (self.nodes[i, j].velocity_new .* (dt ./ self.cellsize[1])) + [i-1, j-1]
            if new_pos[1] < BSPLINE_RADIUS || new_pos[1] > self.cellcount - BSPLINE_RADIUS-1
                self.nodes[i, j].velocity_new[1] = 0
                self.nodes[i, j].velocity_new[2] *= 0.9
            end
            if new_pos[2] < BSPLINE_RADIUS || new_pos[2] > self.cellcount - BSPLINE_RADIUS-1
                self.nodes[i, j].velocity_new[2] = 0
                self.nodes[i, j].velocity_new[1] *= 0.9
            end
        end
    end
end

function initializeVelocities()
    for p in particles
        gridIndex::Vector{Int64} = floor.(Int64, p.position ./ self.cellsize[1])
        for i in 0:3, j in 0:3
            indexI = gridIndex[1] + i
            indexJ = gridIndex[2] + j
            w = p.weights[i+1, j+1]
            if w > BSPLINE_EPSILON
                self.nodes[indexI, indexJ].velocity += p.velocity .* (w * p.mass)
                self.nodes[indexI, indexJ].active = true
            end
        end
    end
    for i in 1:self.cellcount, j in 1:self.cellcount
        if self.nodes[i, j].active 
            self.nodes[i, j].velocity = self.nodes[i, j].velocity ./ self.nodes[i, j].mass
        end
    end
    collisionGrid()
end

function calculateVolumes()
    for p in particles
        gridIndex::Vector{Int64} = floor.(Int64, p.position ./ self.cellsize[1])
        density = 0.0
        for i in 0:3, j in 0:3
            indexI = gridIndex[1] + i
            indexJ = gridIndex[2] + j
            w = p.weights[i+1, j+1]
            if w > BSPLINE_EPSILON
                density += w * self.nodes[indexI, indexJ].mass
            end
        end
        density /= self.node_area
        p.volume = p.mass / density
    end
end

function explicitVelocities(gravity::Vector{Float64})
    for p in particles
        energy = energyDerivative(p)
        gridIndex::Vector{Int64} = floor.(Int64, p.position ./ self.cellsize[1])
        for i in 0:3, j in 0:3
            indexI = gridIndex[1] + i
            indexJ = gridIndex[2] + j
            w = p.weights[i+1, j+1]
            if w > BSPLINE_EPSILON
                self.nodes[indexI, indexJ].velocity_new += energy * [p.weight_gradientsX[i+1, j+1], p.weight_gradientsY[i+1, j+1]]
            end
        end
    end
    for i in 1:self.cellcount, j in 1:self.cellcount
        if self.nodes[i, j].active == true
            self.nodes[i, j].velocity_new = self.nodes[i, j].velocity + dt .* ([0.0, GRAVITY] - (self.nodes[i, j].velocity_new ./ self.nodes[i, j].mass))
        end
    end
    collisionGrid()
end


function collisionParticles()
    for p in particles
        grid_position = p.position ./ self.cellsize[1]
        new_pos = grid_position + (dt .* (p.velocity ./ self.cellsize[1]))
        if new_pos[1] < BSPLINE_RADIUS || new_pos[1] > self.cellcount - BSPLINE_RADIUS
            p.velocity[1] = -0.9 * p.velocity[1]
        end
        if new_pos[2] < BSPLINE_RADIUS || new_pos[2] > self.cellcount - BSPLINE_RADIUS
            p.velocity[2] = -0.9 * p.velocity[2]
        end
    end
end

function updateVelocities()
    for p in particles
        pic = [0.0, 0.0]
        flip = copy(p.velocity)
        p.velocity_gradient = zeros(Float64, 2, 2)
        gridIndex::Vector{Int64} = floor.(Int64, p.position ./ self.cellsize[1])
        for i in 0:3, j in 0:3
            indexI = gridIndex[1] + i
            indexJ = gridIndex[2] + j
            w = p.weights[i+1, j+1]
            if w > BSPLINE_EPSILON
                pic += self.nodes[indexI, indexJ].velocity_new .* w
                flip += (self.nodes[indexI, indexJ].velocity_new - self.nodes[indexI, indexJ].velocity) .* w
                p.velocity_gradient += self.nodes[indexI, indexJ].velocity_new * transpose([p.weight_gradientsX[i+1, j+1], p.weight_gradientsY[i+1, j+1]])
            end
        end
        p.velocity = flip .* 0.95 + pic .* (1 - 0.95)
    end
    collisionParticles()
end

initializeMass()
calculateVolumes()

frame = 0


for i in 1:1000
    global frame += 1
    gravity = [0.0, GRAVITY]
    initializeMass()
    initializeVelocities()
    explicitVelocities(gravity)
    updateVelocities()
    global max_velocity = 0.0
    for p in particles
        p.position = p.position + (dt .* p.velocity)

        p.velocity_gradient = p.velocity_gradient .* dt
        p.velocity_gradient = p.velocity_gradient + [1.0 0.0; 0.0 1.0]
        p.def_elastic = p.velocity_gradient * p.def_elastic

        f_all::Matrix{Float64} = p.def_elastic * p.def_plastic
        svdResult = svd(copy(p.def_elastic))
        w::Matrix{Float64} = copy(svdResult.U)
        v::Matrix{Float64} = copy(svdResult.V)
        e::Matrix{Float64} = diagm(copy(svdResult.S))
        for i in 1:2
            if e[i, i] < CRIT_COMPRESS
                e[i, i] = CRIT_COMPRESS
            elseif e[i, i] > CRIT_STRETCH
                e[i, i] = CRIT_STRETCH
            end
        end
        vcopy = copy(v * inv(e))
        wcopy = copy(w * e)
        p.def_plastic = vcopy * w' * f_all
        p.def_elastic = wcopy * v'

    end

    println(frame)

    sum = 0.0
    vsum = 0.0
    for p in particles
        sum += p.mass
        vsum += p.volume
    end
    println(sum)
    println(vsum)
    println(particles[1].velocity)

    @png begin
        sethue("pink")
        for p in particles
            ellipse(p.position[1] * 600 - 300, p.position[2] * 600 - 300, 3, 3, :fill)
        end
    end 600 600 "./images/mpm-$(i).png"
end
