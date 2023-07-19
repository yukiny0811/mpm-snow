
# import Libraries
# Linear Algebra: used for singular value decomposition
# Luxor: used for creating output png images of the simulation result
# Colors: Color support for Luxor
using LinearAlgebra, Luxor, Colors

dt = 0.0002 # timestep
CRIT_COMPRESS = 1-1.9e-2 # Fracture threshold for compression
CRIT_STRETCH = 1+7.5e-3 # Fracture threshold for stretching
HARDENING = 5.0 # How much plastic deformation strengthens material
YOUNGS_MODULUS = 1.5e5 # Young's modulus (springiness)
POISSONS_RATIO = 0.2 # Poisson's ratio (transverse/axial strain ratio) 
BSPLINE_EPSILON = 1e-4
BSPLINE_RADIUS = 2
PARTICLE_DIAM = .0072 # Diameter of each particle; smaller = higher resolution
DENSITY = 100.0 # Density of snow in kg/m^2
GRAVITY = -9.8

# Hardening parameters
LAMBDA = YOUNGS_MODULUS*POISSONS_RATIO/((1+POISSONS_RATIO)*(1-2*POISSONS_RATIO))
MU = YOUNGS_MODULUS/(2+2*POISSONS_RATIO)

# Particle parameters
PARTICLE_AREA = PARTICLE_DIAM^2
PARTICLE_MASS = PARTICLE_AREA * DENSITY

# Particle class
mutable struct Particle
    volume::Float64 #体積
    mass::Float64 #質量
    position::Vector{Float64} #位置
    velocity::Vector{Float64} #速度
    def_elastic::Matrix{Float64} #変形勾配の弾性部分
    def_plastic::Matrix{Float64} #変形勾配の塑性部分

    # intermediate parameters (中間パラメーター)
    weight_gradientsX::Matrix{Float64} #重みの勾配X
    weight_gradientsY::Matrix{Float64} #重みの勾配Y
    weights::Matrix{Float64} #重み
    velocity_gradient::Matrix{Float64} #速度勾配
end

#Particle constructor
function Particle(pos::Vector{Float64}, vel::Vector{Float64}, mass::Float64)::Particle
    volume = 0.0 #とりあえず
    def_elastic = [1.0 0.0; 0.0 1.0] #変形勾配の初期値は単位行列
    def_plastic = [1.0 0.0; 0.0 1.0] #変形勾配の初期値は単位行列
    weight_gradientsX = zeros(Float64, 4, 4) #4*4の0行列をつくっておく
    weight_gradientsY = zeros(Float64, 4, 4) #4*4の0行列をつくっておく
    weights = zeros(Float64, 4, 4) #4*4の0行列をつくっておく
    velocity_gradient = zeros(Float64, 2, 2)#2*2の0行列をつくっておく
    return Particle(volume, mass, pos, vel, def_elastic, def_plastic, weight_gradientsX, weight_gradientsY, weights, velocity_gradient)
end

#GridNode class
mutable struct GridNode
    mass::Float64 #質量
    velocity::Vector{Float64} #速度
    velocity_new::Vector{Float64} #新しい速度

    #intermediate parameters (中間パラメーター)
    active::Bool #アクティブかどうか
end

#Grid class
mutable struct Grid
    cellsize::Float64 #セルの一辺の長さ
    node_area::Float64 #セルの面積
    nodes::Matrix{GridNode} #セルたち
    cellcount::Int64 #グリッドの一辺のセル数
end

#Grid constructor
function Grid(cells::Int64)::Grid #cellsはグリッドの一辺のセル数
    cellSize = 1.0 / cells #セルの一辺の長さ
    grid = Grid(cellSize, cellSize * cellSize, Matrix{GridNode}(undef, cells, cells), cells)
    for i in 1:cells, j in 1:cells
        grid.nodes[i, j] = GridNode(0.0, [0.0, 0.0], [0.0, 0.0], false)
    end
    return grid
end

#one-dimensional cubic B-splines
function N(x::Float64)::Float64
    x = abs(x)
    w = 0.0
    if x < 1.0
        w = 1.0/2.0 * x^3 - x^2 + 2.0/3.0;
    elseif x < 2.0
        w = -1.0/6.0 * x^3 + x^2 - 2x + 4.0/3.0; 
    else
        return 0.0
    end
    if w < BSPLINE_EPSILON #値が小さすぎるといろいろ発散しちゃうから制限を設ける
        return 0.0
    end
    return w
end

#N(x)のxでの偏微分
function N_x(x::Float64)::Float64
    absX = abs(x)
    if absX < 1.0
        return 3.0/2.0*x*absX - 2.0*x;
    elseif absX < 2.0
        return -1.0/2.0 * x*absX + 2.0x - 2.0*x/absX;
    else
        return 0.0
    end
end

particles = Array{Particle}([]) #パーティクルの配列を作成
grid = Grid(64) #グリッドを生成

#雪のかたまり1を生成
segmentSize1 = 0.3 #雪のかたまり1の一辺の長さ
area1 = segmentSize1 * segmentSize1 #雪のかたまり1の面積
particleCount1 = round(Int64, area1 / PARTICLE_AREA) #雪のかたまり1の面積と雪の粒子の面積からパーティクル数を計算
for p in 1:particleCount1
    push!(particles, Particle([0.3 + segmentSize1 * rand(), 0.3 + segmentSize1 * rand()], [2.0, 0.0], PARTICLE_MASS))
end

#雪のかたまり2を生成
segmentSize2 = 0.2 #雪のかたまり2の一辺の長さ
area2 = segmentSize2 * segmentSize2 #雪のかたまり2の面積
particleCount2 = round(Int64, area2 / PARTICLE_AREA) #雪のかたまり2の面積と雪の粒子の面積からパーティクル数を計算
for p in 1:particleCount2
    push!(particles, Particle([0.7 + segmentSize2 * rand(), 0.7 + segmentSize2 * rand()], [0.0, -2.0], PARTICLE_MASS))
end

function resetParameters()
    for i in 1:grid.cellcount, j in 1:grid.cellcount #reset all GridNode parameters to 0
        grid.nodes[i, j].mass = 0.0
        grid.nodes[i, j].active = false
        grid.nodes[i, j].velocity = [0.0, 0.0]
        grid.nodes[i, j].velocity_new = [0.0, 0.0]
    end
    for p in particles #reset all weight parameters to 0
        for i in 1:4, j in 1:4
            p.weights[i, j] = 0.0
            p.weight_gradientsX[i, j] = 0.0
            p.weight_gradientsY[i, j] = 0.0
        end
    end
end

# Rasterize particle mass to the grid
function P2Gmass()
    for p in particles
        gridIndex::Vector{Int64} = floor.(Int64, p.position ./ grid.cellsize) #パーティクル座標からグリッド座標に変換
        for i in 0:3, j in 0:3 #パーティクル周囲の4*4グリッドで重みの計算を行う
            distance = (p.position ./ grid.cellsize) - (gridIndex + [i-1, j-1]) #パーティクルと各グリッド点との距離
            indexI = gridIndex[1] + i #グリッド座標
            indexJ = gridIndex[2] + j #グリッド座標
            wy = N(distance[2])
            dy = N_x(distance[2])
            wx = N(distance[1])
            dx = N_x(distance[1])
            weight::Float64 = wx * wy
            p.weights[i+1, j+1] = weight
            p.weight_gradientsX[i+1, j+1] = (dx * wy) / grid.cellsize #なんでgrid.cellsizeで割るのかはわからないけどなぜかこうしないと動かない
            p.weight_gradientsY[i+1, j+1] = (wx * dy) / grid.cellsize #なんでgrid.cellsizeで割るのかはわからないけどなぜかこうしないと動かない
            grid.nodes[indexI, indexJ].mass += weight * p.mass
        end
    end
end

# Rasterize particle velocity to the grid
function P2Gvel()
    for p in particles
        gridIndex::Vector{Int64} = floor.(Int64, p.position ./ grid.cellsize) #パーティクル座標からグリッド座標に変換
        for i in 0:3, j in 0:3
            indexI = gridIndex[1] + i #グリッド座標
            indexJ = gridIndex[2] + j #グリッド座標
            w = p.weights[i+1, j+1]
            if w > BSPLINE_EPSILON #重みが小さすぎるといろいろ値が発散しちゃうから制限している
                grid.nodes[indexI, indexJ].velocity += p.velocity .* (w * p.mass)
                grid.nodes[indexI, indexJ].active = true #グリッドセルをアクティブにする
            end
        end
    end
    for i in 1:grid.cellcount, j in 1:grid.cellcount
        if grid.nodes[i, j].active 
            grid.nodes[i, j].velocity = grid.nodes[i, j].velocity ./ grid.nodes[i, j].mass
        end
    end
end

# calculateVolumes
function calculateVolumes()
    for p in particles
        gridIndex::Vector{Int64} = floor.(Int64, p.position ./ grid.cellsize)
        density = 0.0
        for i in 0:3, j in 0:3
            indexI = gridIndex[1] + i
            indexJ = gridIndex[2] + j
            w = p.weights[i+1, j+1]
            if w > BSPLINE_EPSILON
                density += w * grid.nodes[indexI, indexJ].mass
            end
        end
        density /= grid.node_area
        p.volume = p.mass / density
    end
end

function computeGridForces()
    for p in particles
        JP = det(p.def_plastic)
        JE = det(p.def_elastic)
        svdResult = svd(p.def_elastic)
        W = svdResult.U
        V = svdResult.V
        RE = W * V'
        mu = MU * exp(HARDENING * (1.0 - JP))
        lambda = LAMBDA * exp(HARDENING * (1.0 - JP))
        sigma = 2.0 * mu / JP * (p.def_elastic - RE) * p.def_elastic' + lambda / JP * (JE - 1.0) * JE * [1.0 0.0; 0.0 1.0]
        Jn = det(p.def_elastic * p.def_plastic)
        Vn = Jn * p.volume
        energy = Vn * sigma
        gridIndex::Vector{Int64} = floor.(Int64, p.position ./ grid.cellsize)
        for i in 0:3, j in 0:3
            indexI = gridIndex[1] + i
            indexJ = gridIndex[2] + j
            w = p.weights[i+1, j+1]
            if w > BSPLINE_EPSILON
                grid.nodes[indexI, indexJ].velocity_new -= energy * [p.weight_gradientsX[i+1, j+1], p.weight_gradientsY[i+1, j+1]]
            end
        end
    end
end

function updateGridVelocities()
    for i in 1:grid.cellcount, j in 1:grid.cellcount
        if grid.nodes[i, j].active == true
            grid.nodes[i, j].velocity_new = grid.nodes[i, j].velocity + dt .* ([0.0, GRAVITY] + (grid.nodes[i, j].velocity_new ./ grid.nodes[i, j].mass))
        end
    end
end

function collisionGrid()
    for i in 1:grid.cellcount, j in 1:grid.cellcount
        if grid.nodes[i, j].active
            new_pos = (grid.nodes[i, j].velocity_new .* (dt ./ grid.cellsize)) + [i-1, j-1]
            if new_pos[1] < BSPLINE_RADIUS || new_pos[1] > grid.cellcount - BSPLINE_RADIUS-1
                grid.nodes[i, j].velocity_new[1] = 0
                grid.nodes[i, j].velocity_new[2] *= 0.9
            end
            if new_pos[2] < BSPLINE_RADIUS || new_pos[2] > grid.cellcount - BSPLINE_RADIUS-1
                grid.nodes[i, j].velocity_new[2] = 0
                grid.nodes[i, j].velocity_new[1] *= 0.9
            end
        end
    end
end

function updateVelocities()
    for p in particles
        pic = [0.0, 0.0]
        flip = copy(p.velocity)
        p.velocity_gradient = zeros(Float64, 2, 2)
        gridIndex::Vector{Int64} = floor.(Int64, p.position ./ grid.cellsize)
        for i in 0:3, j in 0:3
            indexI = gridIndex[1] + i
            indexJ = gridIndex[2] + j
            w = p.weights[i+1, j+1]
            if w > BSPLINE_EPSILON
                pic += grid.nodes[indexI, indexJ].velocity_new .* w
                flip += (grid.nodes[indexI, indexJ].velocity_new - grid.nodes[indexI, indexJ].velocity) .* w
                p.velocity_gradient += grid.nodes[indexI, indexJ].velocity_new * transpose([p.weight_gradientsX[i+1, j+1], p.weight_gradientsY[i+1, j+1]])
            end
        end
        p.velocity = flip .* 0.95 + pic .* (1 - 0.95)
    end
end

function collisionParticles()
    for p in particles
        grid_position = p.position ./ grid.cellsize
        new_pos = grid_position + (dt .* (p.velocity ./ grid.cellsize))
        if new_pos[1] < BSPLINE_RADIUS || new_pos[1] > grid.cellcount - BSPLINE_RADIUS
            p.velocity[1] = -0.9 * p.velocity[1]
        end
        if new_pos[2] < BSPLINE_RADIUS || new_pos[2] > grid.cellcount - BSPLINE_RADIUS
            p.velocity[2] = -0.9 * p.velocity[2]
        end
    end
end

function updateDeformationGradient()
    for p in particles
        p.velocity_gradient = [1.0 0.0; 0.0 1.0] + dt .* p.velocity_gradient
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
        p.def_plastic = v * inv(e) * w' * f_all
        p.def_elastic = w * e * v'
    end
end

function updateParticlePositions()
    for p in particles
        p.position = p.position + (dt .* p.velocity)
    end
end

P2Gmass()
calculateVolumes()

frame = 0


for i in 1:1000
    global frame += 1
    gravity = [0.0, GRAVITY]
    resetParameters()
    P2Gmass()
    P2Gvel()
    computeGridForces()
    updateGridVelocities()
    collisionGrid()
    updateDeformationGradient()
    updateVelocities()
    updateParticlePositions()

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

    if i % 20 != 0
        continue    
    end

    @png begin
        sethue("pink")
        for p in particles
            ellipse(p.position[1] * 600 - 300, p.position[2] * 600 - 300, 3, 3, :fill)
        end
    end 600 600 "./images/mpm-$(floor(Int64, i/20)).png"
end
