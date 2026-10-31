# Curvemaster

A first-person shooter in **curved 3-space** (hyperbolic H³, flat E³ or
spherical S³), rendered entirely by raymarching signed distance functions in
a generated fragment shader. Written in Rust with
[macroquad](https://github.com/not-fl3/macroquad).

## Run

```bash
cargo run --release
```

### Windows .exe (cross-compiled from Linux)

```bash
rustup target add x86_64-pc-windows-gnu
```

```bash
sudo apt-get install gcc-mingw-w64-x86-64
```

```bash
cargo build --release --target x86_64-pc-windows-gnu
```

The result is `target/x86_64-pc-windows-gnu/release/curvemaster.exe`, a single
self-contained file that only depends on DLLs that ship with Windows.

## Controls

| Input | Action |
|---|---|
| W / A / S / D | move forward / left / back / right |
| Space / Ctrl (or C) | move up / down |
| Shift | move faster |
| Mouse | look (free 6-DOF, no fixed "up") |
| Q / E | roll |
| Left mouse | shoot (click also captures the mouse) |
| Esc | open the menu (resume, change space, new game) |
| `[` / `]` | lower / raise the render resolution |
| R | restart after dying |

Red spheres drift around until they get a clear line of sight to you; then
their eye opens and they fly straight at you along the connecting geodesic.
Touching one costs 20 health. Shoot them first.

## Main menu

* **Space**: Hyperbolic H3, Euclidean E3, or Spherical S3.
* **Grid cells**: the honeycomb that fills space.
  * H3: right-angled dodecahedra `{5,3,4}` and the larger right-angled
    Löbell polyhedra L6, L7, L8 (14–18 faces). Infinite.
  * E3: cubes `{4,3,4}`. Infinite.
  * S3: the six regular 4-polytopes, from the 5-cell (5 tetrahedra) up to the
    600-cell (600 tetrahedra). The world is finite: walk far enough in a
    straight line and you come back from behind. Objects near the
    antipodal point look magnified.
* **Curvature radius R** / **Cell size**: in curved space a regular grid's
  cell size is fixed relative to the curvature radius. The two settings
  are linked, and choosing a different grid gives a different ratio. In E3 the
  cube size is free.

Esc during play opens the menu (Resume / New game).

## How it works

**Unified model.** With curvature sign `k ∈ {-1, 0, +1}` points are
4-vectors on `x²+y²+z²+k·w² = k` (H3 hyperboloid, S3 unit sphere) or
`(x,y,z,1)` (E3). Geodesics are `p(t) = C(t)·p + S(t)·v` with
`(C,S) = (cosh,sinh) | (1,t) | (cos,sin)`. The player is an orthonormal frame,
movement is a boost/translation/rotation along a geodesic, and look/roll are
rotations. Internal lengths are measured in units of R, and gameplay sizes and
speeds are in world units, so changing R really changes how curved the space
feels.

**Folding.** Every face plane of a honeycomb cell is a mirror of the
honeycomb. The raymarcher keeps the ray inside the central cell and
reflects the ray's position and direction across any face it crosses. Physics
folds points the same way, and when the player leaves the central cell the
whole world is reflected back. Coordinates therefore never blow up, even in
infinite H3.

**Cell identity** (the procedural seed for walls and orbs) has to stay the same
no matter which reflections reached the cell:

* *Right-angled H3 cells*: the face reflections generate a right-angled
  Coxeter group with the cell as strict fundamental domain, so the per-face
  parity of reflections is a well-defined label.
* *E3 and S3*: the shader tracks the matrix from local to absolute
  coordinates (a rank-one update per reflection) and quantises the cell's
  absolute centre. For S3, the CPU enumerates every cell of the finite world
  and picks a quantisation that keeps every centre clear of rounding
  boundaries by about 100× the f32 error, so GPU and CPU always agree.

Each face is seeded by its own shared identity, so both neighbours agree on
whether it is open, solid or holed. Edge beams use the exact distance to the
intersection line of two planes, `S(d)² = (sᵢ² − 2g·sᵢsⱼ + sⱼ²)/(1−g²)`.
Enemies and shots are intersected analytically with the geodesic ray.

**One SDF, two places.** `src/world.rs` evaluates the scene on the CPU in f64
for collisions, line of sight and spawning. `src/shader.rs` generates the
matching GLSL 1.00 from the same cell data, and hashing uses only exact
small-integer float arithmetic.

## Tests

```bash
cargo test
```

Runs for every geometry and grid: geodesic and frame maths, cell combinatorics
(including that the S3 tilings close up with 5/8/16/24/120/600 cells),
path-independent cell labels, shared-face agreement, projectile/wall and
projectile/enemy collisions, enemy chasing, walls blocking sight, and long
walks staying numerically sane.

Set `CURVEMASTER_SHOT=out.png` (optionally with `CURVEMASTER_SPACE=h|e|s`,
`CURVEMASTER_TILING=<n>`, `CURVEMASTER_R=<radius>`) to run a short scripted
fly-through and save a screenshot (handy under `xvfb-run`).
