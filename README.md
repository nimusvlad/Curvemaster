# Curvemaster

A first-person shooter in **infinite hyperbolic 3-space (H³)**, rendered entirely by
raymarching signed distance functions in a fragment shader. Written in Rust with
[macroquad](https://github.com/not-fl3/macroquad).

## Run

```bash
cargo run --release
```

## Controls

| Input | Action |
|---|---|
| W / A / S / D | move forward / left / back / right |
| Space / Ctrl (or C) | move up / down |
| Shift | move faster |
| Mouse | look (free 6-DOF, no fixed "up") |
| Q / E | roll |
| Left mouse | shoot (click also captures the mouse) |
| Esc | release the mouse |
| `[` / `]` | lower / raise the render resolution |
| R | restart after dying |

Red spheres drift around until they get a clear line of sight to you; then
their eye opens and they fly straight at you along the connecting geodesic.
Touching one costs 20 health. Shoot them first.

## How it works

**Model.** Points are 4-vectors on the hyperboloid `x²+y²+z²−w² = −1` with the
Minkowski form. The player is a Lorentz frame (right, up, forward, position);
movement is a boost, looking is a rotation. Rays, projectiles and enemies all
follow geodesics `p(t) = cosh t·p + sinh t·v`.

**The world** is the right-angled dodecahedral honeycomb `{5,3,4}`: regular
dodecahedra with 90° dihedral angles, eight around each vertex. Because the
angles are right angles, the 12 face reflections of one cell form a Coxeter
group with that cell as a strict fundamental domain, so:

* the raymarcher keeps the ray inside the central cell, reflecting ray position
  and direction across whichever face it crosses. Coordinates stay small no
  matter how far the ray goes;
* each cell gets a well-defined 12-bit **label** (the parity of reflections per
  face), which seeds the procedural content: per-face plates (open, solid, or
  pierced by a round hole; seeded so both cells sharing a face agree), a
  coloured lamp, and an optional glowing orb at the centre. Edges carry beams
  and vertices carry nodes;
* when the player leaves the central cell the whole world (player, enemies,
  projectiles) is reflected back and the global label is updated, so the
  world is unbounded but numbers never blow up.

All distances are true hyperbolic distances: `asinh⟨n,p⟩` to a plane,
`asinh√(s_i²+s_j²)` to an edge (intersection of orthogonal planes),
`acosh(−⟨p,c⟩)` to a point. Enemies and projectiles are intersected
analytically with the geodesic ray (a quadratic in `eᵗ`).

**One SDF, two places.** `src/world.rs` evaluates the scene on the CPU in f64
for collisions, line-of-sight and spawning. `src/shader.rs` generates the
matching GLSL 1.00 (normals, edges and vertices unrolled from the same Rust
data), and the cell hash uses only exact small-integer float arithmetic, so the
GPU and CPU agree on which walls exist.

## Tests

```bash
cargo test
```

Covers frame/geodesic maths, the honeycomb's right angles, path-independent cell
labels, shared-face agreement, projectile/wall and projectile/enemy collisions,
enemy chasing, and walls blocking sight.

Set `CURVEMASTER_SHOT=out.png` to run a short scripted fly-through and save a
screenshot (handy under `xvfb-run`).
