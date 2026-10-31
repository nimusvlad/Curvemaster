//! Procedural worlds built on honeycombs of constant-curvature space.
//!
//! The space is tiled by congruent cells. Every face plane of the central cell
//! is a mirror of the honeycomb, so any point can be *folded* into the central
//! cell by repeated face reflections. Raymarching, physics and recentering all
//! work in the central cell, so coordinates never grow.
//!
//! Each cell needs a stable identity to seed its procedural content:
//! * **Parity labels** (right-angled hyperbolic cells): the 12+ face
//!   reflections generate a right-angled Coxeter group with the cell as strict
//!   fundamental domain, so the per-face parity of reflections used is well
//!   defined. Recentering just toggles bits — unbounded worlds stay exact.
//! * **Centre keys** (Euclidean cubes, all spherical tilings): track the matrix
//!   from local to absolute coordinates and quantise the cell's absolute
//!   centre. The quantisation is checked on the CPU to be unambiguous with a
//!   wide safety margin, so GPU (f32) and CPU (f64) always agree.
//!
//! Faces are keyed by their own (shared) identity, so both cells agree on
//! every wall. The SDF is evaluated identically on CPU (f64) and GPU (f32, see
//! `shader.rs`); the hash uses only exact small-integer float arithmetic.

use crate::geom::*;
use std::collections::VecDeque;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TilingKind {
    /// Right-angled Löbell polyhedron L(n) in H3 (n = 5 is the dodecahedron).
    Lobell(usize),
    /// Euclidean cube honeycomb.
    Cubes,
    /// Regular honeycomb {p,q,r} of S3 (a regular 4-polytope).
    Regular(u8, u8, u8),
}

pub struct TilingOption {
    pub name: &'static str,
    pub kind: TilingKind,
}

pub fn catalog(space: Space) -> Vec<TilingOption> {
    use TilingKind::*;
    let o = |name, kind| TilingOption { name, kind };
    match space {
        Space::Hyperbolic => vec![
            o("Right-angled dodecahedra {5,3,4}", Lobell(5)),
            o("Right-angled Lobell L6 (14 faces)", Lobell(6)),
            o("Right-angled Lobell L7 (16 faces)", Lobell(7)),
            o("Right-angled Lobell L8 (18 faces)", Lobell(8)),
        ],
        Space::Euclidean => vec![o("Cubes {4,3,4}", Cubes)],
        Space::Spherical => vec![
            o("5-cell: 5 tetrahedra {3,3,3}", Regular(3, 3, 3)),
            o("8-cell: 8 cubes {4,3,3}", Regular(4, 3, 3)),
            o("16-cell: 16 tetrahedra {3,3,4}", Regular(3, 3, 4)),
            o("24-cell: 24 octahedra {3,4,3}", Regular(3, 4, 3)),
            o("120-cell: 120 dodecahedra {5,3,3}", Regular(5, 3, 3)),
            o("600-cell: 600 tetrahedra {3,3,5}", Regular(3, 3, 5)),
        ],
    }
}

#[derive(Clone, Debug)]
pub enum LabelMode {
    Parity,
    Center { scale: f64, off: V4 },
}

/// Identity of the central cell.
#[derive(Clone, Copy, Debug)]
pub enum Label {
    Parity(f64),
    /// Local → absolute coordinates.
    Center(M4),
}

/// Geometry of the central cell of a honeycomb.
pub struct Shape {
    pub g: Geom,
    /// Face plane covectors, outward (`φ·p > 0` outside).
    pub faces: Vec<V4>,
    /// Euclidean unit directions of the faces (hole axes).
    pub dirs: Vec<[f64; 3]>,
    /// Face centres (foot of the perpendicular from the cell centre).
    pub fcent: Vec<V4>,
    /// Edges as (face a, face b, dual Gram value).
    pub edges: Vec<(usize, usize, f64)>,
    pub verts: Vec<V4>,
    /// Inradius of each face polygon.
    pub face_inr: Vec<f64>,
    /// Inradius of the cell.
    pub rho: f64,
    pub mode: LabelMode,
    /// Number of cells for a finite (spherical) world.
    pub cell_count: Option<usize>,
}

fn unit3(v: [f64; 3]) -> [f64; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    [v[0] / l, v[1] / l, v[2] / l]
}

fn polyhedron_dirs(p: u8, q: u8) -> Vec<[f64; 3]> {
    let phi = (1.0 + 5f64.sqrt()) / 2.0;
    let mut d = Vec::new();
    match (p, q) {
        (3, 3) => {
            d = vec![[1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]];
        }
        (4, 3) => {
            for i in 0..3 {
                for s in [1.0, -1.0] {
                    let mut v = [0.0; 3];
                    v[i] = s;
                    d.push(v);
                }
            }
        }
        (3, 4) => {
            for a in [1.0, -1.0] {
                for b in [1.0, -1.0] {
                    for c in [1.0, -1.0] {
                        d.push([a, b, c]);
                    }
                }
            }
        }
        (5, 3) => {
            for a in [1.0, -1.0] {
                for b in [1.0, -1.0] {
                    d.push([0.0, a, b * phi]);
                    d.push([a, b * phi, 0.0]);
                    d.push([b * phi, 0.0, a]);
                }
            }
        }
        _ => panic!("unsupported cell {{{p},{q}}}"),
    }
    d.into_iter().map(unit3).collect()
}

/// Plane `u·x = s·w` as a normalised covector.
fn plane(g: &Geom, u: [f64; 3], s: f64) -> V4 {
    let phi = [u[0], u[1], u[2], -s];
    scale(phi, 1.0 / g.ip(phi, phi).sqrt())
}

impl Shape {
    /// `e3_cell` is the Euclidean cube edge length (ignored elsewhere).
    pub fn build(space: Space, kind: TilingKind, e3_cell: f64) -> Shape {
        let g = Geom::new(space);
        let (faces, dirs): (Vec<V4>, Vec<[f64; 3]>) = match kind {
            TilingKind::Cubes => {
                let d = polyhedron_dirs(4, 3);
                (d.iter().map(|&u| [u[0], u[1], u[2], -e3_cell / 2.0]).collect(), d)
            }
            TilingKind::Regular(p, q, r) => {
                let d = polyhedron_dirs(p, q);
                let mut c = -1.0f64;
                for i in 1..d.len() {
                    c = c.max(d[0][0] * d[i][0] + d[0][1] * d[i][1] + d[0][2] * d[i][2]);
                }
                let ca = (2.0 * std::f64::consts::PI / r as f64).cos();
                // Dihedral angle 2π/r: ip(φi,φj) = -cos α for adjacent faces.
                let s2 = -(ca + c) / (g.k * (1.0 + ca));
                let s = s2.sqrt();
                (d.iter().map(|&u| plane(&g, u, s)).collect(), d)
            }
            TilingKind::Lobell(n) => {
                let pi = std::f64::consts::PI;
                let nf = n as f64;
                let tan2 = ((pi / nf).cos() - (2.0 * pi / nf).cos()) / 2.0;
                let phi_el = tan2.sqrt().atan();
                let (sp, cp) = (phi_el.sin(), phi_el.cos());
                let su = (cp * cp * (2.0 * pi / nf).cos() + sp * sp).sqrt();
                let st = sp / su;
                let mut f = vec![plane(&g, [0.0, 0.0, 1.0], st), plane(&g, [0.0, 0.0, -1.0], st)];
                let mut d = vec![[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]];
                for i in 0..n {
                    let a = 2.0 * pi * i as f64 / nf;
                    let u = [a.cos() * cp, a.sin() * cp, sp];
                    f.push(plane(&g, u, su));
                    d.push(u);
                    let b = a + pi / nf;
                    let l = [b.cos() * cp, b.sin() * cp, -sp];
                    f.push(plane(&g, l, su));
                    d.push(l);
                }
                (f, d)
            }
        };
        let nf = faces.len();

        // Vertices: intersections of three planes that lie on the cell.
        let mut verts: Vec<V4> = Vec::new();
        for a in 0..nf {
            for b in a + 1..nf {
                for c in b + 1..nf {
                    let x = cross4(faces[a], faces[b], faces[c]);
                    let cands = match space {
                        Space::Euclidean => {
                            if x[3].abs() < 1e-9 {
                                continue;
                            }
                            vec![scale(x, 1.0 / x[3])]
                        }
                        Space::Hyperbolic => {
                            if -g.ip(x, x) <= 1e-12 {
                                continue;
                            }
                            vec![g.norm_point(x)]
                        }
                        Space::Spherical => {
                            let p = g.norm_point(x);
                            vec![p, scale(p, -1.0)]
                        }
                    };
                    for p in cands {
                        if faces.iter().all(|&f| dot4(f, p) <= 1e-9)
                            && !verts.iter().any(|&v| dot4(lin(v, 1.0, p, -1.0), lin(v, 1.0, p, -1.0)) < 1e-12)
                        {
                            verts.push(p);
                        }
                    }
                }
            }
        }
        let on = |f: V4, v: V4| dot4(f, v).abs() < 1e-8;
        let mut edges = Vec::new();
        for a in 0..nf {
            for b in a + 1..nf {
                if verts.iter().filter(|&&v| on(faces[a], v) && on(faces[b], v)).count() >= 2 {
                    edges.push((a, b, g.ip(faces[a], faces[b])));
                }
            }
        }
        // Euler characteristic sanity check: V - E + F = 2.
        assert_eq!(verts.len() + nf, edges.len() + 2, "bad cell for {kind:?}");

        let fcent: Vec<V4> = faces
            .iter()
            .map(|&f| g.norm_point(lin(ORIGIN, 1.0, g.refl_vec(f), -dot4(f, ORIGIN))))
            .collect();
        let face_inr = (0..nf)
            .map(|i| {
                edges
                    .iter()
                    .filter(|e| e.0 == i || e.1 == i)
                    .map(|&(a, b, gr)| {
                        let (sa, sb) = (dot4(faces[a], fcent[i]), dot4(faces[b], fcent[i]));
                        let f2 = (sa * sa - 2.0 * gr * sa * sb + sb * sb) / (1.0 - gr * gr);
                        g.asf(f2.max(0.0).sqrt())
                    })
                    .fold(f64::INFINITY, f64::min)
            })
            .collect();
        let rho = faces
            .iter()
            .map(|f| g.asf(-dot4(*f, ORIGIN)))
            .fold(f64::INFINITY, f64::min);

        let mut shape = Shape {
            g,
            faces,
            dirs,
            fcent,
            edges,
            verts,
            face_inr,
            rho,
            mode: LabelMode::Parity,
            cell_count: None,
        };
        shape.mode = match kind {
            TilingKind::Lobell(_) => LabelMode::Parity,
            TilingKind::Cubes => LabelMode::Center {
                scale: 2.0 / e3_cell,
                off: [0.5; 4],
            },
            TilingKind::Regular(..) => {
                let (n, mode) = shape.spherical_keys();
                shape.cell_count = Some(n);
                mode
            }
        };
        shape
    }

    /// Enumerate every cell of a finite spherical honeycomb, then pick a
    /// quantisation of absolute coordinates that separates all cell centres
    /// (and all face centres) with a safe margin from rounding boundaries.
    fn spherical_keys(&self) -> (usize, LabelMode) {
        let g = &self.g;
        let mut mats = vec![M4::identity()];
        let mut centers = vec![ORIGIN];
        let mut queue = VecDeque::from([0usize]);
        while let Some(i) = queue.pop_front() {
            for f in &self.faces {
                let m = mats[i].mul_reflect(g, *f);
                let c = g.norm_point(m.c[3]);
                if !centers.iter().any(|&x| dot4(x, c) > 1.0 - 1e-9) {
                    mats.push(m);
                    centers.push(c);
                    queue.push_back(mats.len() - 1);
                }
            }
            assert!(mats.len() <= 1000, "spherical tiling did not close");
        }
        let mut fcs: Vec<V4> = Vec::new();
        for m in &mats {
            for fc in &self.fcent {
                let c = m.apply(*fc);
                if !fcs.iter().any(|&x| dot4(x, c) > 1.0 - 1e-9) {
                    fcs.push(c);
                }
            }
        }
        let margin = 2e-4; // in coordinate units; f32 error is ~1e-6
        let mut seed: u64 = 0x9E3779B97F4A7C15;
        let mut rnd = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        for &sc in &[3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 16.0] {
            for _ in 0..600 {
                let off = [rnd(), rnd(), rnd(), rnd()];
                let ok = |set: &Vec<V4>| {
                    let mut keys: Vec<[i64; 4]> = Vec::with_capacity(set.len());
                    for c in set {
                        let mut k = [0i64; 4];
                        for j in 0..4 {
                            let x = c[j] * sc + off[j];
                            let fr = x - x.floor();
                            if fr < margin * sc || fr > 1.0 - margin * sc {
                                return false;
                            }
                            k[j] = x.floor() as i64;
                        }
                        keys.push(k);
                    }
                    keys.sort();
                    keys.windows(2).all(|w| w[0] != w[1])
                };
                if ok(&centers) && ok(&fcs) {
                    return (centers.len(), LabelMode::Center { scale: sc, off });
                }
            }
        }
        panic!("no safe quantisation found");
    }
}

// --- Hashing --------------------------------------------------------------

/// Float modulo robust to division rounding, for integer-valued inputs.
#[inline]
pub fn imod(x: f64, m: f64) -> f64 {
    x - m * ((x + 0.5) / m).floor()
}

/// Hash of an integer 4-vector, in [0,1). Every intermediate value is an
/// integer below ~2^20, so f32 (GPU) and f64 (CPU) agree exactly.
pub fn hash4(q: V4, salt: f64) -> f64 {
    let mut a = imod(imod(q[0], 1009.0) * 37.0 + salt * 101.0 + 7.0, 1009.0);
    a = imod(a * a + imod(q[1], 1009.0) * 13.0 + 11.0, 1013.0);
    a = imod(a * (imod(q[2], 1009.0) + 3.0) + 5.0, 1019.0);
    a = imod(a * (imod(q[3], 1009.0) + 7.0) + salt + 1.0, 1021.0);
    a = imod(a * a + 3.0, 1031.0);
    a / 1031.0
}

/// Split a non-negative integer < 2^24 into exact 8-bit chunks.
pub fn chunks(k: f64) -> V4 {
    let c2 = (k / 65536.0).floor();
    let r = k - c2 * 65536.0;
    let c1 = (r / 256.0).floor();
    [r - c1 * 256.0, c1, c2, 0.0]
}

pub fn bit(lab: f64, i: usize) -> f64 {
    imod((lab / (1u32 << i) as f64).floor(), 2.0)
}

pub fn toggle(lab: f64, i: usize) -> f64 {
    let p = (1u32 << i) as f64;
    if bit(lab, i) > 0.5 {
        lab - p
    } else {
        lab + p
    }
}

pub fn quantize(c: V4, sc: f64, off: V4) -> V4 {
    [
        (c[0] * sc + off[0]).floor(),
        (c[1] * sc + off[1]).floor(),
        (c[2] * sc + off[2]).floor(),
        (c[3] * sc + off[3]).floor(),
    ]
}

// --- World -----------------------------------------------------------------

pub const MAT_FRAME: u8 = 0;
pub const MAT_PLATE: u8 = 1;
pub const MAT_ORB: u8 = 2;

#[derive(Clone, Debug)]
pub struct CellParams {
    /// Per face: -1 open, 0 solid, >0 hole radius.
    pub face: Vec<f64>,
    pub orb: f64,
}

pub struct World {
    pub shape: Shape,
    pub beam: f64,
    pub node: f64,
    pub plate: f64,
    pub slack: f64,
}

impl World {
    pub fn new(shape: Shape) -> World {
        let r = shape.rho;
        World {
            beam: 0.068 * r,
            node: 0.136 * r,
            plate: 0.037 * r,
            slack: 0.025 * r,
            shape,
        }
    }

    pub fn g(&self) -> &Geom {
        &self.shape.g
    }

    pub fn initial_label(&self) -> Label {
        match self.shape.mode {
            LabelMode::Parity => Label::Parity(0.0),
            LabelMode::Center { .. } => Label::Center(M4::identity()),
        }
    }

    pub fn reflect_label(&self, l: Label, i: usize) -> Label {
        match l {
            Label::Parity(lab) => Label::Parity(toggle(lab, i)),
            Label::Center(m) => Label::Center(m.mul_reflect(self.g(), self.shape.faces[i])),
        }
    }

    pub fn cell_key(&self, l: &Label) -> V4 {
        match (l, &self.shape.mode) {
            (Label::Parity(lab), _) => chunks(*lab),
            (Label::Center(m), LabelMode::Center { scale, off }) => quantize(m.c[3], *scale, *off),
            _ => unreachable!(),
        }
    }

    pub fn face_key(&self, l: &Label, i: usize) -> V4 {
        match (l, &self.shape.mode) {
            (Label::Parity(lab), _) => {
                let cleared = lab - bit(*lab, i) * (1u32 << i) as f64;
                chunks(cleared * self.shape.faces.len() as f64 + i as f64)
            }
            (Label::Center(m), LabelMode::Center { scale, off }) => {
                quantize(m.apply(self.shape.fcent[i]), *scale, *off)
            }
            _ => unreachable!(),
        }
    }

    pub fn hole_max(&self, i: usize) -> f64 {
        self.shape.face_inr[i] - 2.2 * self.beam
    }

    pub fn face_param(&self, q: V4, i: usize) -> f64 {
        let r = hash4(q, 1.0);
        if r < 0.42 {
            -1.0
        } else if r < 0.80 {
            ((0.3 + 0.35 * hash4(q, 2.0)) * self.shape.face_inr[i]).min(self.hole_max(i))
        } else {
            0.0
        }
    }

    pub fn orb_param(&self, qc: V4) -> f64 {
        if hash4(qc, 3.0) < 0.33 {
            (0.17 + 0.25 * hash4(qc, 4.0)) * self.shape.rho
        } else {
            0.0
        }
    }

    pub fn cell_params(&self, l: &Label) -> CellParams {
        let face = (0..self.shape.faces.len())
            .map(|i| self.face_param(self.face_key(l, i), i))
            .collect();
        CellParams { face, orb: self.orb_param(self.cell_key(l)) }
    }

    /// Index of the face the point is furthest outside of, if any.
    pub fn outside_face(&self, p: V4) -> Option<usize> {
        let mut best = None;
        let mut bv = 1e-12;
        for (i, f) in self.shape.faces.iter().enumerate() {
            let s = dot4(*f, p);
            if s > bv {
                bv = s;
                best = Some(i);
            }
        }
        best
    }

    pub fn fold(&self, mut p: V4, mut l: Label) -> (V4, Label) {
        for _ in 0..512 {
            match self.outside_face(p) {
                Some(i) => {
                    p = self.g().reflect(p, self.shape.faces[i]);
                    l = self.reflect_label(l, i);
                }
                None => break,
            }
        }
        (p, l)
    }

    /// Unclamped SDF of the central cell's own content: (distance, material,
    /// max face value).
    pub fn content(&self, p: V4, cp: &CellParams) -> (f64, u8, f64) {
        let g = self.g();
        let sh = &self.shape;
        let s: Vec<f64> = sh.faces.iter().map(|f| dot4(*f, p)).collect();
        let smax = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut e2 = f64::INFINITY;
        for &(a, b, gr) in &sh.edges {
            e2 = e2.min((s[a] * s[a] - 2.0 * gr * s[a] * s[b] + s[b] * s[b]) / (1.0 - gr * gr));
        }
        let vd = sh.verts.iter().map(|v| g.dist(p, *v)).fold(f64::INFINITY, f64::min);
        let mut d = (g.asf(e2.max(0.0).sqrt()) - self.beam).min(vd - self.node);
        let mut mat = MAT_FRAME;
        for (i, &f) in cp.face.iter().enumerate() {
            if f < -0.5 {
                continue;
            }
            let mut slab = g.asf(s[i]).abs() - self.plate;
            if f > 0.0 {
                slab = slab.max(f - self.line_dist(p, sh.dirs[i]));
            }
            if slab < d {
                d = slab;
                mat = MAT_PLATE;
            }
        }
        if cp.orb > 0.0 {
            let o = g.dist(p, ORIGIN) - cp.orb;
            if o < d {
                d = o;
                mat = MAT_ORB;
            }
        }
        (d, mat, smax)
    }

    /// Distance to the geodesic through the cell centre with direction `u`.
    pub fn line_dist(&self, p: V4, u: [f64; 3]) -> f64 {
        let ux = u[0] * p[0] + u[1] * p[1] + u[2] * p[2];
        match self.g().space {
            Space::Hyperbolic => (p[3] * p[3] - ux * ux).max(1.0).sqrt().acosh(),
            Space::Spherical => (p[3] * p[3] + ux * ux).clamp(0.0, 1.0).sqrt().acos(),
            Space::Euclidean => (p[0] * p[0] + p[1] * p[1] + p[2] * p[2] - ux * ux).max(0.0).sqrt(),
        }
    }

    /// Raymarch-safe SDF inside the central cell.
    pub fn sdf_local(&self, p: V4, cp: &CellParams) -> (f64, u8) {
        let (d, mat, smax) = self.content(p, cp);
        (d.min(self.g().asf(-smax) + self.slack), mat)
    }

    pub fn sdf(&self, p: V4, l: Label) -> f64 {
        let (q, l) = self.fold(p, l);
        self.sdf_local(q, &self.cell_params(&l)).0
    }

    /// Accurate wall distance for physics (cell + neighbours across nearby faces).
    pub fn wall_dist(&self, p: V4, l: Label) -> f64 {
        let (q, l) = self.fold(p, l);
        let mut d = self.content(q, &self.cell_params(&l)).0;
        for (i, f) in self.shape.faces.iter().enumerate() {
            if self.g().asf(-dot4(*f, q)) < 0.5 * self.shape.rho {
                let r = self.g().reflect(q, *f);
                let ln = self.reflect_label(l, i);
                d = d.min(self.content(r, &self.cell_params(&ln)).0);
            }
        }
        d
    }

    /// March a geodesic; distance to the first wall closer than `margin`, or
    /// `None` if `max_t` is reached first.
    pub fn march(&self, p: V4, v: V4, max_t: f64, margin: f64, l: Label) -> Option<f64> {
        let mut t = 0.0;
        let (mut q, mut w) = (p, v);
        for _ in 0..600 {
            let d = self.sdf(q, l);
            if d < margin {
                return Some(t);
            }
            let step = (d - margin * 0.5).max(0.004 * self.shape.rho);
            if t + step >= max_t {
                return None;
            }
            t += step;
            let (nq, nw) = self.g().geodesic(q, w, step);
            q = nq;
            w = nw;
        }
        Some(t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_shapes() -> Vec<Shape> {
        let mut v = Vec::new();
        for s in [Space::Hyperbolic, Space::Euclidean, Space::Spherical] {
            for o in catalog(s) {
                v.push(Shape::build(s, o.kind, 1.5));
            }
        }
        v
    }

    #[test]
    fn cells_have_expected_combinatorics() {
        let expect = |s: &Shape| (s.faces.len(), s.edges.len(), s.verts.len(), s.cell_count);
        let shapes = all_shapes();
        assert_eq!(expect(&shapes[0]), (12, 30, 20, None));
        assert_eq!(expect(&shapes[3]), (18, 48, 32, None));
        assert_eq!(expect(&shapes[4]), (6, 12, 8, None));
        let counts: Vec<_> = shapes[5..].iter().map(|s| s.cell_count.unwrap()).collect();
        assert_eq!(counts, vec![5, 8, 16, 24, 120, 600]);
    }

    #[test]
    fn lobell_is_right_angled() {
        let s = Shape::build(Space::Hyperbolic, TilingKind::Lobell(7), 1.0);
        for &(_, _, gr) in &s.edges {
            assert!(gr.abs() < 1e-9);
        }
    }

    #[test]
    fn shared_faces_agree() {
        for shape in all_shapes() {
            let w = World::new(shape);
            // Random-ish walk; at every step both sides of every face agree.
            let mut l = w.initial_label();
            for step in 0..60 {
                let nf = w.shape.faces.len();
                for i in 0..nf {
                    let n = w.reflect_label(l, i);
                    assert_eq!(w.face_key(&l, i), w.face_key(&n, i));
                }
                l = w.reflect_label(l, (step * 7 + 3) % nf);
            }
        }
    }

    #[test]
    fn labels_are_path_independent() {
        for shape in all_shapes() {
            let w = World::new(shape);
            let g = *w.g();
            let mut f = Frame::identity();
            f.rotate(0, 2, 0.37);
            f.rotate(1, 2, 0.21);
            f.translate(&g, [0.0, 0.0, 1.0], 2.3 * w.shape.rho);
            let p = f.pos();
            let l0 = w.initial_label();
            let (q1, l1) = w.fold(p, l0);
            for i in 0..w.shape.faces.len() {
                let (q2, l2) =
                    w.fold(g.reflect(p, w.shape.faces[i]), w.reflect_label(l0, i));
                assert_eq!(w.cell_key(&l1), w.cell_key(&l2));
                // Same cell; the local point may differ by a cell symmetry.
                assert!((g.dist(q1, ORIGIN) - g.dist(q2, ORIGIN)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn hash_in_range() {
        for k in 0..20000 {
            let h = hash4(chunks(k as f64 * 211.0), 1.0);
            assert!((0.0..1.0).contains(&h));
            let h = hash4([-(k as f64), k as f64 * 3.0, 7.0, -2.0], 5.0);
            assert!((0.0..1.0).contains(&h));
        }
    }
}
