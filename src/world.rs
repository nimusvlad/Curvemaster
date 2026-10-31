//! The infinite procedural world.
//!
//! H3 is tiled by the right-angled dodecahedral honeycomb {5,3,4}: regular
//! dodecahedra with 90° dihedral angles, 8 around every vertex. Because every
//! dihedral angle is π/2, the 12 face reflections of one dodecahedron generate
//! a right-angled Coxeter group for which the dodecahedron is a *strict*
//! fundamental domain. Folding any point into the central cell by repeated
//! face reflections therefore identifies its cell uniquely, and the parity
//! vector of the reflections used (a 12-bit "label", the image of the cell's
//! group element in the abelianisation (Z/2)^12) is a well-defined procedural
//! seed for that cell.
//!
//! Every cell is decorated with:
//!  * beams along all edges and nodes at all vertices (shared, uniform),
//!  * a plate on each face: open, solid, or pierced by a round hole. The face
//!    is seeded from the label with the face's own bit cleared, so both cells
//!    sharing a face agree on it,
//!  * optionally a glowing orb at the cell centre.
//!
//! The same SDF is evaluated on the CPU (f64, for physics) and on the GPU
//! (f32, generated GLSL in [`crate::shader`]). The hash uses only exact
//! small-integer float arithmetic so both sides agree bit-for-bit.

use crate::hyper::*;

pub const NFACES: usize = 12;
pub const BEAM_R: f64 = 0.055;
pub const NODE_R: f64 = 0.11;
pub const PLATE_TH: f64 = 0.03;
/// Extra step allowed past the cell boundary. Content of neighbouring cells
/// that lies within this distance of a shared face is covered by beams.
pub const BOUND_SLACK: f64 = 0.02;

pub const MAT_FRAME: u8 = 0;
pub const MAT_PLATE: u8 = 1;
pub const MAT_ORB: u8 = 2;

pub struct Honeycomb {
    /// Unit spacelike normals of the 12 face planes (outward: `<n,p> > 0` outside).
    pub normals: [V4; NFACES],
    /// Euclidean unit directions of face centres (for hole axes).
    pub dirs: [[f64; 3]; NFACES],
    /// Adjacent face pairs (the 30 edges).
    pub edges: Vec<(usize, usize)>,
    /// Mutually adjacent face triples (the 20 vertices).
    pub verts: Vec<(usize, usize, usize)>,
}

#[derive(Clone, Copy, Debug)]
pub struct CellParams {
    /// Per face: -1 = open, 0 = solid, >0 = hole radius.
    pub face: [f64; NFACES],
    /// Orb radius (0 = none).
    pub orb: f64,
}

/// Float modulo that is robust to division rounding (x, m are integers).
#[inline]
pub fn imod(x: f64, m: f64) -> f64 {
    x - m * ((x + 0.5) / m).floor()
}

/// Deterministic hash of an integer key (< 65536) and salt, in [0,1).
/// All intermediate values are integers below 2^21, exact in f32 and f64.
pub fn hash(k: f64, salt: f64) -> f64 {
    let hi = (k / 64.0).floor();
    let lo = k - hi * 64.0;
    let a = imod(hi * 67.0 + lo * 13.0 + salt * 29.0 + 5.0, 1009.0);
    let b = imod(a * a + lo * 7.0 + 11.0, 1013.0);
    let c = imod(b * (a + 3.0) + salt + 1.0, 1019.0);
    c / 1019.0
}

#[inline]
pub fn bit(lab: f64, i: usize) -> f64 {
    let p = (1u32 << i) as f64;
    imod((lab / p).floor(), 2.0)
}

#[inline]
pub fn toggle(lab: f64, i: usize) -> f64 {
    let p = (1u32 << i) as f64;
    if bit(lab, i) > 0.5 {
        lab - p
    } else {
        lab + p
    }
}

/// Face plate parameter for face `i` of the cell with label `lab`.
pub fn face_param(lab: f64, i: usize) -> f64 {
    let cleared = lab - bit(lab, i) * (1u32 << i) as f64;
    let key = cleared * 12.0 + i as f64;
    let r = hash(key, 1.0);
    if r < 0.42 {
        -1.0
    } else if r < 0.80 {
        0.2 + 0.22 * hash(key, 2.0)
    } else {
        0.0
    }
}

pub fn orb_param(lab: f64) -> f64 {
    if hash(lab, 3.0) < 0.33 {
        0.14 + 0.2 * hash(lab, 4.0)
    } else {
        0.0
    }
}

pub fn cell_params(lab: f64) -> CellParams {
    let mut face = [0.0; NFACES];
    for (i, f) in face.iter_mut().enumerate() {
        *f = face_param(lab, i);
    }
    CellParams { face, orb: orb_param(lab) }
}

impl Honeycomb {
    pub fn new() -> Self {
        let phi = (1.0 + 5f64.sqrt()) / 2.0;
        let mut dirs = Vec::new();
        for &a in &[1.0, -1.0] {
            for &b in &[1.0, -1.0] {
                dirs.push([0.0, a, b * phi]);
                dirs.push([a, b * phi, 0.0]);
                dirs.push([b * phi, 0.0, a]);
            }
        }
        let dirs: Vec<[f64; 3]> = dirs
            .into_iter()
            .map(|d| {
                let l = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                [d[0] / l, d[1] / l, d[2] / l]
            })
            .collect();
        // Right angles: s^2 = cos(angle between adjacent face normals) = 1/sqrt(5).
        let s = (1.0 / 5f64.sqrt()).sqrt();
        let k = (1.0 - s * s).sqrt();
        let mut normals = [[0.0; 4]; NFACES];
        let mut dd = [[0.0; 3]; NFACES];
        for i in 0..NFACES {
            let d = dirs[i];
            normals[i] = [d[0] / k, d[1] / k, d[2] / k, s / k];
            dd[i] = d;
        }
        let dot3 = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let adj = |i: usize, j: usize| (dot3(dd[i], dd[j]) - 1.0 / 5f64.sqrt()).abs() < 1e-6;
        let mut edges = Vec::new();
        let mut verts = Vec::new();
        for i in 0..NFACES {
            for j in i + 1..NFACES {
                if adj(i, j) {
                    edges.push((i, j));
                    for l in j + 1..NFACES {
                        if adj(i, l) && adj(j, l) {
                            verts.push((i, j, l));
                        }
                    }
                }
            }
        }
        assert_eq!(edges.len(), 30);
        assert_eq!(verts.len(), 20);
        Honeycomb { normals, dirs: dd, edges, verts }
    }

    /// Index of a face the point lies outside of, if any.
    pub fn outside_face(&self, p: V4) -> Option<usize> {
        let mut best = None;
        let mut bv = 1e-12;
        for i in 0..NFACES {
            let s = mdot(self.normals[i], p);
            if s > bv {
                bv = s;
                best = Some(i);
            }
        }
        best
    }

    /// Fold a point into the central cell. Returns the folded point and the
    /// label of the cell that contained it.
    pub fn fold(&self, mut p: V4, mut lab: f64) -> (V4, f64) {
        for _ in 0..256 {
            match self.outside_face(p) {
                Some(i) => {
                    p = reflect(p, self.normals[i]);
                    lab = toggle(lab, i);
                }
                None => break,
            }
        }
        (p, lab)
    }

    /// Unclamped SDF of the central cell's own content, its material, and the
    /// largest face value `max_i <n_i,p>` (negative inside the cell).
    pub fn content(&self, p: V4, cp: &CellParams) -> (f64, u8, f64) {
        let mut s = [0.0; NFACES];
        let mut smax = f64::NEG_INFINITY;
        for i in 0..NFACES {
            s[i] = mdot(self.normals[i], p);
            smax = smax.max(s[i]);
        }
        // Frame: edges and vertices. For mutually orthogonal planes the
        // distance to their intersection satisfies sinh^2 d = sum s_i^2.
        let mut e2 = f64::INFINITY;
        for &(i, j) in &self.edges {
            e2 = e2.min(s[i] * s[i] + s[j] * s[j]);
        }
        let mut v2 = f64::INFINITY;
        for &(i, j, l) in &self.verts {
            v2 = v2.min(s[i] * s[i] + s[j] * s[j] + s[l] * s[l]);
        }
        let mut d = (e2.sqrt().asinh() - BEAM_R).min(v2.sqrt().asinh() - NODE_R);
        let mut mat = MAT_FRAME;
        // Plates.
        for i in 0..NFACES {
            let f = cp.face[i];
            if f < -0.5 {
                continue;
            }
            let mut slab = s[i].asinh().abs() - PLATE_TH;
            if f > 0.0 {
                // Distance to the geodesic through the cell centre and the
                // face centre: cosh^2 d = w^2 - (u.x)^2.
                let u = self.dirs[i];
                let ux = u[0] * p[0] + u[1] * p[1] + u[2] * p[2];
                let c = p[3] * p[3] - ux * ux;
                slab = slab.max(f - c.max(1.0).sqrt().acosh());
            }
            if slab < d {
                d = slab;
                mat = MAT_PLATE;
            }
        }
        // Orb.
        if cp.orb > 0.0 {
            let o = p[3].max(1.0).acosh() - cp.orb;
            if o < d {
                d = o;
                mat = MAT_ORB;
            }
        }
        (d, mat, smax)
    }

    /// Raymarching-safe SDF for a point (roughly) inside the central cell:
    /// never steps further than just past the cell boundary, since the
    /// neighbouring cells may hold different content.
    pub fn sdf_local(&self, p: V4, cp: &CellParams) -> (f64, u8) {
        let (d, mat, smax) = self.content(p, cp);
        (d.min((-smax).asinh() + BOUND_SLACK), mat)
    }

    /// Conservative world SDF at a point in world coordinates, where `label`
    /// is the label of the central cell.
    pub fn sdf(&self, p: V4, label: f64) -> f64 {
        let (q, lab) = self.fold(p, label);
        self.sdf_local(q, &cell_params(lab)).0
    }

    /// Accurate distance to the nearest wall, used for physics: considers the
    /// containing cell and the neighbours across any nearby face.
    pub fn wall_dist(&self, p: V4, label: f64) -> f64 {
        let (q, lab) = self.fold(p, label);
        let mut d = self.content(q, &cell_params(lab)).0;
        for i in 0..NFACES {
            if (-mdot(self.normals[i], q)).asinh() < 0.4 {
                let r = reflect(q, self.normals[i]);
                d = d.min(self.content(r, &cell_params(toggle(lab, i))).0);
            }
        }
        d
    }

    /// March along a geodesic; returns the distance travelled before hitting
    /// a wall (closer than `margin`), or `None` if `max_t` is reached first.
    pub fn march(&self, p: V4, v: V4, max_t: f64, margin: f64, label: f64) -> Option<f64> {
        let mut t = 0.0;
        let (mut q, mut w) = (p, v);
        for _ in 0..400 {
            let d = self.sdf(q, label);
            if d < margin {
                return Some(t);
            }
            let step = (d - margin * 0.5).max(0.005);
            if t + step >= max_t {
                return None;
            }
            t += step;
            let (nq, nw) = geodesic(q, w, step);
            q = nq;
            w = nw;
        }
        Some(t)
    }
}

impl Default for Honeycomb {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn faces_are_right_angled() {
        let h = Honeycomb::new();
        for &(i, j) in &h.edges {
            assert!(mdot(h.normals[i], h.normals[j]).abs() < 1e-9);
        }
        for i in 0..NFACES {
            assert!((mdot(h.normals[i], h.normals[i]) - 1.0).abs() < 1e-9);
            assert!(mdot(h.normals[i], ORIGIN) < 0.0);
        }
    }

    #[test]
    fn labels_are_path_independent() {
        // Folding a far point gives the same cell no matter the start label
        // offset convention: fold(reflect_i(p)) == fold(p) with label^bit_i.
        let h = Honeycomb::new();
        let mut f = Frame::identity();
        f.rotate(0, 2, 0.37);
        f.rotate(1, 2, 0.21);
        f.translate([0.0, 0.0, 1.0], 3.7);
        let p = f.pos();
        let (q1, l1) = h.fold(p, 0.0);
        for i in 0..NFACES {
            let (q2, l2) = h.fold(reflect(p, h.normals[i]), toggle(0.0, i));
            assert_eq!(l1, l2);
            assert!(dist(q1, q2) < 1e-6);
        }
    }

    #[test]
    fn shared_faces_agree() {
        for lab in 0..4096 {
            let lab = lab as f64;
            for i in 0..NFACES {
                assert_eq!(face_param(lab, i), face_param(toggle(lab, i), i));
            }
        }
    }

    #[test]
    fn hash_in_range() {
        for k in 0..60000 {
            let h = hash(k as f64, 1.0);
            assert!((0.0..1.0).contains(&h));
        }
    }
}
