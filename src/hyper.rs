//! Hyperboloid-model math for H3.
//!
//! Points of H3 are 4-vectors `p = (x, y, z, w)` with `<p,p> = -1` and `w > 0`,
//! where `<a,b> = a.x*b.x + a.y*b.y + a.z*b.z - a.w*b.w` is the Minkowski form.
//! Tangent vectors `v` at `p` satisfy `<v,p> = 0`. Isometries are 4x4 Lorentz
//! matrices. The origin is `(0, 0, 0, 1)`.

pub type V4 = [f64; 4];

#[allow(dead_code)]
pub const ORIGIN: V4 = [0.0, 0.0, 0.0, 1.0];

#[inline]
pub fn mdot(a: V4, b: V4) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] - a[3] * b[3]
}

#[inline]
pub fn add(a: V4, b: V4) -> V4 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

#[inline]
pub fn scale(a: V4, s: f64) -> V4 {
    [a[0] * s, a[1] * s, a[2] * s, a[3] * s]
}

/// `a*s + b*t`
#[inline]
pub fn lin(a: V4, s: f64, b: V4, t: f64) -> V4 {
    add(scale(a, s), scale(b, t))
}

/// Rescale a (roughly) timelike vector back onto the hyperboloid.
pub fn normalize_point(p: V4) -> V4 {
    let n = (-mdot(p, p)).max(1e-30).sqrt();
    let p = scale(p, 1.0 / n);
    if p[3] < 0.0 {
        scale(p, -1.0)
    } else {
        p
    }
}

/// Project `v` onto the tangent space at `p` and normalize it.
pub fn normalize_tangent(p: V4, v: V4) -> V4 {
    let v = lin(v, 1.0, p, mdot(v, p));
    let n = mdot(v, v).max(1e-30).sqrt();
    scale(v, 1.0 / n)
}

/// Hyperbolic distance between two points.
pub fn dist(a: V4, b: V4) -> f64 {
    (-mdot(a, b)).max(1.0).acosh()
}

/// Unit tangent at `from` pointing along the geodesic towards `to`.
pub fn dir_to(from: V4, to: V4) -> V4 {
    normalize_tangent(from, to)
}

/// Follow the geodesic from `p` with unit tangent `v` for distance `t`.
/// Returns the new point and the transported tangent.
pub fn geodesic(p: V4, v: V4, t: f64) -> (V4, V4) {
    let (c, s) = (t.cosh(), t.sinh());
    let np = lin(p, c, v, s);
    let nv = lin(p, s, v, c);
    let np = normalize_point(np);
    (np, normalize_tangent(np, nv))
}

/// Reflect a vector across the hyperplane with unit spacelike normal `n`.
#[inline]
pub fn reflect(v: V4, n: V4) -> V4 {
    lin(v, 1.0, n, -2.0 * mdot(v, n))
}

/// A frame (Lorentz matrix) stored as 4 columns:
/// `c[0]` = right, `c[1]` = up, `c[2]` = forward, `c[3]` = position.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub c: [V4; 4],
}

impl Frame {
    #[allow(dead_code)]
    pub fn identity() -> Self {
        Frame {
            c: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn pos(&self) -> V4 {
        self.c[3]
    }

    /// Map a local 4-vector into world coordinates.
    pub fn apply(&self, l: V4) -> V4 {
        let mut r = [0.0; 4];
        for k in 0..4 {
            r = lin(r, 1.0, self.c[k], l[k]);
        }
        r
    }

    /// World tangent vector corresponding to a local spatial direction.
    pub fn local_dir(&self, d: [f64; 3]) -> V4 {
        self.apply([d[0], d[1], d[2], 0.0])
    }

    /// Rotate the frame in the plane of local axes `a` and `b` by `angle`
    /// (turns axis `a` towards axis `b`).
    pub fn rotate(&mut self, a: usize, b: usize, angle: f64) {
        let (c, s) = (angle.cos(), angle.sin());
        let ca = self.c[a];
        let cb = self.c[b];
        self.c[a] = lin(ca, c, cb, s);
        self.c[b] = lin(cb, c, ca, -s);
    }

    /// Translate the frame along the local direction `dir` (unit length in R^3)
    /// by hyperbolic distance `t`. Spatial axes are parallel-transported.
    pub fn translate(&mut self, dir: [f64; 3], t: f64) {
        if t.abs() < 1e-12 {
            return;
        }
        let u = self.local_dir(dir);
        let p = self.c[3];
        let (ch, sh) = (t.cosh(), t.sinh());
        // Boost in the (u, p) plane; components orthogonal to u are untouched.
        for k in 0..3 {
            let a = dir[k];
            // c_k = a*u + (orthogonal part)
            let ortho = lin(self.c[k], 1.0, u, -a);
            let nu = lin(u, ch, p, sh);
            self.c[k] = lin(ortho, 1.0, nu, a);
        }
        self.c[3] = lin(p, ch, u, sh);
        self.orthonormalize();
    }

    /// Apply an isometry given as a function on vectors (e.g. a reflection).
    pub fn map(&mut self, f: impl Fn(V4) -> V4) {
        for k in 0..4 {
            self.c[k] = f(self.c[k]);
        }
    }

    /// Minkowski Gram-Schmidt to remove numerical drift.
    pub fn orthonormalize(&mut self) {
        let p = normalize_point(self.c[3]);
        self.c[3] = p;
        for k in 0..3 {
            let mut v = lin(self.c[k], 1.0, p, mdot(self.c[k], p));
            for j in 0..k {
                v = lin(v, 1.0, self.c[j], -mdot(v, self.c[j]));
            }
            let n = mdot(v, v).max(1e-30).sqrt();
            self.c[k] = scale(v, 1.0 / n);
        }
    }

    pub fn to_f32_cols(&self) -> [[f32; 4]; 4] {
        let mut m = [[0.0f32; 4]; 4];
        for k in 0..4 {
            for i in 0..4 {
                m[k][i] = self.c[k][i] as f32;
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn translate_moves_by_distance() {
        let mut f = Frame::identity();
        f.translate([0.0, 0.0, 1.0], 1.5);
        assert!((dist(f.pos(), ORIGIN) - 1.5).abs() < 1e-9);
        f.rotate(0, 2, 0.7);
        f.translate([0.0, 0.0, 1.0], 0.5);
        assert!((mdot(f.pos(), f.pos()) + 1.0).abs() < 1e-9);
        for k in 0..3 {
            assert!(mdot(f.c[k], f.pos()).abs() < 1e-9);
            assert!((mdot(f.c[k], f.c[k]) - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn geodesic_round_trip() {
        let p = ORIGIN;
        let v = [1.0, 0.0, 0.0, 0.0];
        let (q, w) = geodesic(p, v, 2.0);
        assert!((dist(p, q) - 2.0).abs() < 1e-9);
        let (r, _) = geodesic(q, scale(w, -1.0), 2.0);
        assert!(dist(p, r) < 1e-9);
    }
}
