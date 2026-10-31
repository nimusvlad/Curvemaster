//! Constant-curvature 3-space in a unified 4D model.
//!
//! The curvature sign `k` selects the geometry:
//! * `k = -1` hyperbolic H3: points on `x²+y²+z²−w² = −1` (hyperboloid model),
//! * `k =  0` Euclidean E3: points `(x, y, z, 1)` (affine model),
//! * `k = +1` spherical S3: points on `x²+y²+z²+w² = 1`.
//!
//! All internal lengths are in units of the curvature radius (E3 uses the raw
//! world scale). Tangent vectors are measured with
//! `ip(a,b) = a.xyz·b.xyz + k·a.w·b.w`, and geodesics are
//! `p(t) = C(t)·p + S(t)·v` with `(C,S) = (cosh,sinh) | (1,t) | (cos,sin)`.
//!
//! Planes are stored as covectors `φ` (so `φ·p` uses the plain 4D dot product)
//! normalised with `ip(φ,φ) = 1`; then `φ·p` is `sinh d | d | sin d` of the
//! signed distance to the plane, positive on the "outside".

pub type V4 = [f64; 4];

pub const ORIGIN: V4 = [0.0, 0.0, 0.0, 1.0];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    Hyperbolic,
    Euclidean,
    Spherical,
}

impl Space {
    pub fn k(self) -> f64 {
        match self {
            Space::Hyperbolic => -1.0,
            Space::Euclidean => 0.0,
            Space::Spherical => 1.0,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Space::Hyperbolic => "Hyperbolic H3",
            Space::Euclidean => "Euclidean E3",
            Space::Spherical => "Spherical S3",
        }
    }
}

#[inline]
pub fn dot4(a: V4, b: V4) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
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

/// Vector orthogonal (in plain R^4) to three given vectors.
pub fn cross4(a: V4, b: V4, c: V4) -> V4 {
    let det3 = |m: [[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let mut r = [0.0; 4];
    for (i, ri) in r.iter_mut().enumerate() {
        let cols: Vec<usize> = (0..4).filter(|&j| j != i).collect();
        let m = [
            [a[cols[0]], a[cols[1]], a[cols[2]]],
            [b[cols[0]], b[cols[1]], b[cols[2]]],
            [c[cols[0]], c[cols[1]], c[cols[2]]],
        ];
        let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
        *ri = sign * det3(m);
    }
    r
}

#[derive(Clone, Copy, Debug)]
pub struct Geom {
    pub space: Space,
    pub k: f64,
}

impl Geom {
    pub fn new(space: Space) -> Self {
        Geom { space, k: space.k() }
    }

    /// Metric on tangent vectors / dual metric on plane covectors.
    #[inline]
    pub fn ip(&self, a: V4, b: V4) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + self.k * a[3] * b[3]
    }

    #[inline]
    pub fn cf(&self, t: f64) -> f64 {
        match self.space {
            Space::Hyperbolic => t.cosh(),
            Space::Euclidean => 1.0,
            Space::Spherical => t.cos(),
        }
    }

    #[inline]
    pub fn sf(&self, t: f64) -> f64 {
        match self.space {
            Space::Hyperbolic => t.sinh(),
            Space::Euclidean => t,
            Space::Spherical => t.sin(),
        }
    }

    /// Inverse of `sf`: converts a plane value `φ·p` into a signed distance.
    #[inline]
    pub fn asf(&self, x: f64) -> f64 {
        match self.space {
            Space::Hyperbolic => x.asinh(),
            Space::Euclidean => x,
            Space::Spherical => x.clamp(-1.0, 1.0).asin(),
        }
    }

    pub fn norm_point(&self, p: V4) -> V4 {
        match self.space {
            Space::Euclidean => scale(p, 1.0 / p[3]),
            Space::Hyperbolic => {
                let n = (-self.ip(p, p)).max(1e-300).sqrt();
                scale(p, if p[3] < 0.0 { -1.0 / n } else { 1.0 / n })
            }
            Space::Spherical => scale(p, 1.0 / dot4(p, p).max(1e-300).sqrt()),
        }
    }

    /// Project a vector onto the tangent space at `p`.
    #[inline]
    pub fn proj(&self, v: V4, p: V4) -> V4 {
        match self.space {
            Space::Euclidean => lin(v, 1.0, p, -v[3]),
            _ => lin(v, 1.0, p, -self.k * self.ip(v, p)),
        }
    }

    pub fn norm_tangent(&self, p: V4, v: V4) -> V4 {
        let v = self.proj(v, p);
        scale(v, 1.0 / self.ip(v, v).max(1e-300).sqrt())
    }

    pub fn dist(&self, a: V4, b: V4) -> f64 {
        match self.space {
            Space::Hyperbolic => (-self.ip(a, b)).max(1.0).acosh(),
            Space::Euclidean => {
                let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2], 0.0];
                dot4(d, d).sqrt()
            }
            Space::Spherical => dot4(a, b).clamp(-1.0, 1.0).acos(),
        }
    }

    /// Unit tangent at `from` along the (shortest) geodesic towards `to`.
    pub fn dir_to(&self, from: V4, to: V4) -> V4 {
        self.norm_tangent(from, to)
    }

    /// Follow a geodesic for distance `t`; returns new point and tangent.
    pub fn geodesic(&self, p: V4, v: V4, t: f64) -> (V4, V4) {
        let (c, s) = (self.cf(t), self.sf(t));
        let np = self.norm_point(lin(p, c, v, s));
        let nv = lin(p, -self.k * s, v, c);
        (np, self.norm_tangent(np, nv))
    }

    /// The vector a reflection moves points along (metric dual of `φ`).
    #[inline]
    pub fn refl_vec(&self, phi: V4) -> V4 {
        [phi[0], phi[1], phi[2], self.k * phi[3]]
    }

    /// Reflect a point or vector across the plane `φ`.
    #[inline]
    pub fn reflect(&self, v: V4, phi: V4) -> V4 {
        lin(v, 1.0, self.refl_vec(phi), -2.0 * dot4(phi, v))
    }
}

/// A 4x4 matrix stored as columns.
#[derive(Clone, Copy, Debug)]
pub struct M4 {
    pub c: [V4; 4],
}

impl M4 {
    pub fn identity() -> Self {
        M4 {
            c: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        }
    }

    pub fn apply(&self, v: V4) -> V4 {
        let mut r = [0.0; 4];
        for k in 0..4 {
            r = lin(r, 1.0, self.c[k], v[k]);
        }
        r
    }

    /// `self * Reflection(φ)`, a rank-one update.
    pub fn mul_reflect(&self, g: &Geom, phi: V4) -> M4 {
        let a = self.apply(g.refl_vec(phi));
        let mut m = *self;
        for k in 0..4 {
            m.c[k] = lin(m.c[k], 1.0, a, -2.0 * phi[k]);
        }
        m
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

/// An orthonormal frame: `c[0]` right, `c[1]` up, `c[2]` forward, `c[3]` position.
#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub c: [V4; 4],
}

impl Frame {
    #[allow(dead_code)]
    pub fn identity() -> Self {
        Frame { c: M4::identity().c }
    }

    pub fn pos(&self) -> V4 {
        self.c[3]
    }

    pub fn local_dir(&self, d: [f64; 3]) -> V4 {
        let mut r = [0.0; 4];
        for k in 0..3 {
            r = lin(r, 1.0, self.c[k], d[k]);
        }
        r
    }

    /// Rotate local axis `a` towards local axis `b` by `angle`.
    pub fn rotate(&mut self, a: usize, b: usize, angle: f64) {
        let (c, s) = (angle.cos(), angle.sin());
        let (ca, cb) = (self.c[a], self.c[b]);
        self.c[a] = lin(ca, c, cb, s);
        self.c[b] = lin(cb, c, ca, -s);
    }

    /// Move along the local unit direction `dir` by distance `t`,
    /// parallel-transporting the axes.
    pub fn translate(&mut self, g: &Geom, dir: [f64; 3], t: f64) {
        if t.abs() < 1e-15 {
            return;
        }
        let u = self.local_dir(dir);
        let p = self.c[3];
        let (c, s) = (g.cf(t), g.sf(t));
        let nu = lin(p, -g.k * s, u, c);
        for k in 0..3 {
            let ortho = lin(self.c[k], 1.0, u, -dir[k]);
            self.c[k] = lin(ortho, 1.0, nu, dir[k]);
        }
        self.c[3] = lin(p, c, u, s);
        self.orthonormalize(g);
    }

    pub fn map(&mut self, f: impl Fn(V4) -> V4) {
        for k in 0..4 {
            self.c[k] = f(self.c[k]);
        }
    }

    pub fn orthonormalize(&mut self, g: &Geom) {
        let p = g.norm_point(self.c[3]);
        self.c[3] = p;
        for k in 0..3 {
            let mut v = g.proj(self.c[k], p);
            for j in 0..k {
                v = lin(v, 1.0, self.c[j], -g.ip(v, self.c[j]));
            }
            self.c[k] = scale(v, 1.0 / g.ip(v, v).max(1e-300).sqrt());
        }
    }

    pub fn to_f32_cols(&self) -> [[f32; 4]; 4] {
        M4 { c: self.c }.to_f32_cols()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Space; 3] = [Space::Hyperbolic, Space::Euclidean, Space::Spherical];

    #[test]
    fn translate_moves_by_distance() {
        for s in ALL {
            let g = Geom::new(s);
            let mut f = Frame::identity();
            f.translate(&g, [0.0, 0.0, 1.0], 0.9);
            assert!((g.dist(f.pos(), ORIGIN) - 0.9).abs() < 1e-9, "{s:?}");
            f.rotate(0, 2, 0.7);
            f.translate(&g, [0.6, 0.0, 0.8], 0.5);
            for k in 0..3 {
                assert!(g.ip(f.c[k], g.proj(f.c[k], f.pos())).abs() > 0.99);
                assert!((g.ip(f.c[k], f.c[k]) - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn geodesic_round_trip() {
        for s in ALL {
            let g = Geom::new(s);
            let v = [1.0, 0.0, 0.0, 0.0];
            let (q, w) = g.geodesic(ORIGIN, v, 1.3);
            assert!((g.dist(ORIGIN, q) - 1.3).abs() < 1e-9, "{s:?}");
            let (r, _) = g.geodesic(q, scale(w, -1.0), 1.3);
            assert!(g.dist(ORIGIN, r) < 1e-9, "{s:?}");
        }
    }

    #[test]
    fn reflection_is_isometry() {
        for s in ALL {
            let g = Geom::new(s);
            let phi = [0.6, 0.0, 0.8, -0.3];
            let n = g.ip(phi, phi).sqrt();
            let phi = scale(phi, 1.0 / n);
            let (a, _) = g.geodesic(ORIGIN, [0.0, 1.0, 0.0, 0.0], 0.7);
            let (b, _) = g.geodesic(ORIGIN, [1.0, 0.0, 0.0, 0.0], 0.4);
            let d0 = g.dist(a, b);
            let d1 = g.dist(g.reflect(a, phi), g.reflect(b, phi));
            assert!((d0 - d1).abs() < 1e-9, "{s:?}");
        }
    }
}
