//! GLSL generation. Geometry helpers, the cell's planes, edges and vertices
//! and the labelling scheme are emitted from the same Rust data the CPU uses,
//! as unrolled GLSL 1.00 (no constant arrays, no integer ops).

use crate::geom::*;
use crate::world::*;

pub const MAX_ENEMIES: usize = 24;
pub const MAX_PROJECTILES: usize = 32;

pub const VERTEX: &str = r#"#version 100
precision highp float;
attribute vec3 position;
attribute vec2 texcoord;
uniform mat4 Model;
uniform mat4 Projection;
void main() {
    gl_Position = Projection * Model * vec4(position, 1.0);
}
"#;

/// Rendering constants that depend on the world scale.
pub struct RenderParams {
    /// Max ray length (internal units).
    pub max_dist: f64,
    /// Fog density per internal unit.
    pub fog: f64,
    /// Glow falloff per internal unit.
    pub glow_k: f64,
}

fn f(x: f64) -> String {
    format!("{x:.12}")
}
fn v4(v: V4) -> String {
    format!("vec4({}, {}, {}, {})", f(v[0]), f(v[1]), f(v[2]), f(v[3]))
}
fn v3(v: [f64; 3]) -> String {
    format!("vec3({}, {}, {})", f(v[0]), f(v[1]), f(v[2]))
}

const H3: &str = r#"
const float KC = -1.0;
float mdot(vec4 a, vec4 b) { return dot(a.xyz, b.xyz) - a.w * b.w; }
float Cf(float t) { float e = exp(t); return 0.5 * (e + 1.0 / e); }
float Sf(float t) { float e = exp(t); return 0.5 * (e - 1.0 / e); }
float asf(float x) { return sign(x) * log(abs(x) + sqrt(x * x + 1.0)); }
float acoshF(float x) { x = max(x, 1.0); return log(x + sqrt(x * x - 1.0)); }
float vertDist(float m, vec4 p) { return acoshF(m); }
float orbDist(vec4 p) { return acoshF(p.w); }
float lineDist(vec4 p, vec3 u) { float ux = dot(u, p.xyz); return acoshF(sqrt(max(p.w * p.w - ux * ux, 1.0))); }
vec4 proj(vec4 v, vec4 p) { return v + mdot(v, p) * p; }
vec4 normPoint(vec4 p) { return p * inversesqrt(max(-mdot(p, p), 1e-20)); }
vec4 gradVec(vec4 g) { return vec4(g.xyz, -g.w); }
float hitBall(vec4 ro, vec4 rd, vec4 c, float r) {
    float A = -mdot(ro, c);
    float B = -mdot(rd, c);
    float C = Cf(r);
    float disc = C * C - A * A + B * B;
    if (disc < 0.0) return -1.0;
    float x = (C - sqrt(disc)) / (A + B);
    if (x <= 1.0) return -1.0;
    return log(x);
}
// x: parameter of closest approach, y: closest distance
vec2 closest(vec4 ro, vec4 rd, vec4 c) {
    float A = -mdot(ro, c);
    float B = -mdot(rd, c);
    if (B >= 0.0 || -B >= A) return vec2(-1.0, 0.0);
    return vec2(0.5 * log((A - B) / (A + B)), acoshF(sqrt(max(A * A - B * B, 1.0))));
}
"#;

const E3: &str = r#"
const float KC = 0.0;
float mdot(vec4 a, vec4 b) { return dot(a.xyz, b.xyz); }
float Cf(float t) { return 1.0; }
float Sf(float t) { return t; }
float asf(float x) { return x; }
float vertDist(float m, vec4 p) { return sqrt(max(m + dot(p.xyz, p.xyz), 0.0)); }
float orbDist(vec4 p) { return length(p.xyz); }
float lineDist(vec4 p, vec3 u) { float ux = dot(u, p.xyz); return sqrt(max(dot(p.xyz, p.xyz) - ux * ux, 0.0)); }
vec4 proj(vec4 v, vec4 p) { return v - v.w * p; }
vec4 normPoint(vec4 p) { return p / p.w; }
vec4 gradVec(vec4 g) { return vec4(g.xyz, 0.0); }
float hitBall(vec4 ro, vec4 rd, vec4 c, float r) {
    vec3 oc = ro.xyz - c.xyz;
    float b = dot(oc, rd.xyz);
    float h = b * b - dot(oc, oc) + r * r;
    if (h < 0.0) return -1.0;
    float t = -b - sqrt(h);
    return t > 0.0 ? t : -1.0;
}
vec2 closest(vec4 ro, vec4 rd, vec4 c) {
    vec3 oc = ro.xyz - c.xyz;
    float b = dot(oc, rd.xyz);
    if (b >= 0.0) return vec2(-1.0, 0.0);
    return vec2(-b, sqrt(max(dot(oc, oc) - b * b, 0.0)));
}
"#;

const S3: &str = r#"
const float KC = 1.0;
const float TAU = 6.283185307;
float mdot(vec4 a, vec4 b) { return dot(a, b); }
float Cf(float t) { return cos(t); }
float Sf(float t) { return sin(t); }
float asf(float x) { return asin(clamp(x, -1.0, 1.0)); }
float vertDist(float m, vec4 p) { return acos(clamp(-m, -1.0, 1.0)); }
float orbDist(vec4 p) { return acos(clamp(p.w, -1.0, 1.0)); }
float lineDist(vec4 p, vec3 u) { float ux = dot(u, p.xyz); return acos(sqrt(clamp(p.w * p.w + ux * ux, 0.0, 1.0))); }
vec4 proj(vec4 v, vec4 p) { return v - dot(v, p) * p; }
vec4 normPoint(vec4 p) { return normalize(p); }
vec4 gradVec(vec4 g) { return g; }
float hitBall(vec4 ro, vec4 rd, vec4 c, float r) {
    float A = dot(ro, c);
    float B = dot(rd, c);
    float C = cos(r);
    float R = sqrt(A * A + B * B);
    if (R < C || A > C) return -1.0;
    float t = atan(B, A) - acos(clamp(C / R, -1.0, 1.0));
    return mod(t, TAU);
}
vec2 closest(vec4 ro, vec4 rd, vec4 c) {
    float A = dot(ro, c);
    float B = dot(rd, c);
    return vec2(mod(atan(B, A), TAU), acos(clamp(sqrt(A * A + B * B), -1.0, 1.0)));
}
"#;

pub fn fragment(w: &World, rp: &RenderParams) -> String {
    let sh = &w.shape;
    let g = &sh.g;
    let nf = sh.faces.len();
    let parity = matches!(sh.mode, LabelMode::Parity);

    let mut consts = String::new();
    for i in 0..nf {
        consts += &format!("const vec4 P{i} = {};\n", v4(sh.faces[i]));
        consts += &format!("const vec4 R{i} = {};\n", v4(g.refl_vec(sh.faces[i])));
        consts += &format!("const vec3 U{i} = {};\n", v3(sh.dirs[i]));
        consts += &format!("const vec4 FC{i} = {};\n", v4(sh.fcent[i]));
    }
    // Vertex "closeness" vectors: min over j of dot(p, VP_j) gives the
    // nearest vertex (see vertDist).
    for (j, v) in sh.verts.iter().enumerate() {
        let vp = match g.space {
            Space::Hyperbolic => [-v[0], -v[1], -v[2], v[3]],
            Space::Spherical => scale(*v, -1.0),
            Space::Euclidean => [-2.0 * v[0], -2.0 * v[1], -2.0 * v[2], v[0] * v[0] + v[1] * v[1] + v[2] * v[2]],
        };
        consts += &format!("const vec4 VP{j} = {};\n", v4(vp));
    }
    let (qs, qo) = match sh.mode {
        LabelMode::Center { scale, off } => (scale, off),
        LabelMode::Parity => (1.0, [0.0; 4]),
    };
    consts += &format!("const float QS = {};\nconst vec4 QO = {};\n", f(qs), v4(qo));
    consts += &format!(
        "const float BEAM = {};\nconst float NODE = {};\nconst float PLATE = {};\nconst float SLACK = {};\nconst float RHO = {};\nconst float NFACES = {nf}.0;\n",
        f(w.beam), f(w.node), f(w.plate), f(w.slack), f(sh.rho)
    );
    consts += &format!(
        "const float MAX_DIST = {};\nconst float FOGK = {};\nconst float GLOWK = {};\n",
        f(rp.max_dist), f(rp.fog), f(rp.glow_k)
    );

    let mut globals = String::new();
    let mut cell = String::new();
    for i in 0..nf {
        globals += &format!("float gF{i}; float gH{i};\n");
        let q = if parity {
            format!("faceKeyP(lab, {i}.0, {}.0)", 1u32 << i)
        } else {
            format!("floor((gG * FC{i}) * QS + QO)")
        };
        cell += &format!(
            "    q = {q}; gF{i} = faceParam(q, {}, {}); gH{i} = hash4(q, 7.0);\n",
            f(sh.face_inr[i]),
            f(w.hole_max(i))
        );
    }
    let cell_q = if parity { "chunks(lab)" } else { "floor(gG[3] * QS + QO)" };

    let mut fold = String::new();
    for i in 0..nf {
        let upd = if parity {
            format!("lab = toggle(lab, {}.0);", 1u32 << i)
        } else {
            format!("vec4 a = gG * R{i}; gG[0] -= 2.0 * P{i}.x * a; gG[1] -= 2.0 * P{i}.y * a; gG[2] -= 2.0 * P{i}.z * a; gG[3] -= 2.0 * P{i}.w * a;")
        };
        fold += &format!(
            "        s = dot(P{i}, p); if (s > 0.0) {{ p -= 2.0 * s * R{i}; d -= 2.0 * dot(P{i}, d) * R{i}; {upd} changed = true; }}\n"
        );
    }

    let mut sdf = String::new();
    for i in 0..nf {
        sdf += &format!("    float s{i} = dot(P{i}, p);\n");
    }
    sdf += "    float smax = s0;\n";
    for i in 1..nf {
        sdf += &format!("    smax = max(smax, s{i});\n");
    }
    sdf += "    float e2 = 1e9;\n";
    for &(a, b, gr) in &sh.edges {
        if gr.abs() < 1e-12 {
            sdf += &format!("    e2 = min(e2, s{a}*s{a} + s{b}*s{b});\n");
        } else {
            sdf += &format!(
                "    e2 = min(e2, (s{a}*s{a} - {} * s{a}*s{b} + s{b}*s{b}) * {});\n",
                f(2.0 * gr),
                f(1.0 / (1.0 - gr * gr))
            );
        }
    }
    sdf += "    float vm = 1e9;\n";
    for j in 0..sh.verts.len() {
        sdf += &format!("    vm = min(vm, dot(p, VP{j}));\n");
    }
    sdf += "    float dd = min(asf(sqrt(max(e2, 0.0))) - BEAM, vertDist(vm, p) - NODE);\n    float mat = 0.0; float fid = -1.0;\n";
    for i in 0..nf {
        sdf += &format!(
            "    if (gF{i} > -0.5) {{ float sl = abs(asf(s{i})) - PLATE; if (sl < dd) {{ if (gF{i} > 0.0) sl = max(sl, gF{i} - lineDist(p, U{i})); if (sl < dd) {{ dd = sl; mat = 1.0; fid = {i}.0; }} }} }}\n"
        );
    }
    sdf += "    if (gOrb > 0.0) { float o = orbDist(p) - gOrb; if (o < dd) { dd = o; mat = 2.0; } }\n";
    sdf += "    return vec4(dd, mat, fid, asf(-smax) + SLACK);\n";

    let mut face_sel = String::new();
    for i in 0..nf {
        face_sel += &format!("    if (fid < {i}.5) {{ u = U{i}; return gH{i}; }}\n");
    }

    let geo = match g.space {
        Space::Hyperbolic => H3,
        Space::Euclidean => E3,
        Space::Spherical => S3,
    };

    TEMPLATE
        .replace("//GEOMETRY", geo)
        .replace("//CONSTS", &consts)
        .replace("//GLOBALS", &globals)
        .replace("//CELLQ", cell_q)
        .replace("//CELL", &cell)
        .replace("//FOLD", &fold)
        .replace("//SDF", &sdf)
        .replace("//FACESEL", &face_sel)
        .replace("MAXE", &MAX_ENEMIES.to_string())
        .replace("MAXP", &MAX_PROJECTILES.to_string())
}

const TEMPLATE: &str = r#"#version 100
precision highp float;

uniform vec2 uRes;
uniform mat4 uFrame;
uniform float uLabel;
uniform mat4 uG;
uniform float uTime;
uniform float uFov;
uniform vec4 uEnemy[MAXE];
uniform vec4 uEnemyInfo[MAXE];
uniform float uNumEnemies;
uniform vec4 uProj[MAXP];
uniform float uNumProj;
uniform float uProjR;

//GEOMETRY

//CONSTS

// Exact small-integer hashing, identical to world.rs.
float imod(float x, float m) { return x - m * floor((x + 0.5) / m); }
float hash4(vec4 q, float salt) {
    float a = imod(imod(q.x, 1009.0) * 37.0 + salt * 101.0 + 7.0, 1009.0);
    a = imod(a * a + imod(q.y, 1009.0) * 13.0 + 11.0, 1013.0);
    a = imod(a * (imod(q.z, 1009.0) + 3.0) + 5.0, 1019.0);
    a = imod(a * (imod(q.w, 1009.0) + 7.0) + salt + 1.0, 1021.0);
    a = imod(a * a + 3.0, 1031.0);
    return a / 1031.0;
}
vec4 chunks(float k) {
    float c2 = floor(k / 65536.0);
    float r = k - c2 * 65536.0;
    float c1 = floor(r / 256.0);
    return vec4(r - c1 * 256.0, c1, c2, 0.0);
}
float bitOf(float lab, float p2) { return imod(floor(lab / p2), 2.0); }
float toggle(float lab, float p2) { return bitOf(lab, p2) > 0.5 ? lab - p2 : lab + p2; }
vec4 faceKeyP(float lab, float i, float p2) { return chunks((lab - bitOf(lab, p2) * p2) * NFACES + i); }
float faceParam(vec4 q, float inr, float hmax) {
    float r = hash4(q, 1.0);
    if (r < 0.42) return -1.0;
    if (r < 0.80) return min((0.3 + 0.35 * hash4(q, 2.0)) * inr, hmax);
    return 0.0;
}

//GLOBALS
float gOrb;
float gC1;
float gC2;
mat4 gG;

void cellParams(float lab) {
    vec4 q;
//CELL
    vec4 qc = //CELLQ;
    gOrb = hash4(qc, 3.0) < 0.33 ? (0.17 + 0.25 * hash4(qc, 4.0)) * RHO : 0.0;
    gC1 = hash4(qc, 5.0);
    gC2 = hash4(qc, 6.0);
}

// x: content distance, y: material, z: face id, w: distance bound to cell exit
vec4 sdfLocal(vec4 p) {
//SDF
}

float faceSel(float fid, out vec3 u) {
//FACESEL
    u = vec3(0.0, 0.0, 1.0);
    return 0.0;
}

vec3 hsv(float h, float s, float v) {
    vec3 k = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return v * mix(vec3(1.0), k, s);
}

float glowBall(vec4 ro, vec4 rd, vec4 c, float r, float tmax, float k) {
    vec2 cl = closest(ro, rd, c);
    if (cl.x < 0.0 || cl.x > tmax) return 0.0;
    // GLOWK converts internal to world units; fade with distance so a row of
    // shots along the view axis doesn't saturate.
    float tw = cl.x * GLOWK;
    return exp(-k * GLOWK * max(cl.y - r, 0.0)) / (1.0 + 3.0 * tw * tw);
}

const int MAX_STEPS = 180;
const vec3 FOG = vec3(0.025, 0.018, 0.055);

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;
    vec3 ld = normalize(vec3(uv * uFov, 1.0));
    vec4 ro = uFrame[3];
    vec4 rd = uFrame * vec4(ld, 0.0);
    float pix = uFov / uRes.y;

    // March inside the central cell, reflecting the ray whenever it leaves.
    vec4 p = ro;
    vec4 d = rd;
    float lab = uLabel;
    gG = uG;
    cellParams(lab);
    float t = 0.0;
    bool hit = false;
    vec4 res = vec4(0.0);
    float steps = 0.0;
    for (int i = 0; i < MAX_STEPS; i++) {
        bool changed = false;
        float s;
        for (int k = 0; k < 2; k++) {
//FOLD
        }
        if (changed) cellParams(lab);
        res = sdfLocal(p);
        float h = min(res.x, res.w);
        float eps = max(0.0007 * RHO, 0.7 * pix * abs(Sf(t)));
        if (res.x < eps) { hit = true; break; }
        t += h;
        steps += 1.0;
        if (t > MAX_DIST) break;
        float c = Cf(h);
        float sn = Sf(h);
        vec4 np = c * p + sn * d;
        d = -KC * sn * p + c * d;
        p = normPoint(np);
        d = proj(d, p);
        d *= inversesqrt(mdot(d, d));
    }

    vec3 col = FOG;
    float tHit = hit ? t : 1e9;

    if (hit) {
        float e = 0.002 * RHO;
        vec4 gr = vec4(
            sdfLocal(p + vec4(e, 0.0, 0.0, 0.0)).x - sdfLocal(p - vec4(e, 0.0, 0.0, 0.0)).x,
            sdfLocal(p + vec4(0.0, e, 0.0, 0.0)).x - sdfLocal(p - vec4(0.0, e, 0.0, 0.0)).x,
            sdfLocal(p + vec4(0.0, 0.0, e, 0.0)).x - sdfLocal(p - vec4(0.0, 0.0, e, 0.0)).x,
            sdfLocal(p + vec4(0.0, 0.0, 0.0, e)).x - sdfLocal(p - vec4(0.0, 0.0, 0.0, e)).x);
        vec4 n = proj(gradVec(gr), p);
        n *= inversesqrt(max(mdot(n, n), 1e-20));
        vec4 v = -d;
        if (mdot(n, v) < 0.0) n = -n;

        // A lamp at the centre of each cell, tinted with the cell's colour.
        vec3 lampCol = hsv(gC1, 0.55 + 0.35 * gC2, 1.0);
        float dl = orbDist(p) / RHO;
        vec4 L = proj(vec4(0.0, 0.0, 0.0, 1.0), p);
        L *= inversesqrt(max(mdot(L, L), 1e-20));
        float lamp = max(mdot(n, L), 0.0) / (1.0 + 1.6 * dl * dl);

        float head = max(mdot(n, v), 0.0);
        float ao = clamp(1.0 - steps / 100.0, 0.35, 1.0);

        vec3 base;
        float spec = 0.0;
        vec3 emit = vec3(0.0);
        if (res.y < 0.5) {
            base = vec3(0.78, 0.62, 0.38);
            spec = pow(head, 24.0) * 0.6;
        } else if (res.y < 1.5) {
            vec3 fu;
            float fh = faceSel(res.z, fu);
            vec3 fc = hsv(fh, 0.45 + 0.4 * fract(fh * 7.31), 0.9);
            float r = lineDist(p, fu) / RHO;
            float rings = smoothstep(0.35, 0.5, abs(fract(r * 7.3) - 0.5) * 2.0);
            base = fc * (0.55 + 0.45 * rings);
            spec = pow(head, 12.0) * 0.25;
        } else {
            base = lampCol;
            emit = lampCol * 0.65;
            spec = pow(head, 40.0) * 0.8;
        }
        col = (base * (0.05 + 0.55 * head + 1.1 * lamp * lampCol) + spec + emit) * ao;
    }

    // Entities live in (central-cell) world coordinates: intersect analytically.
    float tEnt = tHit;
    vec3 entCol = vec3(0.0);
    vec3 glow = vec3(0.0);
    for (int i = 0; i < MAXE; i++) {
        if (float(i) >= uNumEnemies) break;
        vec4 c = uEnemy[i];
        vec4 info = uEnemyInfo[i];
        float r = info.x;
        vec3 ecol = mix(vec3(0.55, 0.05, 0.12), vec3(1.0, 0.12, 0.05), info.y);
        ecol = mix(ecol, vec3(1.0, 0.8, 0.5), info.z);
        float th = hitBall(ro, rd, c, r);
        if (th > 0.0 && th < tEnt) {
            tEnt = th;
            vec4 hp = normPoint(Cf(th) * ro + Sf(th) * rd);
            vec4 hd = proj(-KC * Sf(th) * ro + Cf(th) * rd, hp);
            hd *= inversesqrt(max(mdot(hd, hd), 1e-20));
            vec4 n = -proj(c, hp);
            n *= inversesqrt(max(mdot(n, n), 1e-20));
            float fac = max(mdot(n, -hd), 0.0);
            float pulse = 0.5 + 0.5 * sin(uTime * (3.0 + 6.0 * info.y) + float(i));
            vec3 cc = ecol * (0.25 + 0.85 * fac) + pow(fac, 30.0) * 0.6;
            float eye = smoothstep(0.975, 0.99, fac) * info.y * (1.0 - info.z);
            cc = mix(cc, vec3(1.0, 0.95, 0.6) * (0.8 + 0.4 * pulse), eye);
            cc += ecol * 0.25 * pulse;
            entCol = cc;
        }
        glow += ecol * glowBall(ro, rd, c, r, tEnt, 9.0) * (0.25 + 0.35 * max(info.y, info.z));
    }
    for (int i = 0; i < MAXP; i++) {
        if (float(i) >= uNumProj) break;
        vec4 c = uProj[i];
        float th = hitBall(ro, rd, c, uProjR);
        if (th > 0.0 && th < tEnt) {
            tEnt = th;
            entCol = vec3(0.35, 0.55, 1.0) * 1.7;
        }
        glow += vec3(0.1, 0.3, 1.0) * glowBall(ro, rd, c, uProjR, tEnt, 40.0) * 0.35;
    }
    if (tEnt < tHit) col = entCol;

    float tf = min(tEnt, MAX_DIST);
    col = mix(FOG, col, exp(-FOGK * tf));
    col += glow;
    col = col / (1.0 + col * 0.35);
    col = pow(col, vec3(0.66));
    gl_FragColor = vec4(col, 1.0);
}
"#;
