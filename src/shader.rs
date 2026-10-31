//! GLSL generation. The honeycomb's face normals, edge and vertex lists are
//! emitted as unrolled code so the shader stays GLSL 1.00 compatible (no
//! constant arrays, no integer ops) and mirrors `world.rs` exactly.

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

fn v4(v: [f64; 4]) -> String {
    format!("vec4({:.12}, {:.12}, {:.12}, {:.12})", v[0], v[1], v[2], v[3])
}

fn v3(v: [f64; 3]) -> String {
    format!("vec3({:.12}, {:.12}, {:.12})", v[0], v[1], v[2])
}

pub fn fragment(h: &Honeycomb) -> String {
    let mut consts = String::new();
    for i in 0..NFACES {
        consts += &format!("const vec4 N{i} = {};\n", v4(h.normals[i]));
        consts += &format!("const vec3 U{i} = {};\n", v3(h.dirs[i]));
    }

    // Per-cell parameters, recomputed whenever the ray enters a new cell.
    let mut globals = String::new();
    let mut cell = String::new();
    for i in 0..NFACES {
        globals += &format!("float gF{i};\n");
        cell += &format!("    gF{i} = faceParam(lab, {i}.0, {}.0);\n", 1u32 << i);
    }

    // Fold: reflect the ray state across any face it has crossed.
    let mut fold = String::new();
    for i in 0..NFACES {
        fold += &format!(
            "        s = mdot(N{i}, p); if (s > 0.0) {{ p -= 2.0 * s * N{i}; d -= 2.0 * mdot(d, N{i}) * N{i}; lab = toggle(lab, {}.0); changed = true; }}\n",
            1u32 << i
        );
    }

    let mut sdf = String::new();
    for i in 0..NFACES {
        sdf += &format!("    float s{i} = mdot(N{i}, p);\n");
    }
    sdf += "    float smax = s0;\n";
    for i in 1..NFACES {
        sdf += &format!("    smax = max(smax, s{i});\n");
    }
    sdf += "    float e2 = 1e9;\n";
    for &(i, j) in &h.edges {
        sdf += &format!("    e2 = min(e2, s{i}*s{i} + s{j}*s{j});\n");
    }
    sdf += "    float v2 = 1e9;\n";
    for &(i, j, l) in &h.verts {
        sdf += &format!("    v2 = min(v2, s{i}*s{i} + s{j}*s{j} + s{l}*s{l});\n");
    }
    sdf += &format!(
        "    float dd = min(hasinh(sqrt(e2)) - {BEAM_R:.6}, hasinh(sqrt(v2)) - {NODE_R:.6});\n    float mat = 0.0; float fid = -1.0;\n"
    );
    for i in 0..NFACES {
        sdf += &format!(
            "    if (gF{i} > -0.5) {{ float sl = abs(hasinh(s{i})) - {PLATE_TH:.6}; if (sl < dd) {{ if (gF{i} > 0.0) {{ float ux = dot(U{i}, p.xyz); sl = max(sl, gF{i} - hacosh(sqrt(max(p.w*p.w - ux*ux, 1.0)))); }} if (sl < dd) {{ dd = sl; mat = 1.0; fid = {i}.0; }} }} }}\n"
        );
    }
    sdf += "    if (gOrb > 0.0) { float o = hacosh(p.w) - gOrb; if (o < dd) { dd = o; mat = 2.0; } }\n";
    sdf += &format!("    return vec4(dd, mat, fid, hasinh(-smax) + {BOUND_SLACK:.6});\n");

    let mut face_sel = String::new();
    for i in 0..NFACES {
        face_sel += &format!(
            "    if (fid < {i}.5) {{ n = N{i}; u = U{i}; p2 = {}.0; return; }}\n",
            1u32 << i
        );
    }

    TEMPLATE
        .replace("//CONSTS", &consts)
        .replace("//GLOBALS", &globals)
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
uniform float uTime;
uniform float uFov;
uniform vec4 uEnemy[MAXE];
uniform vec4 uEnemyInfo[MAXE];
uniform float uNumEnemies;
uniform vec4 uProj[MAXP];
uniform float uNumProj;

//CONSTS

float mdot(vec4 a, vec4 b) { return dot(a.xyz, b.xyz) - a.w * b.w; }
float hcosh(float x) { float e = exp(x); return 0.5 * (e + 1.0 / e); }
float hsinh(float x) { float e = exp(x); return 0.5 * (e - 1.0 / e); }
float hasinh(float x) { return sign(x) * log(abs(x) + sqrt(x * x + 1.0)); }
float hacosh(float x) { x = max(x, 1.0); return log(x + sqrt(x * x - 1.0)); }

// Exact small-integer hash, identical to world.rs.
float imod(float x, float m) { return x - m * floor((x + 0.5) / m); }
float hsh(float k, float salt) {
    float hi = floor(k / 64.0);
    float lo = k - hi * 64.0;
    float a = imod(hi * 67.0 + lo * 13.0 + salt * 29.0 + 5.0, 1009.0);
    float b = imod(a * a + lo * 7.0 + 11.0, 1013.0);
    float c = imod(b * (a + 3.0) + salt + 1.0, 1019.0);
    return c / 1019.0;
}
float bitOf(float lab, float p2) { return imod(floor(lab / p2), 2.0); }
float toggle(float lab, float p2) { return bitOf(lab, p2) > 0.5 ? lab - p2 : lab + p2; }
float faceKey(float lab, float i, float p2) { return (lab - bitOf(lab, p2) * p2) * 12.0 + i; }
float faceParam(float lab, float i, float p2) {
    float key = faceKey(lab, i, p2);
    float r = hsh(key, 1.0);
    if (r < 0.42) return -1.0;
    if (r < 0.80) return 0.2 + 0.22 * hsh(key, 2.0);
    return 0.0;
}

//GLOBALS
float gOrb;

void cellParams(float lab) {
//CELL
    gOrb = hsh(lab, 3.0) < 0.33 ? 0.14 + 0.2 * hsh(lab, 4.0) : 0.0;
}

// x: content distance, y: material, z: face id, w: distance bound to cell exit
vec4 sdfLocal(vec4 p) {
//SDF
}

void faceSel(float fid, out vec4 n, out vec3 u, out float p2) {
//FACESEL
    n = N11; u = U11; p2 = 2048.0;
}

vec3 hsv(float h, float s, float v) {
    vec3 k = clamp(abs(mod(h * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
    return v * mix(vec3(1.0), k, s);
}

vec3 cellColor(float lab) { return hsv(hsh(lab, 5.0), 0.55 + 0.35 * hsh(lab, 6.0), 1.0); }

// First intersection of the geodesic ray (ro, rd) with a ball. Returns -1 on miss.
float hitBall(vec4 ro, vec4 rd, vec4 c, float r) {
    float A = -mdot(ro, c);
    float B = -mdot(rd, c);
    float C = hcosh(r);
    float disc = C * C - A * A + B * B;
    if (disc < 0.0) return -1.0;
    float x = (C - sqrt(disc)) / (A + B);
    if (x <= 1.0) return -1.0;
    return log(x);
}

// Glow from a ball: closest approach of the ray, if it lies before tmax.
float glowBall(vec4 ro, vec4 rd, vec4 c, float r, float tmax, float k) {
    float A = -mdot(ro, c);
    float B = -mdot(rd, c);
    if (B >= 0.0 || -B >= A) return 0.0;
    float ts = 0.5 * log((A - B) / (A + B)); // atanh(-B/A)
    if (ts > tmax) return 0.0;
    float dmin = hacosh(sqrt(max(A * A - B * B, 1.0)));
    return exp(-k * max(dmin - r, 0.0)) / (1.0 + 0.6 * ts);
}

const int MAX_STEPS = 160;
const float MAX_DIST = 9.0;
const vec3 FOG = vec3(0.025, 0.018, 0.055);

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;
    vec3 ld = normalize(vec3(uv * uFov, 1.0));
    vec4 ro = uFrame[3];
    vec4 rd = uFrame * vec4(ld, 0.0);
    float pix = uFov / uRes.y;

    // March the walls inside the central cell, reflecting the ray across
    // faces as it leaves, so coordinates stay small however far it goes.
    vec4 p = ro;
    vec4 d = rd;
    float lab = uLabel;
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
        float eps = max(0.0006, 0.7 * pix * hsinh(t));
        if (res.x < eps) { hit = true; break; }
        t += h;
        steps += 1.0;
        if (t > MAX_DIST) break;
        float c = hcosh(h);
        float sn = hsinh(h);
        vec4 np = c * p + sn * d;
        d = sn * p + c * d;
        p = np * inversesqrt(-mdot(np, np));
        d += mdot(d, p) * p;
        d *= inversesqrt(mdot(d, d));
    }

    vec3 col = FOG;
    float tHit = hit ? t : 1e9;

    if (hit) {
        // Normal from the ambient gradient, raised with the Minkowski metric
        // and projected onto the tangent space at p.
        float e = 0.0015;
        vec4 g = vec4(
            sdfLocal(p + vec4(e, 0.0, 0.0, 0.0)).x - sdfLocal(p - vec4(e, 0.0, 0.0, 0.0)).x,
            sdfLocal(p + vec4(0.0, e, 0.0, 0.0)).x - sdfLocal(p - vec4(0.0, e, 0.0, 0.0)).x,
            sdfLocal(p + vec4(0.0, 0.0, e, 0.0)).x - sdfLocal(p - vec4(0.0, 0.0, e, 0.0)).x,
            sdfLocal(p + vec4(0.0, 0.0, 0.0, e)).x - sdfLocal(p - vec4(0.0, 0.0, 0.0, e)).x);
        vec4 n = vec4(g.xyz, -g.w);
        n += mdot(n, p) * p;
        n *= inversesqrt(max(mdot(n, n), 1e-12));
        vec4 v = -d;
        if (mdot(n, v) < 0.0) n = -n;

        // Each cell has a lamp at its centre, tinted with the cell's colour.
        vec3 lampCol = cellColor(lab);
        vec4 o = vec4(0.0, 0.0, 0.0, 1.0);
        float dl = hacosh(p.w);
        vec4 L = o - p.w * p;
        L *= inversesqrt(max(mdot(L, L), 1e-12));
        float lamp = max(mdot(n, L), 0.0) / (1.0 + 2.5 * dl * dl);

        float head = max(mdot(n, v), 0.0);
        float ao = clamp(1.0 - steps / 90.0, 0.35, 1.0);

        vec3 base;
        float spec = 0.0;
        vec3 emit = vec3(0.0);
        if (res.y < 0.5) {
            base = vec3(0.78, 0.62, 0.38);
            spec = pow(head, 24.0) * 0.6;
        } else if (res.y < 1.5) {
            vec4 fn; vec3 fu; float p2;
            faceSel(res.z, fn, fu, p2);
            float key = faceKey(lab, res.z, p2);
            vec3 fc = hsv(hsh(key, 7.0), 0.45 + 0.4 * hsh(key, 8.0), 0.9);
            float ux = dot(fu, p.xyz);
            float r = hacosh(sqrt(max(p.w * p.w - ux * ux, 1.0)));
            float rings = smoothstep(0.35, 0.5, abs(fract(r * 9.0) - 0.5) * 2.0);
            base = fc * (0.55 + 0.45 * rings);
            spec = pow(head, 12.0) * 0.25;
        } else {
            base = lampCol;
            emit = lampCol * 0.65;
            spec = pow(head, 40.0) * 0.8;
        }
        vec3 lit = base * (0.06 + 0.7 * head + 1.4 * lamp * lampCol) + spec + emit;
        col = lit * ao;
    }

    // Entities live in world coordinates; intersect them analytically.
    float tEnt = tHit;
    vec3 entCol = vec3(0.0);
    vec3 glow = vec3(0.0);
    for (int i = 0; i < MAXE; i++) {
        if (float(i) >= uNumEnemies) break;
        vec4 c = uEnemy[i];
        vec4 info = uEnemyInfo[i];
        float r = info.x;
        vec3 ecol = mix(vec3(0.55, 0.05, 0.12), vec3(1.0, 0.12, 0.05), info.y);
        ecol = mix(ecol, vec3(1.0), info.z);
        float th = hitBall(ro, rd, c, r);
        if (th > 0.0 && th < tEnt) {
            tEnt = th;
            vec4 hp = hcosh(th) * ro + hsinh(th) * rd;
            vec4 hd = hsinh(th) * ro + hcosh(th) * rd;
            vec4 n = -(c + mdot(c, hp) * hp);
            n *= inversesqrt(max(mdot(n, n), 1e-12));
            float fac = max(mdot(n, -hd), 0.0);
            float pulse = 0.5 + 0.5 * sin(uTime * (3.0 + 6.0 * info.y) + float(i));
            vec3 cc = ecol * (0.25 + 0.85 * fac) + pow(fac, 30.0) * 0.6;
            // An eye that opens when the enemy has line of sight.
            float eye = smoothstep(0.975, 0.99, fac) * info.y;
            cc = mix(cc, vec3(1.0, 0.95, 0.6) * (0.8 + 0.4 * pulse), eye);
            cc += ecol * 0.25 * pulse;
            entCol = cc;
        }
        glow += ecol * glowBall(ro, rd, c, r, tEnt, 9.0) * (0.25 + 0.35 * info.y);
    }
    for (int i = 0; i < MAXP; i++) {
        if (float(i) >= uNumProj) break;
        vec4 c = uProj[i];
        float th = hitBall(ro, rd, c, 0.03);
        if (th > 0.0 && th < tEnt) {
            tEnt = th;
            entCol = vec3(0.7, 1.0, 1.0) * 1.6;
        }
        glow += vec3(0.2, 0.8, 1.0) * glowBall(ro, rd, c, 0.03, tEnt, 14.0) * 0.9;
    }
    if (tEnt < tHit) col = entCol;

    float tf = min(tEnt, MAX_DIST);
    col = mix(FOG, col, exp(-0.32 * tf));
    col += glow;
    col = col / (1.0 + col * 0.35);
    col = pow(col, vec3(0.6));
    gl_FragColor = vec4(col, 1.0);
}
"#;
