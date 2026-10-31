#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

//! Curvemaster — a raymarched shooter in hyperbolic, Euclidean or spherical
//! 3-space.
//!
//! Controls: WASD move, mouse look, Q/E roll, Space/Ctrl up/down,
//! left click shoot, [ / ] render scale, Esc menu.

mod geom;
mod shader;
mod world;

use geom::*;
use macroquad::prelude::*;
use macroquad::rand::gen_range;
use shader::{RenderParams, MAX_ENEMIES, MAX_PROJECTILES};
use world::*;

// Gameplay sizes and speeds, in world units (independent of curvature).
const PLAYER_R: f64 = 0.018;
const PLAYER_SPEED: f64 = 0.9;
const ROLL_SPEED: f64 = 1.6;
const MOUSE_SENS: f64 = 1.6;
const PROJ_SPEED: f64 = 3.2;
const PROJ_R: f64 = 0.009;
const PROJ_LIFE: f64 = 5.0;
const FIRE_COOLDOWN: f64 = 0.16;
const ENEMY_R: f64 = 0.09;
const ENEMY_CHASE_SPEED: f64 = 0.5;
const ENEMY_WANDER_SPEED: f64 = 0.1;
const ENEMY_SIGHT: f64 = 6.0;
const ENEMY_TARGET: usize = 14;
const BURST_TIME: f64 = 0.3;
const BURST_R: f64 = 0.045;
const FOV: f32 = 1.2;

#[derive(Clone)]
struct Settings {
    space: Space,
    /// Selected tiling per space.
    tiling: [usize; 3],
    /// Curvature radius R for H3 / S3, in world units.
    radius: [f64; 3],
    /// Cube edge for E3, in world units.
    e3_cell: f64,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { space: Space::Hyperbolic, tiling: [0, 0, 4], radius: [1.0, 1.0, 1.5], e3_cell: 1.6 }
    }
}

fn space_idx(s: Space) -> usize {
    match s {
        Space::Hyperbolic => 0,
        Space::Euclidean => 1,
        Space::Spherical => 2,
    }
}

impl Settings {
    fn kind(&self) -> TilingKind {
        let i = space_idx(self.space);
        catalog(self.space)[self.tiling[i]].kind
    }
    /// World units per internal unit.
    fn unit(&self) -> f64 {
        match self.space {
            Space::Euclidean => 1.0,
            s => self.radius[space_idx(s)],
        }
    }
    fn shape(&self) -> Shape {
        Shape::build(self.space, self.kind(), self.e3_cell)
    }
}

struct Enemy {
    pos: V4,
    dir: V4,
    alert: f64,
    sees: bool,
    los_timer: f64,
}

struct Projectile {
    pos: V4,
    dir: V4,
    life: f64,
}

struct Burst {
    pos: V4,
    age: f64,
}

struct Game {
    w: World,
    g: Geom,
    /// World units per internal unit (curvature radius).
    unit: f64,
    player: Frame,
    /// Identity of the central cell; the world is reflected back into the
    /// central cell whenever the player leaves it.
    label: Label,
    enemies: Vec<Enemy>,
    projectiles: Vec<Projectile>,
    bursts: Vec<Burst>,
    health: f64,
    score: u32,
    cooldown: f64,
    hurt: f64,
    spawn_timer: f64,
    dead: bool,
    time: f64,
    travelled: f64,
    cells_visited: u32,
}

impl Game {
    fn new(settings: &Settings) -> Self {
        let w = World::new(settings.shape());
        let g = *w.g();
        let mut label = w.initial_label();
        // Start in a cell without an orb at its centre.
        let nf = w.shape.faces.len();
        for i in 0..200 {
            if w.orb_param(w.cell_key(&label)) == 0.0 {
                break;
            }
            label = w.reflect_label(label, (i * 5 + 1) % nf);
        }
        // Face the most open wall.
        let cp = w.cell_params(&label);
        let score = |f: f64| if f < -0.5 { 2.0 } else { f };
        let fi = (0..nf)
            .max_by(|&a, &b| score(cp.face[a]).partial_cmp(&score(cp.face[b])).unwrap())
            .unwrap();
        let fwd = w.shape.dirs[fi];
        let helper = if fwd[1].abs() < 0.9 { [0.0, 1.0, 0.0] } else { [1.0, 0.0, 0.0] };
        let cross = |a: [f64; 3], b: [f64; 3]| {
            [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
        };
        let right = cross(helper, fwd);
        let up = cross(fwd, right);
        let mut player = Frame {
            c: [
                [right[0], right[1], right[2], 0.0],
                [up[0], up[1], up[2], 0.0],
                [fwd[0], fwd[1], fwd[2], 0.0],
                ORIGIN,
            ],
        };
        player.orthonormalize(&g);
        let mut game = Game {
            w,
            g,
            unit: settings.unit(),
            player,
            label,
            enemies: Vec::new(),
            projectiles: Vec::new(),
            bursts: Vec::new(),
            health: 100.0,
            score: 0,
            cooldown: 0.0,
            hurt: 0.0,
            spawn_timer: 0.0,
            dead: false,
            time: 0.0,
            travelled: 0.0,
            cells_visited: 1,
        };
        for _ in 0..game.enemy_target() {
            game.try_spawn_enemy();
        }
        game
    }

    /// Convert a world length to internal units.
    fn i(&self, world: f64) -> f64 {
        world / self.unit
    }

    fn enemy_target(&self) -> usize {
        let cap = self.w.shape.cell_count.map(|n| (n * 2).max(4)).unwrap_or(ENEMY_TARGET);
        ENEMY_TARGET.min(cap).min(MAX_ENEMIES)
    }

    /// Internal distance beyond which enemies are dropped (keeps numbers sane
    /// in H3; S3 is finite so nothing is ever too far).
    fn despawn_dist(&self) -> f64 {
        match self.g.space {
            Space::Hyperbolic => self.i(7.5).min(7.0),
            Space::Euclidean => self.i(7.5),
            Space::Spherical => f64::INFINITY,
        }
    }

    fn spawn_range(&self) -> (f64, f64) {
        let (a, b) = (self.i(1.5), self.i(4.5));
        match self.g.space {
            Space::Hyperbolic => (a.min(3.0), b.min(5.0)),
            Space::Euclidean => (a, b),
            Space::Spherical => {
                let pi = std::f64::consts::PI;
                (a.min(0.35 * pi), b.min(0.95 * pi))
            }
        }
    }

    fn random_dir(&self, p: V4) -> V4 {
        loop {
            let v = [gen_range(-1.0, 1.0), gen_range(-1.0, 1.0), gen_range(-1.0, 1.0)];
            let l2: f64 = v.iter().map(|x| x * x).sum();
            if l2 > 0.01 && l2 <= 1.0 {
                // Any vector works: proj() makes it tangent at p.
                let base = [v[0], v[1], v[2], 0.0];
                return self.g.norm_tangent(p, base);
            }
        }
    }

    fn try_spawn_enemy(&mut self) {
        let pp = self.player.pos();
        let (lo, hi) = self.spawn_range();
        for _ in 0..25 {
            let dir = self.random_dir(pp);
            let (pos, _) = self.g.geodesic(pp, dir, gen_range(lo, hi));
            if self.w.wall_dist(pos, self.label) > self.i(ENEMY_R) * 1.6 {
                let dir = self.random_dir(pos);
                self.enemies.push(Enemy { pos, dir, alert: 0.0, sees: false, los_timer: gen_range(0.0, 0.2) });
                return;
            }
        }
    }

    /// Reflect the whole world so the player is back in the central cell.
    fn recenter(&mut self) {
        while let Some(i) = self.w.outside_face(self.player.pos()) {
            let phi = self.w.shape.faces[i];
            let g = self.g;
            self.player.map(|v| g.reflect(v, phi));
            for e in &mut self.enemies {
                e.pos = g.reflect(e.pos, phi);
                e.dir = g.reflect(e.dir, phi);
            }
            for p in &mut self.projectiles {
                p.pos = g.reflect(p.pos, phi);
                p.dir = g.reflect(p.dir, phi);
            }
            for b in &mut self.bursts {
                b.pos = g.reflect(b.pos, phi);
            }
            self.label = self.w.reflect_label(self.label, i);
            if let Label::Center(m) = &mut self.label {
                // Keep the absolute frame orthonormal (S3) / exact (E3).
                if g.space == Space::Spherical {
                    let mut f = Frame { c: m.c };
                    f.orthonormalize(&g);
                    m.c = f.c;
                }
            }
            self.cells_visited += 1;
        }
        self.player.orthonormalize(&self.g);
    }

    fn resolve_player_collision(&mut self) {
        let r = self.i(PLAYER_R);
        for _ in 0..4 {
            let d = self.w.wall_dist(self.player.pos(), self.label);
            if d >= r {
                return;
            }
            let h = 1e-3 * self.w.shape.rho;
            let mut grad = [0.0; 3];
            for (k, gk) in grad.iter_mut().enumerate() {
                let mut dir = [0.0; 3];
                dir[k] = 1.0;
                let mut a = self.player;
                a.translate(&self.g, dir, h);
                let mut b = self.player;
                b.translate(&self.g, dir, -h);
                *gk = (self.w.wall_dist(a.pos(), self.label) - self.w.wall_dist(b.pos(), self.label)) / (2.0 * h);
            }
            let l = (grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]).sqrt();
            if l < 1e-12 {
                return;
            }
            self.player.translate(&self.g, [grad[0] / l, grad[1] / l, grad[2] / l], r - d + 1e-5);
        }
    }

    /// Move the player along a local direction, sub-stepped against walls.
    fn move_player(&mut self, dir: [f64; 3], world_dist: f64) {
        let step = self.i(world_dist);
        let n = (step / (0.4 * self.i(PLAYER_R))).ceil().max(1.0) as usize;
        for _ in 0..n {
            self.player.translate(&self.g, dir, step / n as f64);
            self.resolve_player_collision();
        }
        self.travelled += world_dist;
    }

    fn fire(&mut self) {
        if self.projectiles.len() >= MAX_PROJECTILES {
            return;
        }
        let (pos, dir) = self.g.geodesic(self.player.pos(), self.player.c[2], self.i(0.07));
        self.projectiles.push(Projectile { pos, dir, life: PROJ_LIFE });
    }

    fn update(&mut self, dt: f64) {
        self.time += dt;
        self.hurt = (self.hurt - dt * 2.0).max(0.0);
        for b in &mut self.bursts {
            b.age += dt;
        }
        self.bursts.retain(|b| b.age < BURST_TIME);
        if self.dead {
            return;
        }

        let md = mouse_delta_position();
        self.player.rotate(2, 0, -md.x as f64 * MOUSE_SENS);
        self.player.rotate(2, 1, md.y as f64 * MOUSE_SENS);
        let mut roll = 0.0;
        if is_key_down(KeyCode::Q) {
            roll -= 1.0;
        }
        if is_key_down(KeyCode::E) {
            roll += 1.0;
        }
        self.player.rotate(0, 1, roll * ROLL_SPEED * dt);

        let mut mv = [0.0f64; 3];
        let keys = [
            (KeyCode::W, 2, 1.0),
            (KeyCode::S, 2, -1.0),
            (KeyCode::D, 0, 1.0),
            (KeyCode::A, 0, -1.0),
            (KeyCode::Space, 1, 1.0),
            (KeyCode::LeftControl, 1, -1.0),
            (KeyCode::C, 1, -1.0),
        ];
        for (k, axis, s) in keys {
            if is_key_down(k) {
                mv[axis] += s;
            }
        }
        let l = (mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2]).sqrt();
        if l > 0.0 {
            let boost = if is_key_down(KeyCode::LeftShift) { 1.8 } else { 1.0 };
            self.move_player([mv[0] / l, mv[1] / l, mv[2] / l], PLAYER_SPEED * boost * dt);
        }
        self.player.orthonormalize(&self.g);
        self.recenter();

        self.cooldown -= dt;
        if is_mouse_button_down(MouseButton::Left) && self.cooldown <= 0.0 {
            self.cooldown = FIRE_COOLDOWN;
            self.fire();
        }

        self.update_projectiles(dt);
        self.update_enemies(dt);

        let pp = self.player.pos();
        let dd = self.despawn_dist();
        let g = self.g;
        self.enemies.retain(|e| g.dist(e.pos, pp) < dd);
        self.spawn_timer -= dt;
        if self.spawn_timer <= 0.0 && self.enemies.len() < self.enemy_target() {
            self.spawn_timer = 0.7;
            self.try_spawn_enemy();
        }

        if self.health <= 0.0 {
            self.health = 0.0;
            self.dead = true;
        }
    }

    fn update_projectiles(&mut self, dt: f64) {
        let pr_r = self.i(PROJ_R);
        let en_r = self.i(ENEMY_R);
        let min_step = 0.004 * self.w.shape.rho;
        let mut i = 0;
        while i < self.projectiles.len() {
            self.projectiles[i].life -= dt;
            let mut alive = self.projectiles[i].life > 0.0;
            let mut remaining = self.i(PROJ_SPEED * dt);
            // Advance in SDF-bounded sub-steps so nothing is skipped.
            while alive && remaining > 0.0 {
                let (pos, dir) = (self.projectiles[i].pos, self.projectiles[i].dir);
                let dw = self.w.wall_dist(pos, self.label) - pr_r;
                if dw <= 0.0 {
                    self.bursts.push(Burst { pos, age: 0.0 });
                    alive = false;
                    break;
                }
                let mut de = f64::INFINITY;
                let mut hit = None;
                for (k, e) in self.enemies.iter().enumerate() {
                    let d = self.g.dist(pos, e.pos) - en_r - pr_r;
                    if d < de {
                        de = d;
                        hit = Some(k);
                    }
                }
                if de <= 0.0 {
                    let e = self.enemies.swap_remove(hit.unwrap());
                    self.bursts.push(Burst { pos: e.pos, age: 0.0 });
                    self.score += 1;
                    alive = false;
                    break;
                }
                let step = remaining.min(dw.min(de).max(min_step));
                let (np, nd) = self.g.geodesic(pos, dir, step);
                self.projectiles[i].pos = np;
                self.projectiles[i].dir = nd;
                remaining -= step;
            }
            if alive && self.g.dist(self.projectiles[i].pos, self.player.pos()) > self.despawn_dist() + 1.0 {
                alive = false;
            }
            if alive {
                i += 1;
            } else {
                self.projectiles.swap_remove(i);
            }
        }
    }

    fn update_enemies(&mut self, dt: f64) {
        let pp = self.player.pos();
        let label = self.label;
        let (w, g) = (&self.w, self.g);
        let unit = self.unit;
        let (er, pr) = (ENEMY_R / unit, PLAYER_R / unit);
        let sight = (ENEMY_SIGHT / unit).min(if g.space == Space::Hyperbolic { 6.5 } else { f64::INFINITY });
        let mut damage = 0.0;
        let mut rnd_dirs = Vec::new();
        self.enemies.retain_mut(|e| {
            e.los_timer -= dt;
            let d = g.dist(e.pos, pp);
            if e.los_timer <= 0.0 {
                e.los_timer = 0.15;
                // Line of sight: march the geodesic from the enemy to the
                // player; any wall in between blocks the view.
                e.sees = d < sight && {
                    let v = g.dir_to(e.pos, pp);
                    let start = er * 0.5;
                    let (q, wv) = g.geodesic(e.pos, v, start);
                    w.march(q, wv, (d - start - pr).max(0.0), 0.15 * w.beam, label).is_none()
                };
            }
            let target = if e.sees { 1.0 } else { 0.0 };
            e.alert += (target - e.alert) * (dt * 5.0).min(1.0);
            if e.sees {
                let v = g.dir_to(e.pos, pp);
                let (np, _) = g.geodesic(e.pos, v, ENEMY_CHASE_SPEED / unit * dt);
                if w.wall_dist(np, label) > er * 0.8 {
                    e.pos = np;
                }
                e.dir = g.dir_to(e.pos, pp);
            } else {
                let (np, nd) = g.geodesic(e.pos, e.dir, ENEMY_WANDER_SPEED / unit * dt);
                if w.wall_dist(np, label) > er * 1.1 {
                    e.pos = np;
                    e.dir = nd;
                } else {
                    rnd_dirs.push(e.pos);
                    e.dir = [f64::NAN; 4];
                }
            }
            if g.dist(e.pos, pp) < er + pr {
                damage += 20.0;
                return false;
            }
            true
        });
        for e in self.enemies.iter_mut() {
            if e.dir[0].is_nan() {
                let p = e.pos;
                e.dir = {
                    let v = [gen_range(-1.0, 1.0), gen_range(-1.0, 1.0), gen_range(-1.0, 1.0), 0.0];
                    g.norm_tangent(p, v)
                };
            }
        }
        if damage > 0.0 {
            self.health -= damage;
            self.hurt = 1.0;
        }
    }
}

// --- Rendering ---------------------------------------------------------------

struct Renderer {
    material: Material,
    rt: Option<(RenderTarget, u32, u32)>,
}

fn render_params(s: &Settings, w: &World) -> RenderParams {
    let unit = s.unit();
    match s.space {
        Space::Hyperbolic => RenderParams { max_dist: 9.0, fog: (0.3 * unit).max(0.33), glow_k: unit },
        Space::Euclidean => RenderParams { max_dist: 40.0_f64.max(16.0 * w.shape.rho), fog: 0.11, glow_k: 1.0 },
        Space::Spherical => RenderParams {
            max_dist: 2.0 * std::f64::consts::PI,
            fog: 0.07 * unit,
            glow_k: unit,
        },
    }
}

impl Renderer {
    fn new(settings: &Settings, w: &World) -> Renderer {
        let frag = shader::fragment(w, &render_params(settings, w));
        let material = load_material(
            ShaderSource::Glsl { vertex: shader::VERTEX, fragment: &frag },
            MaterialParams {
                uniforms: vec![
                    UniformDesc::new("uRes", UniformType::Float2),
                    UniformDesc::new("uFrame", UniformType::Mat4),
                    UniformDesc::new("uLabel", UniformType::Float1),
                    UniformDesc::new("uG", UniformType::Mat4),
                    UniformDesc::new("uTime", UniformType::Float1),
                    UniformDesc::new("uFov", UniformType::Float1),
                    UniformDesc::new("uEnemy", UniformType::Float4).array(MAX_ENEMIES),
                    UniformDesc::new("uEnemyInfo", UniformType::Float4).array(MAX_ENEMIES),
                    UniformDesc::new("uNumEnemies", UniformType::Float1),
                    UniformDesc::new("uProj", UniformType::Float4).array(MAX_PROJECTILES),
                    UniformDesc::new("uNumProj", UniformType::Float1),
                    UniformDesc::new("uProjR", UniformType::Float1),
                ],
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("shader compile failed: {e:?}"));
        Renderer { material, rt: None }
    }

    fn draw(&mut self, game: &Game, render_scale: f32) {
        let (sw, sh) = (screen_width(), screen_height());
        let w = ((sw * render_scale) as u32).max(64);
        let h = ((sh * render_scale) as u32).max(36);
        if self.rt.as_ref().map(|r| (r.1, r.2)) != Some((w, h)) {
            let target = render_target(w, h);
            target.texture.set_filter(FilterMode::Linear);
            self.rt = Some((target, w, h));
        }
        let target = self.rt.as_ref().unwrap().0.clone();
        let m = &self.material;
        let mat4 = |c: [[f32; 4]; 4]| {
            Mat4::from_cols(
                Vec4::from_array(c[0]),
                Vec4::from_array(c[1]),
                Vec4::from_array(c[2]),
                Vec4::from_array(c[3]),
            )
        };
        m.set_uniform("uRes", vec2(w as f32, h as f32));
        m.set_uniform("uFrame", mat4(game.player.to_f32_cols()));
        match &game.label {
            Label::Parity(l) => {
                m.set_uniform("uLabel", *l as f32);
                m.set_uniform("uG", Mat4::IDENTITY);
            }
            Label::Center(g) => {
                m.set_uniform("uLabel", 0.0f32);
                m.set_uniform("uG", mat4(g.to_f32_cols()));
            }
        }
        m.set_uniform("uTime", game.time as f32);
        m.set_uniform("uFov", FOV);
        let v4f = |v: V4| vec4(v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32);
        let mut ep = [Vec4::ZERO; MAX_ENEMIES];
        let mut ei = [Vec4::ZERO; MAX_ENEMIES];
        let mut ne = 0;
        for e in game.enemies.iter().take(MAX_ENEMIES) {
            ep[ne] = v4f(e.pos);
            ei[ne] = vec4(game.i(ENEMY_R) as f32, e.alert as f32, 0.0, 0.0);
            ne += 1;
        }
        for b in &game.bursts {
            if ne >= MAX_ENEMIES {
                break;
            }
            let k = b.age / BURST_TIME;
            ep[ne] = v4f(b.pos);
            ei[ne] = vec4(game.i(BURST_R * (0.25 + 0.75 * k)) as f32, 1.0, 1.0 - k as f32, 0.0);
            ne += 1;
        }
        let mut pp = [Vec4::ZERO; MAX_PROJECTILES];
        for (i, p) in game.projectiles.iter().take(MAX_PROJECTILES).enumerate() {
            pp[i] = v4f(p.pos);
        }
        m.set_uniform_array("uEnemy", &ep[..]);
        m.set_uniform_array("uEnemyInfo", &ei[..]);
        m.set_uniform("uNumEnemies", ne as f32);
        m.set_uniform_array("uProj", &pp[..]);
        m.set_uniform("uNumProj", game.projectiles.len().min(MAX_PROJECTILES) as f32);
        m.set_uniform("uProjR", game.i(PROJ_R) as f32);

        set_camera(&Camera2D {
            render_target: Some(target.clone()),
            ..Camera2D::from_display_rect(Rect::new(0.0, 0.0, w as f32, h as f32))
        });
        clear_background(BLACK);
        gl_use_material(m);
        draw_rectangle(0.0, 0.0, w as f32, h as f32, WHITE);
        gl_use_default_material();

        set_default_camera();
        clear_background(BLACK);
        draw_texture_ex(
            &target.texture,
            0.0,
            0.0,
            WHITE,
            DrawTextureParams {
                dest_size: Some(vec2(sw, sh)),
                // Render targets are stored bottom-up.
                flip_y: true,
                ..Default::default()
            },
        );
    }
}

// --- UI -----------------------------------------------------------------------

fn button(r: Rect, text: &str, selected: bool) -> bool {
    let (mx, my) = mouse_position();
    let hover = r.contains(vec2(mx, my));
    let bg = if selected {
        Color::new(0.35, 0.25, 0.75, 0.95)
    } else if hover {
        Color::new(0.25, 0.22, 0.4, 0.95)
    } else {
        Color::new(0.12, 0.1, 0.2, 0.9)
    };
    draw_rectangle(r.x, r.y, r.w, r.h, bg);
    draw_rectangle_lines(r.x, r.y, r.w, r.h, 1.5, Color::new(0.7, 0.6, 1.0, 0.8));
    let fs = 22.0;
    let d = measure_text(text, None, fs as u16, 1.0);
    draw_text(text, r.x + (r.w - d.width) / 2.0, r.y + r.h / 2.0 + d.offset_y / 2.0 - 2.0, fs, WHITE);
    hover && is_mouse_button_pressed(MouseButton::Left)
}

/// A "< value >" stepper; returns -1, 0 or +1.
fn stepper(x: f32, y: f32, w: f32, label: &str, value: &str) -> i32 {
    draw_text(label, x, y + 24.0, 24.0, Color::new(0.85, 0.85, 1.0, 1.0));
    let bx = x + 260.0;
    let mut r = 0;
    if button(Rect::new(bx, y, 40.0, 34.0), "<", false) {
        r = -1;
    }
    let vw = w - 260.0 - 90.0;
    draw_rectangle(bx + 45.0, y, vw, 34.0, Color::new(0.05, 0.04, 0.1, 0.9));
    let d = measure_text(value, None, 22, 1.0);
    draw_text(value, bx + 45.0 + (vw - d.width) / 2.0, y + 23.0, 22.0, WHITE);
    if button(Rect::new(bx + 50.0 + vw, y, 40.0, 34.0), ">", false) {
        r = 1;
    }
    r
}

enum MenuAction {
    None,
    Play,
    Resume,
    Quit,
}

fn menu(settings: &mut Settings, can_resume: bool, cache: &mut Option<(String, Shape)>) -> MenuAction {
    let (sw, sh) = (screen_width(), screen_height());
    draw_rectangle(0.0, 0.0, sw, sh, Color::new(0.0, 0.0, 0.0, 0.6));
    let pw = 760.0f32.min(sw - 40.0);
    let x = (sw - pw) / 2.0;
    let mut y = (sh - 560.0).max(20.0) / 2.0;
    let title = "CURVEMASTER";
    let d = measure_text(title, None, 56, 1.0);
    draw_text(title, (sw - d.width) / 2.0, y + 44.0, 56.0, Color::new(0.8, 0.7, 1.0, 1.0));
    y += 70.0;
    let sub = "a shooter in curved space";
    let d = measure_text(sub, None, 22, 1.0);
    draw_text(sub, (sw - d.width) / 2.0, y, 22.0, GRAY);
    y += 24.0;

    // Geometry selector.
    let spaces = [Space::Hyperbolic, Space::Euclidean, Space::Spherical];
    let bw = (pw - 20.0) / 3.0;
    for (k, s) in spaces.iter().enumerate() {
        if button(Rect::new(x + k as f32 * (bw + 10.0), y, bw, 40.0), s.name(), settings.space == *s) {
            settings.space = *s;
        }
    }
    y += 60.0;

    let si = space_idx(settings.space);
    let cat = catalog(settings.space);
    let ti = &mut settings.tiling[si];
    let step = stepper(x, y, pw, "Grid cells", cat[*ti].name);
    *ti = ((*ti as i32 + step).rem_euclid(cat.len() as i32)) as usize;
    y += 48.0;

    // Cache the built cell shape (spherical ones enumerate the whole world).
    let key = format!("{:?}{:?}{}", settings.space, settings.kind(), settings.e3_cell);
    if cache.as_ref().map(|c| &c.0) != Some(&key) {
        *cache = Some((key, settings.shape()));
    }
    let shape = &cache.as_ref().unwrap().1;
    let rho = shape.rho;

    let fmt = |v: f64| format!("{v:.2}");
    match settings.space {
        Space::Euclidean => {
            let s = stepper(x, y, pw, "Cell size", &fmt(settings.e3_cell));
            settings.e3_cell = (settings.e3_cell * 1.15f64.powi(s)).clamp(0.6, 8.0);
            y += 48.0;
        }
        _ => {
            let r = &mut settings.radius[si];
            let s = stepper(x, y, pw, "Curvature radius R", &fmt(*r));
            *r = (*r * 1.15f64.powi(s)).clamp(0.25, 12.0);
            y += 48.0;
            // Cell size is tied to R by the tiling; the slider moves R.
            let size = 2.0 * rho * *r;
            let s2 = stepper(x, y, pw, "Cell size", &fmt(size));
            *r = (*r * 1.15f64.powi(s2)).clamp(0.25, 12.0);
            y += 48.0;
        }
    }

    // Derived information.
    let unit = settings.unit();
    let mut info = Vec::new();
    match settings.space {
        Space::Hyperbolic => {
            info.push(format!("Curvature K = -1/R^2 = {:.3}", -1.0 / (unit * unit)));
            info.push(format!(
                "Cell inradius {:.2}  ({:.3} R, fixed by the tiling)   faces per cell: {}",
                rho * unit,
                rho,
                shape.faces.len()
            ));
            info.push("Infinite: the number of cells within distance d grows like e^(2d/R)".into());
        }
        Space::Euclidean => {
            info.push("Curvature K = 0 (flat)".into());
            info.push(format!("Cube edge {:.2}; infinite, polynomial growth", settings.e3_cell));
        }
        Space::Spherical => {
            let n = shape.cell_count.unwrap_or(0);
            info.push(format!("Curvature K = +1/R^2 = {:.3}", 1.0 / (unit * unit)));
            info.push(format!(
                "Finite world: {} cells, volume 2pi^2 R^3 = {:.1}, great circle 2pi R = {:.2}",
                n,
                2.0 * std::f64::consts::PI.powi(2) * unit.powi(3),
                2.0 * std::f64::consts::PI * unit
            ));
            info.push(format!("Cell inradius {:.2} ({:.3} R, fixed by the tiling)", rho * unit, rho));
        }
    }
    info.push("In curved space a regular grid's cell size is set by the curvature,".into());
    info.push("so R and cell size move together; pick another grid for other ratios.".into());
    for l in &info {
        draw_text(l, x, y + 18.0, 20.0, Color::new(0.75, 0.75, 0.85, 1.0));
        y += 24.0;
    }
    y += 16.0;

    let mut act = MenuAction::None;
    let nb = if can_resume { 3.0 } else { 2.0 };
    let bw = (pw - 10.0 * (nb - 1.0)) / nb;
    let mut bx = x;
    if can_resume {
        if button(Rect::new(bx, y, bw, 48.0), "RESUME", false) {
            act = MenuAction::Resume;
        }
        bx += bw + 10.0;
    }
    if button(Rect::new(bx, y, bw, 48.0), if can_resume { "NEW GAME" } else { "PLAY" }, false) {
        act = MenuAction::Play;
    }
    bx += bw + 10.0;
    if button(Rect::new(bx, y, bw, 48.0), "QUIT", false) {
        act = MenuAction::Quit;
    }
    y += 70.0;
    for l in [
        "WASD move   mouse look   Q/E roll   Space/Ctrl up/down   Shift sprint",
        "Left click shoot   [ ] render resolution   Esc menu",
    ] {
        let d = measure_text(l, None, 18, 1.0);
        draw_text(l, (sw - d.width) / 2.0, y, 18.0, GRAY);
        y += 22.0;
    }
    act
}

fn draw_hud(game: &Game, render_scale: f32) {
    let (sw, sh) = (screen_width(), screen_height());
    let (cx, cy) = (sw / 2.0, sh / 2.0);
    let cc = Color::new(1.0, 1.0, 1.0, 0.8);
    draw_line(cx - 10.0, cy, cx - 3.0, cy, 2.0, cc);
    draw_line(cx + 3.0, cy, cx + 10.0, cy, 2.0, cc);
    draw_line(cx, cy - 10.0, cx, cy - 3.0, 2.0, cc);
    draw_line(cx, cy + 3.0, cx, cy + 10.0, 2.0, cc);
    if game.hurt > 0.0 {
        draw_rectangle(0.0, 0.0, sw, sh, Color::new(0.8, 0.0, 0.0, 0.35 * game.hurt as f32));
    }
    let bw = 220.0;
    draw_rectangle(20.0, sh - 40.0, bw, 16.0, Color::new(0.0, 0.0, 0.0, 0.5));
    let hf = (game.health / 100.0) as f32;
    draw_rectangle(20.0, sh - 40.0, bw * hf, 16.0, Color::new(1.0 - hf, 0.3 + 0.6 * hf, 0.3, 0.9));
    draw_rectangle_lines(20.0, sh - 40.0, bw, 16.0, 1.5, WHITE);
    let seeing = game.enemies.iter().filter(|e| e.sees).count();
    let lines = [
        format!("SCORE {}", game.score),
        format!(
            "{}  R={:.2}   cells crossed {}   travelled {:.1}",
            game.g.space.name(),
            game.unit,
            game.cells_visited - 1,
            game.travelled
        ),
        format!(
            "enemies {}  (hunting you: {})   fps {}   scale {:.0}%",
            game.enemies.len(),
            seeing,
            get_fps(),
            render_scale * 100.0
        ),
    ];
    for (i, l) in lines.iter().enumerate() {
        draw_text(l, 20.0, 30.0 + 22.0 * i as f32, 22.0, WHITE);
    }
    if game.dead {
        draw_rectangle(0.0, 0.0, sw, sh, Color::new(0.0, 0.0, 0.0, 0.55));
        let t = "LOST IN CURVED SPACE";
        let d = measure_text(t, None, 48, 1.0);
        draw_text(t, cx - d.width / 2.0, cy - 20.0, 48.0, RED);
        let t2 = format!("score {}  -  press R to restart, Esc for menu", game.score);
        let d2 = measure_text(&t2, None, 28, 1.0);
        draw_text(&t2, cx - d2.width / 2.0, cy + 20.0, 28.0, WHITE);
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Curvemaster".to_owned(),
        window_width: 1280,
        window_height: 720,
        high_dpi: false,
        ..Default::default()
    }
}

fn grab(on: bool) {
    set_cursor_grab(on);
    show_mouse(!on);
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut settings = Settings::default();
    let mut game: Option<(Game, Renderer)> = None;
    let mut in_menu = true;
    let mut render_scale: f32 = 0.5;
    let mut shape_cache: Option<(String, Shape)> = None;

    // Headless capture: CURVEMASTER_SHOT=out.png [CURVEMASTER_SPACE=h|e|s]
    // [CURVEMASTER_TILING=n] [CURVEMASTER_R=x] [CURVEMASTER_MENU=1]
    let shot = std::env::var("CURVEMASTER_SHOT").ok();
    let env = |k: &str| std::env::var(k).ok();
    if shot.is_some() {
        if let Some(s) = env("CURVEMASTER_SPACE") {
            settings.space = match s.as_str() {
                "e" => Space::Euclidean,
                "s" => Space::Spherical,
                _ => Space::Hyperbolic,
            };
        }
        let si = space_idx(settings.space);
        if let Some(t) = env("CURVEMASTER_TILING").and_then(|v| v.parse().ok()) {
            settings.tiling[si] = t;
        }
        if let Some(r) = env("CURVEMASTER_R").and_then(|v| v.parse().ok()) {
            settings.radius[si] = r;
            settings.e3_cell = r;
        }
        if env("CURVEMASTER_MENU").is_none() {
            let g = Game::new(&settings);
            let r = Renderer::new(&settings, &g.w);
            game = Some((g, r));
            in_menu = false;
        }
    }
    let mut frame_no = 0u32;

    loop {
        let dt = get_frame_time().min(0.05) as f64;
        if is_key_pressed(KeyCode::LeftBracket) {
            render_scale = (render_scale - 0.1).max(0.2);
        }
        if is_key_pressed(KeyCode::RightBracket) {
            render_scale = (render_scale + 0.1).min(1.0);
        }

        if in_menu {
            if let Some((g, r)) = &mut game {
                r.draw(g, render_scale);
            } else {
                clear_background(Color::new(0.03, 0.02, 0.06, 1.0));
            }
            match menu(&mut settings, game.is_some(), &mut shape_cache) {
                MenuAction::Play => {
                    let g = Game::new(&settings);
                    let r = Renderer::new(&settings, &g.w);
                    game = Some((g, r));
                    in_menu = false;
                    grab(true);
                }
                MenuAction::Resume => {
                    in_menu = false;
                    grab(true);
                }
                MenuAction::Quit => break,
                MenuAction::None => {}
            }
            if is_key_pressed(KeyCode::Escape) && game.is_some() {
                in_menu = false;
                grab(true);
            }
        } else if let Some((g, r)) = &mut game {
            if shot.is_some() {
                // Scripted fly-through for screenshots.
                g.player.rotate(2, 0, 0.004);
                g.move_player([0.0, 0.0, 1.0], 0.01);
                g.recenter();
                if frame_no % 8 == 0 {
                    g.fire();
                }
                g.update_projectiles(1.0 / 60.0);
                g.update_enemies(1.0 / 60.0);
                for b in &mut g.bursts {
                    b.age += 1.0 / 60.0;
                }
                g.bursts.retain(|b| b.age < BURST_TIME);
            } else {
                g.update(dt);
                if g.dead && is_key_pressed(KeyCode::R) {
                    *g = Game::new(&settings);
                }
            }
            r.draw(g, render_scale);
            draw_hud(g, render_scale);
            if is_key_pressed(KeyCode::Escape) {
                in_menu = true;
                grab(false);
            }
        }

        if let Some(path) = &shot {
            frame_no += 1;
            if frame_no == 90 {
                get_screen_data().export_png(path);
                break;
            }
        }
        next_frame().await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings(space: Space, tiling: usize) -> Settings {
        let mut s = Settings { space, ..Default::default() };
        s.tiling[space_idx(space)] = tiling;
        s
    }

    fn all() -> Vec<Settings> {
        let mut v = Vec::new();
        for sp in [Space::Hyperbolic, Space::Euclidean, Space::Spherical] {
            for t in 0..catalog(sp).len() {
                v.push(settings(sp, t));
            }
        }
        v
    }

    #[test]
    fn projectile_flies_until_it_hits_a_wall() {
        for s in all() {
            let mut g = Game::new(&s);
            g.enemies.clear();
            let p = g.player.pos();
            let v = g.player.c[2];
            g.projectiles.push(Projectile { pos: p, dir: v, life: 1000.0 });
            let mut steps = 0;
            while !g.projectiles.is_empty() && steps < 20000 {
                g.update_projectiles(1.0 / 60.0);
                steps += 1;
            }
            assert!(g.projectiles.is_empty(), "{:?} never collided", s.kind());
            // Either it hit a wall exactly, or it flew out of range through
            // aligned holes (common along a face axis).
            if let Some(b) = g.bursts.first() {
                assert!(g.w.wall_dist(b.pos, g.label) < g.i(PROJ_R) + 1e-6);
            } else {
                assert!(steps as f64 / 60.0 * g.i(PROJ_SPEED) > g.despawn_dist());
            }
        }
    }

    #[test]
    fn projectile_kills_enemy() {
        for s in all() {
            let mut g = Game::new(&s);
            g.enemies.clear();
            let p = g.player.pos();
            let v = g.player.c[2];
            let (ep, _) = g.g.geodesic(p, v, g.i(0.3));
            g.enemies.push(Enemy { pos: ep, dir: v, alert: 0.0, sees: false, los_timer: 9.0 });
            g.projectiles.push(Projectile { pos: p, dir: v, life: 5.0 });
            for _ in 0..60 {
                g.update_projectiles(1.0 / 60.0);
            }
            assert!(g.enemies.is_empty(), "{:?}", s.kind());
            assert_eq!(g.score, 1);
        }
    }

    #[test]
    fn visible_enemy_chases_player() {
        for s in all() {
            let mut g = Game::new(&s);
            g.enemies.clear();
            let p = g.player.pos();
            let v = g.player.c[2];
            let (ep, _) = g.g.geodesic(p, v, g.i(0.3));
            g.enemies.push(Enemy { pos: ep, dir: v, alert: 0.0, sees: false, los_timer: 0.0 });
            let d0 = g.g.dist(ep, p);
            g.update_enemies(0.1);
            assert!(g.enemies[0].sees, "{:?}", s.kind());
            assert!(g.g.dist(g.enemies[0].pos, p) < d0);
        }
    }

    #[test]
    fn walls_block_sight() {
        for s in all() {
            let g = Game::new(&s);
            let cp = g.w.cell_params(&g.label);
            if let Some(i) = (0..cp.face.len()).find(|&i| cp.face[i] == 0.0) {
                let u = g.w.shape.dirs[i];
                let v = [u[0], u[1], u[2], 0.0];
                let face_d = g.g.asf(-dot4(g.w.shape.faces[i], ORIGIN));
                let (behind, _) = g.g.geodesic(ORIGIN, v, face_d + 0.4 * g.w.shape.rho);
                let d = g.g.dist(ORIGIN, behind);
                assert!(g.w.march(ORIGIN, v, d, 0.01 * g.w.shape.rho, g.label).is_some(), "{:?} face {i}", s.kind());
            }
        }
    }

    #[test]
    fn long_walk_stays_consistent() {
        // Walk far in every geometry; the player must stay in the central cell
        // with a well-formed frame, and the cell must remain identifiable.
        for s in all() {
            let mut g = Game::new(&s);
            let mut last = (g.cells_visited, 0);
            for k in 0..3000 {
                g.player.rotate(2, 0, 0.002 * ((k / 300) as f64 - 4.0));
                if k - last.1 > 150 {
                    // Bumped into a solid wall: turn somewhere else.
                    if g.cells_visited == last.0 {
                        g.player.rotate(2, 0, 1.3);
                        g.player.rotate(2, 1, 0.7);
                    }
                    last = (g.cells_visited, k);
                }
                g.move_player([0.0, 0.0, 1.0], 0.02);
                g.recenter();
                assert!(g.w.outside_face(g.player.pos()).is_none());
            }
            let p = g.player.pos();
            assert!(p.iter().all(|x| x.is_finite() && x.abs() < 10.0), "{:?}", s.kind());
            assert!(g.cells_visited > 3, "{:?} stuck: {}", s.kind(), g.cells_visited);
        }
    }
}
