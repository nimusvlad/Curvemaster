#![cfg_attr(all(windows, not(debug_assertions)), windows_subsystem = "windows")]

//! Curvemaster — a raymarched shooter in infinite H3 hyperbolic space.
//!
//! Controls: WASD move, mouse look, Q/E roll, Space/Ctrl up/down,
//! left click shoot, [ / ] render scale, Esc release mouse, R restart.

mod hyper;
mod shader;
mod world;

use hyper::*;
use macroquad::prelude::*;
use macroquad::rand::gen_range;
use shader::{MAX_ENEMIES, MAX_PROJECTILES};
use world::*;

const PLAYER_R: f64 = 0.05;
const PLAYER_SPEED: f64 = 0.9;
const ROLL_SPEED: f64 = 1.6;
const MOUSE_SENS: f64 = 1.6;
const PROJ_SPEED: f64 = 3.5;
const PROJ_R: f64 = 0.03;
const PROJ_LIFE: f64 = 5.0;
const FIRE_COOLDOWN: f64 = 0.16;
const ENEMY_R: f64 = 0.12;
const ENEMY_CHASE_SPEED: f64 = 0.55;
const ENEMY_WANDER_SPEED: f64 = 0.12;
const ENEMY_SIGHT: f64 = 6.0;
const ENEMY_TARGET: usize = 14;
const DESPAWN_DIST: f64 = 7.5;
const FOV: f32 = 1.2;

struct Enemy {
    pos: V4,
    /// Wander direction (unit tangent at pos).
    dir: V4,
    alert: f64,
    sees: bool,
    los_timer: f64,
    flash: f64,
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
    hc: Honeycomb,
    player: Frame,
    /// Label of the central cell. Whenever the player leaves the central cell
    /// the whole world is reflected back so coordinates never blow up.
    label: f64,
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
}

impl Game {
    fn new() -> Self {
        let hc = Honeycomb::new();
        // Start in a cell with no orb at its centre.
        let mut label = 0.0;
        while orb_param(label) > 0.0 {
            label += 1.0;
        }
        // Face an open (or at least pierced) wall so the first view looks out.
        let cp = cell_params(label);
        let fi = (0..NFACES)
            .max_by(|&a, &b| {
                let score = |f: f64| if f < -0.5 { 2.0 } else { f };
                score(cp.face[a]).partial_cmp(&score(cp.face[b])).unwrap()
            })
            .unwrap();
        let fwd = hc.dirs[fi];
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
                [0.0, 0.0, 0.0, 1.0],
            ],
        };
        player.orthonormalize();
        let mut g = Game {
            hc,
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
        };
        for _ in 0..ENEMY_TARGET {
            g.try_spawn_enemy(2.0, 5.0);
        }
        g
    }

    fn random_dir(p: V4) -> V4 {
        loop {
            let v = [gen_range(-1.0, 1.0), gen_range(-1.0, 1.0), gen_range(-1.0, 1.0)];
            let l2: f64 = v.iter().map(|x| x * x).sum();
            if l2 > 0.01 && l2 <= 1.0 {
                return normalize_tangent(p, [v[0], v[1], v[2], 0.0]);
            }
        }
    }

    fn try_spawn_enemy(&mut self, dmin: f64, dmax: f64) {
        let pp = self.player.pos();
        for _ in 0..20 {
            let dir = Self::random_dir(pp);
            let (pos, _) = geodesic(pp, dir, gen_range(dmin, dmax));
            if self.hc.wall_dist(pos, self.label) > ENEMY_R * 1.6 {
                let dir = Self::random_dir(pos);
                self.enemies.push(Enemy {
                    pos,
                    dir,
                    alert: 0.0,
                    sees: false,
                    los_timer: gen_range(0.0, 0.2),
                    flash: 0.0,
                });
                return;
            }
        }
    }

    /// Reflect the whole world so the player is back in the central cell.
    fn recenter(&mut self) {
        while let Some(i) = self.hc.outside_face(self.player.pos()) {
            let n = self.hc.normals[i];
            self.player.map(|v| reflect(v, n));
            for e in &mut self.enemies {
                e.pos = reflect(e.pos, n);
                e.dir = reflect(e.dir, n);
            }
            for p in &mut self.projectiles {
                p.pos = reflect(p.pos, n);
                p.dir = reflect(p.dir, n);
            }
            for b in &mut self.bursts {
                b.pos = reflect(b.pos, n);
            }
            self.label = toggle(self.label, i);
        }
        self.player.orthonormalize();
    }

    /// Push the player out of walls along the SDF gradient.
    fn resolve_player_collision(&mut self) {
        for _ in 0..4 {
            let d = self.hc.wall_dist(self.player.pos(), self.label);
            if d >= PLAYER_R {
                return;
            }
            let h = 1e-3;
            let mut g = [0.0; 3];
            for (k, gk) in g.iter_mut().enumerate() {
                let mut dir = [0.0; 3];
                dir[k] = 1.0;
                let mut a = self.player;
                a.translate(dir, h);
                let mut b = self.player;
                b.translate(dir, -h);
                *gk = (self.hc.wall_dist(a.pos(), self.label)
                    - self.hc.wall_dist(b.pos(), self.label))
                    / (2.0 * h);
            }
            let l = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
            if l < 1e-9 {
                return;
            }
            self.player
                .translate([g[0] / l, g[1] / l, g[2] / l], PLAYER_R - d + 1e-4);
        }
    }

    fn update(&mut self, dt: f64) {
        self.time += dt;
        self.hurt = (self.hurt - dt * 2.0).max(0.0);
        for b in &mut self.bursts {
            b.age += dt;
        }
        self.bursts.retain(|b| b.age < 0.5);

        if self.dead {
            if is_key_pressed(KeyCode::R) {
                *self = Game::new();
            }
            return;
        }

        // --- Look & roll -------------------------------------------------
        if is_mouse_button_pressed(MouseButton::Left) {
            set_cursor_grab(true);
            show_mouse(false);
        }
        if is_key_pressed(KeyCode::Escape) {
            set_cursor_grab(false);
            show_mouse(true);
        }
        let md = mouse_delta_position();
        let yaw = -md.x as f64 * MOUSE_SENS;
        let pitch = md.y as f64 * MOUSE_SENS;
        // Turning right moves forward (2) towards right (0).
        self.player.rotate(2, 0, yaw);
        self.player.rotate(2, 1, pitch);
        let mut roll = 0.0;
        if is_key_down(KeyCode::Q) {
            roll -= 1.0;
        }
        if is_key_down(KeyCode::E) {
            roll += 1.0;
        }
        self.player.rotate(0, 1, roll * ROLL_SPEED * dt);

        // --- Move --------------------------------------------------------
        let mut mv = [0.0f64; 3];
        if is_key_down(KeyCode::W) {
            mv[2] += 1.0;
        }
        if is_key_down(KeyCode::S) {
            mv[2] -= 1.0;
        }
        if is_key_down(KeyCode::D) {
            mv[0] += 1.0;
        }
        if is_key_down(KeyCode::A) {
            mv[0] -= 1.0;
        }
        if is_key_down(KeyCode::Space) {
            mv[1] += 1.0;
        }
        if is_key_down(KeyCode::LeftControl) || is_key_down(KeyCode::C) {
            mv[1] -= 1.0;
        }
        let l = (mv[0] * mv[0] + mv[1] * mv[1] + mv[2] * mv[2]).sqrt();
        if l > 0.0 {
            let boost = if is_key_down(KeyCode::LeftShift) { 1.8 } else { 1.0 };
            let step = PLAYER_SPEED * boost * dt;
            // Sub-step so fast movement can't tunnel through thin plates.
            let n = (step / 0.02).ceil().max(1.0) as usize;
            for _ in 0..n {
                self.player
                    .translate([mv[0] / l, mv[1] / l, mv[2] / l], step / n as f64);
                self.resolve_player_collision();
            }
            self.travelled += step;
        }
        self.player.orthonormalize();
        self.recenter();

        // --- Shoot -------------------------------------------------------
        self.cooldown -= dt;
        let grabbed = is_mouse_button_down(MouseButton::Left);
        if grabbed && self.cooldown <= 0.0 && self.projectiles.len() < MAX_PROJECTILES {
            self.cooldown = FIRE_COOLDOWN;
            let p = self.player.pos();
            let v = self.player.c[2];
            let (pos, dir) = geodesic(p, v, PLAYER_R * 2.5);
            self.projectiles.push(Projectile { pos, dir, life: PROJ_LIFE });
        }

        self.update_projectiles(dt);
        self.update_enemies(dt);

        // --- Population management ---------------------------------------
        let pp = self.player.pos();
        self.enemies.retain(|e| dist(e.pos, pp) < DESPAWN_DIST);
        self.spawn_timer -= dt;
        if self.spawn_timer <= 0.0 && self.enemies.len() < ENEMY_TARGET.min(MAX_ENEMIES) {
            self.spawn_timer = 0.7;
            self.try_spawn_enemy(2.5, 5.0);
        }

        if self.health <= 0.0 {
            self.health = 0.0;
            self.dead = true;
            set_cursor_grab(false);
            show_mouse(true);
        }
    }

    fn update_projectiles(&mut self, dt: f64) {
        let mut i = 0;
        while i < self.projectiles.len() {
            let mut alive = true;
            let mut remaining = PROJ_SPEED * dt;
            {
                let pr = &mut self.projectiles[i];
                pr.life -= dt;
                if pr.life <= 0.0 {
                    alive = false;
                }
            }
            // Advance along the geodesic in SDF-bounded sub-steps so we never
            // skip through a wall or an enemy.
            while alive && remaining > 0.0 {
                let (pos, dir) = (self.projectiles[i].pos, self.projectiles[i].dir);
                let dw = self.hc.wall_dist(pos, self.label) - PROJ_R;
                if dw <= 0.0 {
                    self.bursts.push(Burst { pos, age: 0.0 });
                    alive = false;
                    break;
                }
                let mut de = f64::INFINITY;
                let mut hit_enemy = None;
                for (k, e) in self.enemies.iter().enumerate() {
                    let d = dist(pos, e.pos) - ENEMY_R - PROJ_R;
                    if d < de {
                        de = d;
                        hit_enemy = Some(k);
                    }
                }
                if de <= 0.0 {
                    let k = hit_enemy.unwrap();
                    let e = self.enemies.swap_remove(k);
                    self.bursts.push(Burst { pos: e.pos, age: 0.0 });
                    self.score += 1;
                    alive = false;
                    break;
                }
                let step = remaining.min(dw.min(de).max(0.004));
                let (np, nd) = geodesic(pos, dir, step);
                self.projectiles[i].pos = np;
                self.projectiles[i].dir = nd;
                remaining -= step;
            }
            if alive && dist(self.projectiles[i].pos, self.player.pos()) > DESPAWN_DIST + 1.0 {
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
        let hc = &self.hc;
        let mut damage = 0.0;
        self.enemies.retain_mut(|e| {
            e.flash = (e.flash - dt * 3.0).max(0.0);
            e.los_timer -= dt;
            let d = dist(e.pos, pp);
            if e.los_timer <= 0.0 {
                e.los_timer = 0.15;
                // Line of sight: march the geodesic from the enemy to the
                // player; any wall in between blocks the view.
                e.sees = d < ENEMY_SIGHT && {
                    let v = dir_to(e.pos, pp);
                    let start = ENEMY_R * 0.5;
                    let (q, w) = geodesic(e.pos, v, start);
                    hc.march(q, w, (d - start - PLAYER_R).max(0.0), 0.01, label).is_none()
                };
            }
            let target_alert = if e.sees { 1.0 } else { 0.0 };
            e.alert += (target_alert - e.alert) * (dt * 5.0).min(1.0);

            if e.sees {
                // Move straight at the player along the connecting geodesic.
                let v = dir_to(e.pos, pp);
                let (np, _) = geodesic(e.pos, v, ENEMY_CHASE_SPEED * dt);
                // The sight line is thin; don't let the body clip through a
                // beam it grazes on the way.
                if hc.wall_dist(np, label) > ENEMY_R * 0.8 {
                    e.pos = np;
                }
                e.dir = dir_to(e.pos, pp);
            } else {
                // Idle drift; bounce off walls.
                let (np, nd) = geodesic(e.pos, e.dir, ENEMY_WANDER_SPEED * dt);
                if hc.wall_dist(np, label) > ENEMY_R * 1.1 {
                    e.pos = np;
                    e.dir = nd;
                } else {
                    e.dir = Game::random_dir(e.pos);
                }
            }
            if dist(e.pos, pp) < ENEMY_R + PLAYER_R {
                damage += 20.0;
                return false;
            }
            true
        });
        if damage > 0.0 {
            self.health -= damage;
            self.hurt = 1.0;
        }
    }
}

fn window_conf() -> Conf {
    Conf {
        window_title: "Curvemaster — H3".to_owned(),
        window_width: 1280,
        window_height: 720,
        high_dpi: false,
        ..Default::default()
    }
}

fn v4f(v: V4) -> Vec4 {
    vec4(v[0] as f32, v[1] as f32, v[2] as f32, v[3] as f32)
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut game = Game::new();
    let frag = shader::fragment(&game.hc);
    let material = load_material(
        ShaderSource::Glsl {
            vertex: shader::VERTEX,
            fragment: &frag,
        },
        MaterialParams {
            uniforms: vec![
                UniformDesc::new("uRes", UniformType::Float2),
                UniformDesc::new("uFrame", UniformType::Mat4),
                UniformDesc::new("uLabel", UniformType::Float1),
                UniformDesc::new("uTime", UniformType::Float1),
                UniformDesc::new("uFov", UniformType::Float1),
                UniformDesc::new("uEnemy", UniformType::Float4).array(MAX_ENEMIES),
                UniformDesc::new("uEnemyInfo", UniformType::Float4).array(MAX_ENEMIES),
                UniformDesc::new("uNumEnemies", UniformType::Float1),
                UniformDesc::new("uProj", UniformType::Float4).array(MAX_PROJECTILES),
                UniformDesc::new("uNumProj", UniformType::Float1),
            ],
            ..Default::default()
        },
    )
    .unwrap_or_else(|e| panic!("shader compile failed: {e:?}"));

    let mut render_scale: f32 = 0.5;
    let mut rt: Option<(RenderTarget, u32, u32)> = None;
    set_cursor_grab(true);
    show_mouse(false);

    // Headless capture mode for screenshots: CURVEMASTER_SHOT=path.png
    let shot = std::env::var("CURVEMASTER_SHOT").ok();
    let mut frame_no = 0u32;

    loop {
        let dt = get_frame_time().min(0.05) as f64;
        if is_key_pressed(KeyCode::LeftBracket) {
            render_scale = (render_scale - 0.1).max(0.2);
        }
        if is_key_pressed(KeyCode::RightBracket) {
            render_scale = (render_scale + 0.1).min(1.0);
        }
        if shot.is_some() {
            // Scripted demo: drift forward slowly and turn.
            let yaw: f64 = std::env::var("CURVEMASTER_YAW").ok().and_then(|v| v.parse().ok()).unwrap_or(0.004);
            let speed: f64 = std::env::var("CURVEMASTER_SPEED").ok().and_then(|v| v.parse().ok()).unwrap_or(0.004);
            game.player.rotate(2, 0, yaw);
            game.player.translate([0.0, 0.0, 1.0], speed);
            game.resolve_player_collision();
            game.recenter();
            if frame_no % 8 == 0 {
                let (pos, dir) = geodesic(game.player.pos(), game.player.c[2], 0.06);
                game.projectiles.push(Projectile { pos, dir, life: PROJ_LIFE });
            }
            game.update_projectiles(1.0 / 60.0);
            game.update_enemies(1.0 / 60.0);
            game.update_enemies(1.0 / 60.0);
        } else {
            game.update(dt);
        }

        let (sw, sh) = (screen_width(), screen_height());
        let w = ((sw * render_scale) as u32).max(64);
        let h = ((sh * render_scale) as u32).max(36);
        if rt.as_ref().map(|r| (r.1, r.2)) != Some((w, h)) {
            let target = render_target(w, h);
            target.texture.set_filter(FilterMode::Linear);
            rt = Some((target, w, h));
        }
        let target = &rt.as_ref().unwrap().0;

        // --- Upload scene uniforms ---------------------------------------
        let cols = game.player.to_f32_cols();
        let m = Mat4::from_cols(
            Vec4::from_array(cols[0]),
            Vec4::from_array(cols[1]),
            Vec4::from_array(cols[2]),
            Vec4::from_array(cols[3]),
        );
        material.set_uniform("uRes", vec2(w as f32, h as f32));
        material.set_uniform("uFrame", m);
        material.set_uniform("uLabel", game.label as f32);
        material.set_uniform("uTime", game.time as f32);
        material.set_uniform("uFov", FOV);
        let mut ep = [Vec4::ZERO; MAX_ENEMIES];
        let mut ei = [Vec4::ZERO; MAX_ENEMIES];
        let mut ne = 0;
        for e in game.enemies.iter().take(MAX_ENEMIES) {
            ep[ne] = v4f(e.pos);
            ei[ne] = vec4(ENEMY_R as f32, e.alert as f32, e.flash as f32, 0.0);
            ne += 1;
        }
        // Explosion bursts render as expanding, fading enemy-coloured balls.
        for b in &game.bursts {
            if ne >= MAX_ENEMIES {
                break;
            }
            let k = (b.age / 0.5) as f32;
            ep[ne] = v4f(b.pos);
            ei[ne] = vec4(0.04 + 0.18 * k, 1.0, 1.0 - k, 0.0);
            ne += 1;
        }
        let mut pp = [Vec4::ZERO; MAX_PROJECTILES];
        for (i, p) in game.projectiles.iter().take(MAX_PROJECTILES).enumerate() {
            pp[i] = v4f(p.pos);
        }
        material.set_uniform_array("uEnemy", &ep[..]);
        material.set_uniform_array("uEnemyInfo", &ei[..]);
        material.set_uniform("uNumEnemies", ne as f32);
        material.set_uniform_array("uProj", &pp[..]);
        material.set_uniform("uNumProj", game.projectiles.len().min(MAX_PROJECTILES) as f32);

        // --- Raymarch into the low-res target ----------------------------
        set_camera(&Camera2D {
            render_target: Some(target.clone()),
            ..Camera2D::from_display_rect(Rect::new(0.0, 0.0, w as f32, h as f32))
        });
        clear_background(BLACK);
        gl_use_material(&material);
        draw_rectangle(0.0, 0.0, w as f32, h as f32, WHITE);
        gl_use_default_material();

        // --- Present + HUD -----------------------------------------------
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
        draw_hud(&game, render_scale);

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

    // Health bar
    let bw = 220.0;
    draw_rectangle(20.0, sh - 40.0, bw, 16.0, Color::new(0.0, 0.0, 0.0, 0.5));
    let hf = (game.health / 100.0) as f32;
    draw_rectangle(
        20.0,
        sh - 40.0,
        bw * hf,
        16.0,
        Color::new(1.0 - hf, 0.3 + 0.6 * hf, 0.3, 0.9),
    );
    draw_rectangle_lines(20.0, sh - 40.0, bw, 16.0, 1.5, WHITE);

    let seeing = game.enemies.iter().filter(|e| e.sees).count();
    let lines = [
        format!("SCORE {}", game.score),
        format!("cell #{:04}   travelled {:.1}", game.label as u32, game.travelled),
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
    draw_text(
        "WASD move  mouse look  Q/E roll  Space/Ctrl up/down  LMB shoot  [ ] res  Esc free mouse",
        20.0,
        sh - 55.0,
        18.0,
        Color::new(1.0, 1.0, 1.0, 0.6),
    );

    if game.dead {
        draw_rectangle(0.0, 0.0, sw, sh, Color::new(0.0, 0.0, 0.0, 0.55));
        let t = "LOST IN THE HYPERBOLIC VOID";
        let d = measure_text(t, None, 48, 1.0);
        draw_text(t, cx - d.width / 2.0, cy - 20.0, 48.0, RED);
        let t2 = format!("score {}  -  press R to restart", game.score);
        let d2 = measure_text(&t2, None, 28, 1.0);
        draw_text(&t2, cx - d2.width / 2.0, cy + 20.0, 28.0, WHITE);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projectile_flies_until_it_hits_a_wall() {
        let mut g = Game::new();
        g.enemies.clear();
        let p = g.player.pos();
        let v = g.player.c[2];
        g.projectiles.push(Projectile { pos: p, dir: v, life: 100.0 });
        let mut steps = 0;
        while !g.projectiles.is_empty() && steps < 10000 {
            g.update_projectiles(1.0 / 60.0);
            steps += 1;
        }
        assert!(g.projectiles.is_empty(), "projectile never collided");
        assert_eq!(g.bursts.len(), 1);
        let b = g.bursts[0].pos;
        assert!(g.hc.wall_dist(b, g.label) < PROJ_R + 1e-3);
        // It travelled along a geodesic: distance equals path length.
        let travelled = steps as f64 / 60.0 * PROJ_SPEED;
        assert!(dist(b, p) <= travelled + 1e-6);
        assert!(dist(b, p) > travelled - PROJ_SPEED / 60.0 - 1e-6);
    }

    #[test]
    fn projectile_kills_enemy() {
        let mut g = Game::new();
        g.enemies.clear();
        let p = g.player.pos();
        let v = g.player.c[2];
        let (ep, _) = geodesic(p, v, 0.4);
        g.enemies.push(Enemy { pos: ep, dir: v, alert: 0.0, sees: false, los_timer: 9.0, flash: 0.0 });
        g.projectiles.push(Projectile { pos: p, dir: v, life: 5.0 });
        for _ in 0..60 {
            g.update_projectiles(1.0 / 60.0);
        }
        assert!(g.enemies.is_empty());
        assert_eq!(g.score, 1);
    }

    #[test]
    fn visible_enemy_chases_player() {
        let mut g = Game::new();
        g.enemies.clear();
        let p = g.player.pos();
        // The start pose looks through the most open face.
        let v = g.player.c[2];
        let (ep, _) = geodesic(p, v, 0.6);
        g.enemies.push(Enemy { pos: ep, dir: v, alert: 0.0, sees: false, los_timer: 0.0, flash: 0.0 });
        let d0 = dist(ep, p);
        g.update_enemies(0.1);
        assert!(g.enemies[0].sees);
        assert!(dist(g.enemies[0].pos, p) < d0);
    }

    #[test]
    fn walls_block_sight() {
        let g = Game::new();
        let hc = &g.hc;
        // A point on the far side of a solid face plate cannot be seen.
        let cp = cell_params(g.label);
        if let Some(i) = (0..NFACES).find(|&i| cp.face[i] == 0.0) {
            let u = hc.dirs[i];
            let v = [u[0], u[1], u[2], 0.0];
            let (behind, _) = geodesic(ORIGIN, v, 1.2);
            let d = dist(ORIGIN, behind);
            assert!(hc.march(ORIGIN, v, d, 0.01, g.label).is_some());
        }
    }
}
