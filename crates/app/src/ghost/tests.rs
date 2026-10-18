use super::*;
use glam::Vec3;
use sim::sampling::GhostFinished;
use std::time::Duration;

fn atom(z: u8, pos: [f64; 3]) -> GhostAtom {
    GhostAtom { z, pos }
}

fn cam() -> CamBasis {
    CamBasis::of(&render::camera::OrbitCamera::new(Vec3::ZERO, 10.0))
}

/// A synthetic record of the job `job` over `atoms`.
fn ghost(stage: &'static str, job: GhostJob, atoms: Vec<GhostAtom>) -> PassGhost {
    PassGhost { stage, job, base: atoms.iter().map(|a| a.pos).collect(), atoms, trail: Vec::new(), finished: None }
}

fn water_scan(ratio: f64, r: f64) -> PassGhost {
    let atoms = vec![atom(8, [0.0, 0.0, 0.0]), atom(1, [r, 0.0, 0.0]), atom(1, [-0.5, 1.7, 0.0])];
    ghost("scans", GhostJob::Scan { i: 0, j: 1, k: 9, of: 12, ratio, r }, atoms)
}

fn dist(p: [f32; 3], c: Vec3) -> f32 {
    (Vec3::from_array(p) - c).length()
}

/// The label and the stage line of each kind of job, from synthetic records.
#[test]
fn labels_per_stage() {
    let r = ghost("references", GhostJob::Reference { z: 8 }, vec![atom(8, [0.0; 3])]);
    assert_eq!(label(&r), "reference: isolated O");
    assert_eq!(stage_line(&r), "reference");
    let p = ghost("perturbations", GhostJob::Perturbation { index: 3, of: 12 }, vec![atom(1, [0.0; 3]), atom(1, [1.4, 0.0, 0.0])]);
    assert_eq!(label(&p), "random nudge 3 of 12");
    assert_eq!(stage_line(&p), "random nudges");
    let s = water_scan(1.5, 2.7123);
    assert_eq!(label(&s), "O–H · 1.50 × r₀ · 2.71 Bohr");
    assert_eq!(stage_line(&s), "bond scan · point 9 of 12");
    let b = ghost("hot burst", GhostJob::HotBurst { step: 17, of: 40 }, vec![atom(1, [0.0; 3])]);
    assert_eq!(label(&b), "hot MD burst · step 17 of 40");
    assert_eq!(stage_line(&b), "hot MD burst");
    // a scan names the pair by the ghost's own atoms, whichever way round
    let h = ghost("scans", GhostJob::Scan { i: 1, j: 2, k: 1, of: 5, ratio: 0.7, r: 1.0 }, water_scan(1.0, 1.8).atoms);
    assert_eq!(label(&h), "H–H · 0.70 × r₀ · 1.00 Bohr");
}

/// Green kept, red failed (stalled, the cap or otherwise), grey skipped.
#[test]
fn flash_colours_per_outcome() {
    use sim::activity::ScfEnd;
    assert_eq!(outcome_colour(GhostOutcome::Kept), KEPT);
    for o in [GhostOutcome::Failed(None), GhostOutcome::Failed(Some(ScfEnd::Stalled)), GhostOutcome::Failed(Some(ScfEnd::Cap))] {
        assert_eq!(outcome_colour(o), FAILED);
    }
    assert_eq!(outcome_colour(GhostOutcome::Skipped), SKIPPED);
    assert!(KEPT[1] > KEPT[0] && KEPT[1] > KEPT[2], "green");
    assert!(FAILED[0] > 3.0 * FAILED[1] && FAILED[0] > 3.0 * FAILED[2], "red");
    assert!((SKIPPED[0] - SKIPPED[1]).abs() < 1e-6 && (SKIPPED[1] - SKIPPED[2]).abs() < 1e-6, "grey");
}

/// The flash fades linearly from 1 to 0 over `FLASH_SECS` and is gone after.
#[test]
fn the_flash_expires() {
    assert_eq!(flash_alpha(0.0), 1.0);
    assert!((flash_alpha(FLASH_SECS / 2.0) - 0.5).abs() < 1e-6);
    assert_eq!(flash_alpha(FLASH_SECS), 0.0);
    assert_eq!(flash_alpha(FLASH_SECS + 3.0), 0.0);
    assert_eq!(flash_alpha(-1.0), 0.0, "never from the future");
    assert_eq!(FLASH_SECS, 0.5);
}

fn finished(serial: u64, outcome: GhostOutcome) -> GhostFinished {
    GhostFinished { serial, outcome, atoms: vec![atom(1, [serial as f64, 0.0, 0.0])] }
}

/// A new finished serial starts a flash of its outcome and geometry; the same serial does not
/// restart it; it expires; no ghost (the pass ended) drops it and the next pass's serial 1 flashes
/// again.
#[test]
fn the_tracker_flashes_each_finished_job_once() {
    let t0 = Instant::now();
    let mut g = ghost("perturbations", GhostJob::Perturbation { index: 2, of: 4 }, vec![atom(1, [0.0; 3])]);
    let mut tr = Tracker::default();
    tr.follow(Some(&g), t0);
    assert!(tr.flash(t0).is_none(), "nothing finished yet");
    g.finished = Some(finished(1, GhostOutcome::Kept));
    tr.follow(Some(&g), t0);
    let (f, a) = tr.flash(t0).unwrap();
    assert_eq!((f.outcome, f.atoms.clone(), a), (GhostOutcome::Kept, finished(1, GhostOutcome::Kept).atoms, 1.0));
    // 0.2 s later: fading; a repeat snapshot of the same serial does not restart it
    let t1 = t0 + Duration::from_millis(200);
    tr.follow(Some(&g), t1);
    let (_, a) = tr.flash(t1).unwrap();
    assert!((a - 0.6).abs() < 1e-3, "{a}");
    assert!(tr.animating(t1));
    // expired
    let t2 = t0 + Duration::from_millis(700);
    assert!(tr.flash(t2).is_none() && !tr.animating(t2));
    // the next job fails: a red flash from t2
    g.finished = Some(finished(2, GhostOutcome::Failed(None)));
    tr.follow(Some(&g), t2);
    let (f, a) = tr.flash(t2).unwrap();
    assert_eq!((f.outcome, a), (GhostOutcome::Failed(None), 1.0));
    // the pass ends: the flash goes with the ghost, and a new pass starts counting at 1 again
    tr.follow(None, t2);
    assert!(tr.flash(t2).is_none());
    g.finished = Some(finished(1, GhostOutcome::Skipped));
    tr.follow(Some(&g), t2);
    assert_eq!(tr.flash(t2).unwrap().0.outcome, GhostOutcome::Skipped);
}

/// A lone reference atom: every line vertex lies on the atom's display sphere, in its element
/// colour (linear, `render::elements::colour`).
#[test]
fn a_reference_is_an_outlined_sphere_in_its_element_colour() {
    let g = ghost("references", GhostJob::Reference { z: 8 }, vec![atom(8, [0.0; 3])]);
    let v = build(&g, &cam(), None, &[]);
    let r = elements::radius(8);
    assert!(!v.is_empty() && v.len() % 2 == 0);
    let c = elements::colour(8);
    for p in &v {
        assert!((dist(p.pos, Vec3::ZERO) - r).abs() < 1e-4, "{:?}", p.pos);
        assert_eq!(&p.colour[..3], &c[..]);
    }
    // the silhouette faces the camera; the great circles are fainter
    assert!(v.iter().any(|p| p.colour[3] > 0.9) && v.iter().any(|p| p.colour[3] < 0.5));
}

/// A nudge draws an arrow from each atom's unperturbed position to its nudged one — none for an
/// atom that did not move.
#[test]
fn a_nudge_has_arrows_from_the_unperturbed_positions() {
    let mut g = ghost("perturbations", GhostJob::Perturbation { index: 1, of: 4 }, vec![atom(1, [0.1, 0.0, 0.0]), atom(1, [1.4, 0.0, 0.0])]);
    g.base = vec![[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]];
    let v = build(&g, &cam(), None, &[]);
    let arrows: Vec<_> = v.chunks(2).filter(|s| s[0].colour == ARROW).collect();
    // shaft + two head strokes for the first atom only
    assert_eq!(arrows.len(), 3);
    assert_eq!(arrows[0][0].pos, [0.0, 0.0, 0.0]);
    assert_eq!(arrows[0][1].pos, [0.1, 0.0, 0.0]);
    assert!(arrows[1..].iter().all(|s| s[0].pos == [0.1, 0.0, 0.0] && dist(s[1].pos, Vec3::new(0.1, 0.0, 0.0)) < 0.12));
    // other stages draw none
    let mut s = water_scan(1.0, 1.8);
    s.base[1] = [2.0, 0.0, 0.0];
    assert!(build(&s, &cam(), None, &[]).iter().all(|p| p.colour != ARROW));
}

/// The scanned pair is brighter and thicker: a bundle of bright strands between the spheres'
/// surfaces and a thicker silhouette than the third atom's.
#[test]
fn the_scanned_pair_is_highlighted() {
    let g = water_scan(1.5, 2.7);
    let v = build(&g, &cam(), None, &[]);
    let bright: Vec<_> = v.chunks(2).filter(|s| s[0].colour == PAIR_LINK).collect();
    assert_eq!(bright.len(), PAIR_STRANDS);
    // from O's surface to H's (the bundle's strands are within PAIR_RADIUS of the axis)
    let (ro, rh) = (elements::radius(8), elements::radius(1));
    for s in &bright {
        assert!((s[0].pos[0] - ro).abs() < 0.01 + PAIR_RADIUS && (s[1].pos[0] - (2.7 - rh)).abs() < 0.01 + PAIR_RADIUS, "{s:?}");
    }
    // vertices on the pair's atoms vs the bystander H: the pair's outlines have three silhouettes
    let on = |c: Vec3, rad: f32| v.iter().filter(|p| (dist(p.pos, c) - rad).abs() < 0.2 * rad && p.colour != PAIR_LINK && p.colour != LINK).count();
    let bystander = on(Vec3::new(-0.5, 1.7, 0.0), rh);
    let pair_h = on(Vec3::new(2.7, 0.0, 0.0), rh);
    assert!(pair_h > bystander, "pair H {pair_h} vs bystander {bystander}");
    // a scan always shows its pair link, even stretched past the thin-link distance
    assert!(2.7 > LINK_FACTOR * (ro + rh) * 0.9);
    assert!(links(&g.atoms, Some((0, 1))).contains(&(0, 1)));
}

/// The thin links: H₂ at its own bond length is linked, two H apart are not; the scanned pair is
/// always linked.
#[test]
fn thin_links_follow_the_display_radii() {
    let h2 = [atom(1, [0.0; 3]), atom(1, [1.45, 0.0, 0.0])];
    assert_eq!(links(&h2, None), vec![(0, 1)]);
    let far = [atom(1, [0.0; 3]), atom(1, [4.0, 0.0, 0.0])];
    assert!(links(&far, None).is_empty());
    assert_eq!(links(&far, Some((0, 1))), vec![(0, 1)]);
    // water's O–H at 1.81 Bohr is linked, its H…H at 2.9 is not
    let w = [atom(8, [0.0; 3]), atom(1, [1.81, 0.0, 0.0]), atom(1, [-0.45, 1.75, 0.0])];
    let l = links(&w, None);
    assert!(l.contains(&(0, 1)) && l.contains(&(0, 2)) && !l.contains(&(1, 2)), "{l:?}");
}

/// A hot burst keeps a fading trail of the last solved geometries: segments through each atom's
/// earlier positions to its current one, brighter towards the present.
#[test]
fn a_burst_draws_a_fading_trail() {
    let mut g = ghost("hot burst", GhostJob::HotBurst { step: 5, of: 20 }, vec![atom(1, [0.4, 0.0, 0.0])]);
    g.trail = vec![vec![[0.0, 0.0, 0.0]], vec![[0.1, 0.0, 0.0]], vec![[0.2, 0.0, 0.0]], vec![[0.3, 0.0, 0.0]]];
    let v = build(&g, &cam(), None, &[]);
    let c = elements::colour(1);
    // the straight segments along the x axis of the trail: z = 0 and y = 0 at both ends and not on the sphere
    let trail: Vec<_> = v.chunks(2).filter(|s| s[0].pos[1] == 0.0 && s[1].pos[1] == 0.0 && s[0].pos[2] == 0.0 && s[1].pos[2] == 0.0 && (s[1].pos[0] - s[0].pos[0] - 0.1).abs() < 1e-5).collect();
    assert_eq!(trail.len(), 4, "four segments through five positions");
    assert!(trail.windows(2).all(|w| w[1][0].colour[3] > w[0][0].colour[3]), "brighter towards the present");
    assert!(trail.iter().all(|s| s[0].colour[..3] == c[..] && s[0].colour[3] > 0.0 && s[0].colour[3] < 0.8));
    assert_eq!(trail[3][1].pos, [0.4, 0.0, 0.0]);
    // no trail on the first step
    g.trail.clear();
    assert!(build(&g, &cam(), None, &[]).len() < v.len());
}

/// While a flash fades the finished geometry is outlined once more in the outcome's colour at the
/// flash's strength, on top of the live ghost.
#[test]
fn the_flash_outlines_the_finished_geometry() {
    let g = ghost("perturbations", GhostJob::Perturbation { index: 2, of: 4 }, vec![atom(1, [0.0; 3])]);
    let plain = build(&g, &cam(), None, &[]);
    let f = Flash { at: Instant::now(), outcome: GhostOutcome::Failed(None), atoms: vec![atom(1, [1.0, 0.0, 0.0])] };
    let v = build(&g, &cam(), Some((&f, 0.5)), &[]);
    assert!(v.len() > plain.len());
    assert_eq!(&v[..plain.len()], &plain[..], "drawn over the live ghost");
    let extra = &v[plain.len()..];
    assert!(extra.iter().all(|p| p.colour[..3] == FAILED[..] && p.colour[3] <= 0.5 * RING_ALPHA + 1e-6 && p.colour[3] > 0.0));
    assert!(extra.iter().all(|p| dist(p.pos, Vec3::new(1.0, 0.0, 0.0)) > elements::radius(1) * 0.99));
    // a stronger flash is brighter
    let strong = build(&g, &cam(), Some((&f, 1.0)), &[]);
    assert!(strong[plain.len()].colour[3] > extra[0].colour[3]);
}

#[test]
fn the_camera_basis_is_orthonormal() {
    let mut c = render::camera::OrbitCamera::new(Vec3::new(0.3, -0.2, 0.1), 8.0);
    c.yaw = 0.7;
    c.pitch = 0.4;
    let b = CamBasis::of(&c);
    assert!((b.right.length() - 1.0).abs() < 1e-5 && (b.up.length() - 1.0).abs() < 1e-5 && b.right.dot(b.up).abs() < 1e-5);
}

/// User request 2026-10-18: the last three finished samples stay on screen behind the current one,
/// newest first, gradually more transparent, in their outcome colours; the pass's end clears them.
#[test]
fn the_last_three_samples_stay_and_fade() {
    let t0 = Instant::now();
    let mut g = ghost("perturbations", GhostJob::Perturbation { index: 2, of: 9 }, vec![atom(1, [0.0; 3])]);
    let mut tr = Tracker::default();
    for (k, o) in [GhostOutcome::Kept, GhostOutcome::Failed(None), GhostOutcome::Kept, GhostOutcome::Skipped].into_iter().enumerate() {
        g.finished = Some(finished(k as u64 + 1, o));
        tr.follow(Some(&g), t0);
    }
    let h = tr.history();
    assert_eq!(h.len(), HISTORY);
    assert_eq!(h.iter().map(|(f, a)| (f.outcome, *a)).collect::<Vec<_>>(), vec![
        (GhostOutcome::Skipped, HISTORY_ALPHA[0]),
        (GhostOutcome::Kept, HISTORY_ALPHA[1]),
        (GhostOutcome::Failed(None), HISTORY_ALPHA[2]),
    ]);
    assert!(HISTORY_ALPHA.windows(2).all(|w| w[0] > w[1]), "older is more transparent");
    // drawn: more vertices than without, in the outcome colours at the history alphas
    let plain = build(&g, &cam(), None, &[]);
    let with = build(&g, &cam(), None, &h);
    assert!(with.len() > plain.len());
    tr.follow(None, t0);
    assert!(tr.history().is_empty(), "the pass ended");
}
