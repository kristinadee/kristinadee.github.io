# Site structure

```
index.html            Home — positioning statement + 3 featured projects
projects.html         All projects, with tag filtering
about.html            Bio, skills, contact
style.css             All styling. Edit the :root tokens at the top to restyle everything.
projects/
  _template.html      Copy this to make a new project page. Not linked from anywhere.
  tesla-valve.html    Fully worked example — use it as the reference for tone and depth.
  uterine-fea.html    Stub, needs writing
  robotics-studio.html
  rl-hvac.html
  math-mouse.html
  fea-studies.html
  chladni.html
assets/
  placeholder.svg     Shows automatically wherever a real image is missing
  <project-slug>/     One folder per project
  docs/               resume.pdf and full project reports
```

## Adding a project

1. `cp projects/_template.html projects/my-project.html`
2. Replace every ALL-CAPS placeholder and every `PROJECT-SLUG` in the image paths.
3. Create `assets/my-project/` and add the images.
4. Copy a `<li>` block in `projects.html`, updating the link, thumbnail, blurb, and
   `data-tags`. The `data-tags` values must match the `data-filter` values on the
   filter buttons, or the project won't appear when that filter is active.
5. If it's a top-three project, add it to `index.html` too.

## File naming

Lowercase, hyphens, no spaces, no dates in the name:
`streamlines-reverse.png`, not `Screenshot 2026-09-19 at 6.40.18 PM.png`.

## Image specs

| Use | Width | Target size | Format |
|---|---|---|---|
| Thumbnail (`thumb.jpg`) | 600 px | under 80 KB | JPG |
| Hero and body figures | 1600 px | under 300 KB | JPG for photos and renders, PNG for plots and line art |
| Video | 1280 px | under 10 MB, 10–20 s | MP4 (H.264) |

Compress with squoosh.app before committing. GitHub rejects files over 100 MB, and
Pages sites should stay under about 1 GB. For longer video, upload unlisted to YouTube
and embed the iframe instead.

Every project folder wants, at minimum: `thumb.jpg`, `hero.jpg`, and one method figure.

## Asset checklist by project

**uterine-fea** — Paraview render of patient geometry (hero); image-to-mesh pipeline diagram
(needs making, highest-value missing asset); stress/strain contours across the cohort;
FEBio vs. svMultPhys comparison plot; deformation animation MP4; cohort table.
Clear with your PI before publishing anything.

**robotics-studio** — CAD assembly render; exploded view; photo of the built hardware;
video of it moving (most important single asset); system block diagram; design iteration
sequence.

**rl-hvac** — the four training/test plots you already have; agent–FMU architecture diagram
(needs making); reward function; comparison against a rule-based baseline (needs running);
photo or diagram of the testbed.

**tesla-valve** — streamlines forward and reverse; mesh; pressure contours; diodicity vs. Re;
literature comparison plot; report PDF. Complete already.

**math-mouse** — exploded render; photo of the printed prototype; video of the mechanism
computing 3 + 2; section view; BOM.

**fea-studies** — notched bar mesh convergence plot; force fit and contact stress plots,
cropped free of SolidWorks UI chrome; a statement of loads and boundary conditions for each.

**chladni** — nodal pattern photos; FFT spectra; theory vs. measured table; video of sand
migrating into a pattern during a frequency sweep.

## Still to add

Steinway Tower (retitle it — the report is analytical, not CFD), bike-device attachment,
bi-pedal robot femur, seaplane, dual-axis solar tracker, machining car jack, materials and
manufacturing car, cryotherapy heat transfer.
