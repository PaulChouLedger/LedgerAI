// ============================================================
// CORTEX CAGE — Open lattice exoskeleton for AGX Orin 64GB
// ============================================================
// AGX Dev Kit: 110mm x 110mm x 71.65mm (board + heatsink)
// Mounting holes: 110mm x 110mm pattern, M3
//
// Design: 4 tapered corner pillars, 3 horizontal ring bands,
// diamond lattice infill panels, open top, LED channel.
// Print in copper PLA for sci-fi industrial look.
// ============================================================

$fn = 60;

// AGX dimensions
agx_w = 110;
agx_d = 110;
agx_h = 72;       // board + heatsink + fan

// Enclosure dimensions
clearance = 12;    // gap between AGX and inner wall
wall = 3;          // wall/pillar thickness
inner_w = agx_w + clearance * 2;
inner_d = agx_d + clearance * 2;
outer_w = inner_w + wall * 2;
outer_d = inner_d + wall * 2;
total_h = agx_h + 25;  // 25mm above AGX for airflow

// Pillar dimensions
pillar_w = 12;
pillar_taper = 2;  // narrower at top by this much per side

// Ring band dimensions
band_h = 6;
band_positions = [0, total_h/2 - band_h/2, total_h - band_h];

// Base plate
base_h = 4;

// Lattice
lat_bar = 1.8;     // lattice bar thickness
lat_spacing = 8;   // diamond cell size

// Mounting standoffs
standoff_h = 8;    // raise AGX off base
standoff_od = 7;
standoff_id = 3.2; // M3 clearance

// LED channel (in middle ring band)
led_channel_w = 8;
led_channel_d = 4;

// ── Modules ──

module pillar(x, y) {
    // Tapered pillar — wider at base, narrower at top
    hull() {
        translate([x - pillar_w/2, y - pillar_w/2, 0])
            cube([pillar_w, pillar_w, 0.1]);
        translate([x - (pillar_w - pillar_taper)/2,
                   y - (pillar_w - pillar_taper)/2,
                   total_h])
            cube([pillar_w - pillar_taper, pillar_w - pillar_taper, 0.1]);
    }
}

module ring_band(z) {
    difference() {
        // Outer shell
        translate([0, 0, z])
            linear_extrude(band_h)
                offset(r=2) // rounded corners
                    square([outer_w - 4, outer_d - 4], center=true);
        // Inner cutout
        translate([0, 0, z - 0.1])
            linear_extrude(band_h + 0.2)
                offset(r=2)
                    square([inner_w - 4, inner_d - 4], center=true);
    }
}

module lattice_panel(pw, ph, px, py, pz, rot) {
    // Diamond lattice panel
    translate([px, py, pz])
        rotate(rot)
            intersection() {
                cube([pw, wall + 0.1, ph]);
                union() {
                    for (ix = [-5 : lat_spacing : pw + lat_spacing]) {
                        for (iy = [-5 : lat_spacing : ph + lat_spacing]) {
                            // Diamond bars going one way
                            translate([ix, 0, iy])
                                rotate([0, 45, 0])
                                    cube([lat_bar, wall + 0.1, lat_spacing * 1.8]);
                            // Diamond bars going other way
                            translate([ix, 0, iy])
                                rotate([0, -45, 0])
                                    cube([lat_bar, wall + 0.1, lat_spacing * 1.8]);
                        }
                    }
                }
            }
}

module base_plate() {
    difference() {
        // Solid base with rounded corners
        linear_extrude(base_h)
            offset(r=3)
                square([outer_w - 6, outer_d - 6], center=true);

        // Center cutout for airflow from below
        translate([0, 0, -0.1])
            linear_extrude(base_h + 0.2)
                square([agx_w - 20, agx_d - 20], center=true);
    }
}

module standoff(x, y) {
    translate([x, y, base_h]) {
        difference() {
            cylinder(h=standoff_h, d=standoff_od);
            translate([0, 0, -0.1])
                cylinder(h=standoff_h + 0.2, d=standoff_id);
        }
    }
}

module cable_notch(x, y, rot) {
    translate([x, y, 0])
        rotate([0, 0, rot])
            translate([-10, -wall*2, 0])
                cube([20, wall * 4, base_h + standoff_h + 5]);
}

// ── Assembly ──

color("SaddleBrown", 0.85) {
    // Base plate
    base_plate();

    // Corner pillars (positioned at corners of outer shell)
    cx = outer_w/2 - pillar_w/2;
    cy = outer_d/2 - pillar_w/2;
    pillar(cx, cy);
    pillar(-cx, cy);
    pillar(cx, -cy);
    pillar(-cx, -cy);

    // Horizontal ring bands
    for (z = band_positions) {
        ring_band(z);
    }

    // Lattice panels (4 sides, between ring bands)
    // Front and back
    for (side = [0, 180]) {
        for (band_idx = [0 : len(band_positions) - 2]) {
            z_start = band_positions[band_idx] + band_h;
            z_end = band_positions[band_idx + 1];
            panel_h = z_end - z_start;
            if (panel_h > 0) {
                panel_w = outer_w - pillar_w * 2;
                translate([0, 0, 0])
                    rotate([0, 0, side])
                        lattice_panel(
                            panel_w, panel_h,
                            -panel_w/2, outer_d/2 - wall, z_start,
                            [0, 0, 0]
                        );
            }
        }
    }

    // Left and right lattice panels
    for (side = [90, 270]) {
        for (band_idx = [0 : len(band_positions) - 2]) {
            z_start = band_positions[band_idx] + band_h;
            z_end = band_positions[band_idx + 1];
            panel_h = z_end - z_start;
            if (panel_h > 0) {
                panel_w = outer_d - pillar_w * 2;
                rotate([0, 0, side])
                    lattice_panel(
                        panel_w, panel_h,
                        -panel_w/2, outer_w/2 - wall, z_start,
                        [0, 0, 0]
                    );
            }
        }
    }

    // Mounting standoffs (AGX 110x110mm pattern)
    sx = agx_w / 2;
    sy = agx_d / 2;
    standoff(sx, sy);
    standoff(-sx, sy);
    standoff(sx, -sy);
    standoff(-sx, -sy);
}

// ── Ghost AGX (for visualization, comment out for printing) ──
%color("DimGray", 0.3)
    translate([-agx_w/2, -agx_d/2, base_h + standoff_h])
        cube([agx_w, agx_d, agx_h]);

// ── LED channel indicator (ghost) ──
%color("Cyan", 0.5)
    translate([0, 0, band_positions[1] + band_h/2])
        difference() {
            linear_extrude(2)
                offset(r=2)
                    square([outer_w - 2, outer_d - 2], center=true);
            linear_extrude(2)
                offset(r=2)
                    square([outer_w - led_channel_w, outer_d - led_channel_w], center=true);
        }
