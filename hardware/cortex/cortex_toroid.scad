// ============================================================
// CORTEX TOROID — Ring dock for AGX Orin 64GB
// ============================================================
// AGX Dev Kit: 110mm x 110mm x 71.65mm
// Diagonal: ~155.6mm — inner ring must clear this
//
// Design: Thick donut ring, AGX drops into center hole,
// heatsink exposed above, board visible below.
// LED channel on top face. Cable tunnels through ring body.
// Print in silk silver or copper PLA.
// ============================================================

$fn = 120;  // smooth circles

// AGX dimensions
agx_w = 110;
agx_d = 110;
agx_h = 72;
agx_diag = sqrt(agx_w * agx_w + agx_d * agx_d);  // ~155.6mm

// Ring dimensions
inner_r = agx_diag / 2 + 5;   // ~82.8mm — 5mm clearance from corners
outer_r = inner_r + 25;        // ~107.8mm — 25mm thick ring wall
ring_h = 35;                   // ring height

// Mounting tabs (4 tabs extending inward to hold AGX)
tab_w = 14;
tab_depth = 15;   // how far tab extends into the hole
tab_h = 6;        // tab thickness
tab_z = ring_h / 2 - tab_h / 2;  // centered vertically

// Standoff on each tab
standoff_h = 4;
standoff_od = 7;
standoff_id = 3.2;  // M3

// LED channel on top face
led_w = 5;
led_d = 3;
led_inset = 8;  // from outer edge

// Cable tunnels
tunnel_d = 14;   // diameter for ethernet / USB-C
tunnel_z = ring_h / 2;

// Rubber foot recesses
foot_d = 12;
foot_depth = 2;
n_feet = 6;

// ── Modules ──

module ring_body() {
    difference() {
        // Outer cylinder
        cylinder(h=ring_h, r=outer_r);
        // Inner hole
        translate([0, 0, -0.1])
            cylinder(h=ring_h + 0.2, r=inner_r);
    }
}

module mounting_tab(angle) {
    rotate([0, 0, angle]) {
        translate([0, 0, tab_z]) {
            // Tab extending inward
            difference() {
                translate([-tab_w/2, -(inner_r + tab_depth - 2), 0])
                    cube([tab_w, tab_depth, tab_h]);
                // Round the inner edge
                // (skipped for printability)
            }
            // Standoff on top of tab
            // Position at AGX mounting hole (55mm from center on each axis)
            // Tabs at 45/135/225/315 degrees align with corners
            translate([0, -(agx_w/2 * sqrt(2)/sqrt(2)), tab_h]) {
                difference() {
                    cylinder(h=standoff_h, d=standoff_od);
                    translate([0, 0, -0.1])
                        cylinder(h=standoff_h + 0.2, d=standoff_id);
                }
            }
        }
    }
}

module mounting_tab_rect(x, y) {
    // Rectangular tab at specific AGX mounting hole position
    translate([x, y, tab_z]) {
        // Tab platform
        hull() {
            translate([0, 0, 0])
                cylinder(h=tab_h, d=tab_w);
            translate([x > 0 ? -8 : 8, y > 0 ? -8 : 8, 0])
                cylinder(h=tab_h, d=tab_w);
        }
        // Standoff
        translate([0, 0, tab_h])
            difference() {
                cylinder(h=standoff_h, d=standoff_od);
                translate([0, 0, -0.1])
                    cylinder(h=standoff_h + 0.2, d=standoff_id);
            }
    }
}

module led_channel() {
    // Recessed channel on top face for LED strip
    led_r = outer_r - led_inset;
    translate([0, 0, ring_h - led_d])
        difference() {
            cylinder(h=led_d + 0.1, r=led_r);
            translate([0, 0, -0.1])
                cylinder(h=led_d + 0.3, r=led_r - led_w);
        }
}

module cable_tunnel(angle) {
    rotate([0, 0, angle])
        translate([0, outer_r + 1, tunnel_z])
            rotate([0, 90, 90])
                cylinder(h=outer_r - inner_r + 10, d=tunnel_d, center=true);
}

module foot_recess(angle) {
    rotate([0, 0, angle])
        translate([outer_r - foot_d/2 - 3, 0, -0.1])
            cylinder(h=foot_depth + 0.1, d=foot_d);
}

module decorative_grooves() {
    // Subtle horizontal grooves around the outside — 3 lines
    for (z = [ring_h * 0.25, ring_h * 0.5, ring_h * 0.75]) {
        translate([0, 0, z])
            difference() {
                cylinder(h=1, r=outer_r + 0.1);
                cylinder(h=1, r=outer_r - 1.2);
            }
    }
}

module top_bevel() {
    // Chamfer on top outer edge
    translate([0, 0, ring_h - 2])
        difference() {
            cylinder(h=2.1, r=outer_r + 0.1);
            cylinder(h=2.1, r1=outer_r, r2=outer_r - 2);
        }
}

module bottom_bevel() {
    // Chamfer on bottom outer edge
    translate([0, 0, -0.1])
        difference() {
            cylinder(h=2.1, r=outer_r + 0.1);
            cylinder(h=2.1, r1=outer_r - 2, r2=outer_r);
        }
}

// ── Assembly ──

color("Silver", 0.85) {
    difference() {
        union() {
            ring_body();

            // Mounting tabs at 4 AGX corner positions
            // AGX holes at (55, 55), (-55, 55), (-55, -55), (55, -55)
            mounting_tab_rect(agx_w/2, agx_d/2);
            mounting_tab_rect(-agx_w/2, agx_d/2);
            mounting_tab_rect(-agx_w/2, -agx_d/2);
            mounting_tab_rect(agx_w/2, -agx_d/2);
        }

        // LED channel recess
        led_channel();

        // Cable tunnels — ethernet (back) and power (side)
        cable_tunnel(0);      // front — ethernet
        cable_tunnel(180);    // back — power
        cable_tunnel(90);     // side — USB/debug

        // Decorative grooves
        decorative_grooves();

        // Top and bottom bevels
        top_bevel();
        bottom_bevel();

        // Foot recesses on bottom
        for (i = [0 : n_feet - 1]) {
            foot_recess(i * 360 / n_feet);
        }
    }
}

// ── Ghost AGX (for visualization) ──
%color("DimGray", 0.3)
    translate([-agx_w/2, -agx_d/2, tab_z + tab_h + standoff_h])
        cube([agx_w, agx_d, agx_h]);

// ── Ghost LED ring ──
%color("Cyan", 0.5)
    translate([0, 0, ring_h - 1])
        difference() {
            cylinder(h=1, r=outer_r - led_inset);
            cylinder(h=1, r=outer_r - led_inset - led_w);
        }

// ── Dimension annotations ──
// Total outer diameter: ~216mm
// Inner hole diameter: ~166mm
// Ring height: 35mm
// AGX sits ~20mm above ring base, heatsink rises ~52mm above ring top
echo(str("Outer diameter: ", outer_r * 2, "mm"));
echo(str("Inner diameter: ", inner_r * 2, "mm"));
echo(str("Ring height: ", ring_h, "mm"));
echo(str("Ring wall thickness: ", outer_r - inner_r, "mm"));
