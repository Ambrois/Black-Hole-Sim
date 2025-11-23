// TODO
// - make this work
// - add text
// - come up with some debugging views?
//    - like a 2d view of everything
//    - this may have to go in another shader frag tho

// camera
uniform vec3 cam_pos;
uniform vec3 h_vec;
uniform vec3 w_vec;
// camera direction stuff
uniform vec3 bot_left_dir;
uniform float pixel_width;
uniform float pixel_height;
// bh stuff
uniform vec3 bh_center;
uniform float bh_M;
uniform float bh_squared_radius;
uniform float squared_inner_radius;
uniform float squared_outer_radius;
// world bounds
uniform vec3 min_bounds;
uniform vec3 max_bounds;
// ray marching
uniform float step_dist;
uniform int max_steps;

// --------------------- Math Stuff -------------------------- //

const float PI = 3.1415926538;


// -------- Vec3 Manipulation

vec3 to_polar(vec3 cart) {
  // r, theta, phi
  float r = length(cart);
  float theta = atan(length(cart.xy), cart.z);
  float phi = atan(cart.y , cart.x);
  return vec3(r, theta, phi);
}

vec3 to_cartesian(vec3 pol) {
  float r=pol.x, theta=pol.y, phi=pol.z;
  float rsinth = r*sin(theta);
  return vec3(rsinth*cos(phi), rsinth*sin(phi), r*cos(theta));
}

vec3 rotate_to_equator(vec3 v, vec3 cam_pos_pol) {
  // uses pol coords
  // only works if the black hole is in the center
  float r_v = v.x, theta_v = v.y, phi_v = v.z;
  float theta_c = cam_pos_pol.y, phi_c = cam_pos_pol.z;
  return vec3(r_v, theta_v + (0.5*PI - theta_c), phi_v -phi_c);
}

vec3 rotate_from_equator(vec3 v, vec3 cam_pos_pol) {
  // uses pol coords
  float r_v = v.x, theta_v = v.y, phi_v = v.z;
  float theta_c = cam_pos_pol.y, phi_c = cam_pos_pol.z;
  return vec3(r_v, theta_v - (0.5*PI - theta_c), phi_v +phi_c);
}


// -------- Null Geodesic Stuff
//

float b(vec3 ray_pos_pol, vec3 dir_pol_eq, float bh_M) {
  // compute once per ray
  float psi = dir_pol_eq.z;
  float r0 = ray_pos_pol.x;
  float f_r0 = 1. - 2. * bh_M / r0;
  return r0*sin(psi) / max(1e-9, sqrt(f_r0));
}


float initial_sigma(vec3 dir_pol_eq, vec3 ray_pos_pol_eq) {
  // compute once per ray, sigma changes are handled in rk4_step
  float phi_diff = abs(dir_pol_eq.z - ray_pos_pol_eq.z);
  float sigma = (0.5*PI < phi_diff && phi_diff < 0.75*PI)? -1. : 1. ;
  return sigma;
}

vec3 rk4_step(vec3 ray_pos_pol_eq, float b, inout float sigma, inout int rdot_sign, float h) {
  // Stepping forward a light ray on the equatorial plane, but distance h
  // 

  float r0 = ray_pos_pol_eq.x,
        theta0 = ray_pos_pol_eq.y,
        phi0 = ray_pos_pol_eq.z;

  // lambda=l0
  float f1 = 1. - 2.*bh_M/r0;
  float k1_r = sigma*sqrt( 1. - ( f1*b*b )/(r0*r0) );
  float k1_p = b/(r0*r0);
  float r1 = r0 + k1_r*h/2.;
  
  // lambda = l0 + h/2
  float f2 = 1. - 2.*bh_M/r1;
  float k2_r = sigma*sqrt( 1. - ( f2*b*b )/(r1*r1) );
  float k2_p = b/(r1*r1);
  float r2 = r0 + k2_r*h/2.;
  
  // lambda = l0 + h/2  second time
  float f3 = 1. - 2.*bh_M/r2;
  float k3_r = sigma*sqrt( 1. - ( f3*b*b )/(r2*r2) );
  float k3_p = b/(r2*r2);
  float r3 = r0 + k3_r*h;
  
  // lambda = l0 + h
  float f4 = 1. - 2.*bh_M/r3; float k4_r = sigma*sqrt( 1. - ( f4*b*b )/(r3*r3) );
  float k4_p = b/(r3*r3);

  // derivative of r wrt lambda
  float rdot = (k1_r + 2.*k2_r + 2.*k3_r + k4_r)*.166666666667;
  float phidot = (k1_p + 2.*k2_p + 2.*k3_p + k4_p)*.166666666667;

  vec3 next_ray_pos = vec3(r0 + rdot*h, theta0, phi0 + phidot*h);

  // ----- Set inout vars
  // set rdot sign and flip sigma if direction turns
  if (rdot_sign == 0) {
      if (rdot > 0.0) rdot_sign = 1;
      else if (rdot < 0.0) rdot_sign = -1;
  }
  else if (rdot_sign == 1) {
      if (rdot < 0.0) { rdot_sign = -1; sigma *= -1.0; }
      else if (rdot == 0.0) rdot_sign = 0;
  }
  else if (rdot_sign == -1) {
      if (rdot > 0.0) { rdot_sign = 1; sigma *= -1.0; }
      else if (rdot == 0.0) rdot_sign = 0;
  }
  return next_ray_pos;
} //rk4_step

//
// --------- END Null Geodesic Stuff



// --------- END Math Stuff ------------------------------------- //


// ---------------------- Object Check Funcs ----------------------- //
// Like 90% sure this code works perfectly, since it worked fine for euclidean, and the functionality hasn't been changed

bool inside_bh(vec3 point, inout vec3 color) {
  // check if in event horizon
  vec3 diff = bh_center - point;
  float squared_dist = dot(diff,diff);
  bool in_event_horizon = squared_dist <= bh_squared_radius;
  if (in_event_horizon) {
    // TODO Debugging, blue
    color = min(color+vec3(0.0, 0.0, 1.0), vec3(1.0));
    return true;
  }

  // check if in disk
  bool in_radius = squared_dist >= squared_inner_radius && squared_dist <= squared_outer_radius;
  bool in_height = point.z >= (bh_center.z - 0.5) && point.z <= (bh_center.z + 0.5);
  bool in_disk = in_radius && in_height;
  if (in_disk) {
    // wanna color it depending on the distance
    // test: random function, not physics based, that's for later
    float x = sqrt(squared_dist);
    float a = sqrt(squared_inner_radius);
    float b = sqrt(squared_outer_radius);
    float c_r = 1.;
    float c_w = 25.;
    float red = ( 1./(x-c_r) - 1./(b-c_r) )/( 1./(a-c_r) - 1./(b-c_r) );
    float white = (1./(x-c_w) - 1./(b-c_w) )/( 1./(a-c_w) - 1./(b-c_w) );  
    color = min(color+vec3(red, white, white), vec3(1.) );
  }
  return false; // returning false means transparent
}


bool outside_world(vec3 point, inout vec3 color) {
  bool above_min = point.x>=min_bounds.x &&
                   point.y>=min_bounds.y &&
                   point.z>=min_bounds.z;
  bool below_max = point.x<=max_bounds.x &&
                   point.y<=max_bounds.y &&
                   point.z<=max_bounds.z;
  if (!above_min || !below_max) {
    color = min( color + vec3(0.5), vec3(1.) );
    return true;
  }
  return false;
}

bool inside_test_ball(vec3 point, inout vec3 color) {
  vec3 pos = vec3(100., 100., 100.);
  vec3 diff = pos - point;
  if (dot(diff,diff) <= 100.) {
    color = min( color+vec3(0.0, 1.0, 0.0), vec3(1.) );
    return true;
  }
  return false;
}


// -------- END Object Check Funcs -------------------------------- //




void main() {

  // ------------------------- Compute Pixel Direction ---------------- //
  // x is going right, y is going up
  vec2 pixel = gl_FragCoord.xy;

  vec3 width_adjustment = -((2.*pixel.x+1.)/2.) * pixel_width * w_vec;
  vec3 height_adjustment = ((2.*pixel.y+1.)/2.) * pixel_height * h_vec;
  // pixel direction, equal to (pixel center pos - camera pos)
  vec3 dir = bot_left_dir + width_adjustment + height_adjustment;
  // normalize and scale
  dir *= inversesqrt(dot(dir,dir)) * step_dist;

  // ---------- END Compute Pixel Direction ------------------------- //






  // -------------------------- Ray March Loop -------------------------// 


  // --------- Initialize

  vec3 color = vec3(0.);

  // camera position
  // TODO update rotation calls to and from
  vec3 cam_pos_pol = to_polar(cam_pos);

  // ray position and variants
  vec3 ray_pos = cam_pos,
       ray_pos_pol = to_polar(ray_pos),
       ray_pos_pol_eq = rotate_to_equator(ray_pos_pol, cam_pos_pol);

  // ray direction and variants
  vec3 dir_pol = to_polar(dir),
       dir_pol_eq = rotate_to_equator(dir_pol, cam_pos_pol);

  // create b, init sigma, rdot_sign
  float b = b(ray_pos_pol, dir_pol_eq, bh_M);
  float sigma = initial_sigma(dir_pol_eq, ray_pos_pol_eq);
  int rdot_sign = 0;
  
  // --------- Loop
  for (int step=0; step<max_steps; step++) {
    // Checks

    // if it returns true and breaks, that means opaque
    if (inside_bh(ray_pos, color)) { break; }
    if (inside_test_ball(ray_pos, color)) { break;}
    if (outside_world(ray_pos, color)) { break; }

    // step through geodesic
    ray_pos_pol_eq = rk4_step(
        ray_pos_pol_eq,
        b,
        sigma,        // inout vars
        rdot_sign,    // 
        step_dist
        );
    
    // Rotate back and convert to cartesian for object checks
    ray_pos_pol = rotate_from_equator(ray_pos_pol_eq, cam_pos_pol);
    ray_pos = to_cartesian(ray_pos_pol);
  } // ------ END Ray March Loop -------------------------------- //

  gl_FragColor = vec4(color, 1.0);
}
