// To Dos
// TODO Debugging
//  - fix the butt shape
//    - it could come from numerical problems in the step function
//    - it could do with numerical problems in the rotation
//
// TODO Features
//  - have local stepping params be passed in from cpu
//  - size disk based on stable innermost orbit, and then some fixed reasonable upper bound scaled w/ mass
//  - color disk based on temperature
//  - the center is kinda weird bc param wrt phi collapses, might need to specifically approximate light going directly towards origin with straight line marches
//  - once it's working, can start culling unused functions


// ------------- Uniforms
// camera
uniform vec3 cam_pos_IN;
uniform vec3 h_vec_IN; // orthonormal wrt metric, in terms of global basis
uniform vec3 w_vec_IN;
// camera direction stuff
uniform vec3 bot_left_dir_IN;
uniform float pixel_width;
uniform float pixel_height;
// bh stuff
uniform float M;
uniform float disk_inner_radius;
uniform float disk_outer_radius;
uniform float disk_temp_inner;
uniform float disk_temp_exponent;
uniform float disk_emissivity_exponent;
// world bounds
uniform vec3 min_bounds;
uniform vec3 max_bounds;
// ray marching
uniform float h;
uniform int max_steps;
// starfield
uniform sampler2D starfield;
uniform float star_exposure;


// ---------------------- Misc Math
const float PI = 3.1415926538;
const float EPS = 1e-9;

bool angles_almost_equal(float a1, float a2) {
  float diff = mod(a1 - a2, 2.*PI);
  return min(diff, 2.*PI - diff) < EPS;
}

// map a direction vector to equirectangular UVs
// dir is treated as a vec_cart3
vec2 dir_to_uv(vec3 dir) {
  vec3 n = normalize(dir);
  float phi = atan(n.y, n.x);               // [-pi, pi]
  float theta = acos(clamp(n.z, -1.0, 1.0)); // [0, pi]
  return vec2(phi / (2.0 * PI) + 0.5, theta / PI);
}

// approximate Planckian color for a given temperature (Kelvin).
// via CHATGPT, maps 1000K-40000K to RGB in [0,1].
vec3 blackbody_rgb(float temperature) {
  float t = clamp(temperature, 1000.0, 40000.0) / 100.0;
  float r, g, b;

  // Red channel
  if (t <= 66.0) {
    r = 1.0;
  } else {
    r = 1.292936186062745 * pow(t - 60.0, -0.1332047592);
  }

  // Green channel
  if (t <= 66.0) {
    g = 0.3900815787690196 * log(t) - 0.6318414437886275;
  } else {
    g = 1.129890860895294 * pow(t - 60.0, -0.0755148492);
  }

  // Blue channel
  if (t >= 66.0) {
    b = 1.0;
  } else if (t <= 19.0) {
    b = 0.0;
  } else {
    b = 0.5432067891101961 * log(t - 10.0) - 1.19625408914;
  }

  return clamp(vec3(r, g, b), vec3(0.0), vec3(1.0));
}

// thin-disk temperature profile: T(r) = T_in * (r_in / r)^(3/4)
float disk_temperature(float r) {
  float rin = max(disk_inner_radius, 1e-4);
  return disk_temp_inner * pow(rin / max(r, rin), disk_temp_exponent);
}

// computes if a line segment intersects a parallelogram,
//  - given 3 corners of a parallelogram
//  - also returns 0 if the line or parallelogram is degenerate,
//    or the line is parallel w/ plane
bool line_crosses_parallelogram(
    vec3 line_start, vec3 line_end, 
    vec3 corner1, vec3 corner2, vec3 corner3
    ) {

  vec3 // rename and create temp vectors
    a = corner1,
    b = corner2,
    c = corner3,
    r0 = line_start, // r for ray
    r1 = line_end,

    e1 = b-a, // edges of parallelogram coming from $a$
    e2 = c-a,
    
    d = r1 - r0, // direction of ray

    s = r0 - a; // some vector that isn't really interpretable

  // check if the line is degenerate
  if (abs(d.x)<EPS && abs(d.y)<EPS && (d.z)<EPS) return false;

  // let P be the point of intersection. P = a + u*e1 + v*e2
  // we will compute u and v

  // assuming A (defined later) is invertible,
  // we're solving Ax=s where:
  //  -  matrix A = (-d e1 e2)
  //  -  x = (t u v)^t
  //  -  s = r0 - a
  // using cramer's rule
  //  x_i = det(A_i) / det(A), where
  //    A_i is A but replace i'th col with b

  vec3 e1_cross_e2 = cross(e1, e2);

  // check if the parallelogram is degenerate
  if (abs(e1_cross_e2.x)<EPS && abs(e1_cross_e2.y)<EPS && abs(e1_cross_e2.z)<EPS) return false;

  float detA = dot(-d , e1_cross_e2 );

  // is A invertible?
  if (abs(detA) < EPS) return false;
  float invs_detA = 1. / detA;

  // solve for t
  float detA1 = dot(s, e1_cross_e2);
  float t = detA1 * invs_detA;
  if (t < 0. || t > 1.) return false;

  // u
  float detA2 = dot(-d, cross(s, e2) );
  float u = detA2 * invs_detA;
  if (u < 0. || u > 1.) return false;

  // v
  float detA3 = dot(-d, cross(e1, s) );
  float v = detA3 * invs_detA;
  if (v < 0. || v > 1.) return false;

  return true;
}

// -------- END Misc Math


// --------------------------- Coordinates ------------------------------- //

// --------------- Structs
// pixels
struct pixel_cart2 { float x,y; };

// 3d
struct point_cart3 { float x,y,z; };
struct point_polar3 { float r,theta,phi; };

struct vec_cart3 { point_cart3 p; float x_dot, y_dot, z_dot; };
struct vec_polar3 { point_polar3 p; float r_dot, theta_dot, phi_dot; }; // not used?

// 2d
struct point_cart2 { float x,y; }; // not used?
struct point_polar2 { float r, phi; };

struct vec_cart2 { point_cart2 p; float x_dot, y_dot; }; // not used?
struct vec_polar2 { point_polar2 p; float r_dot, phi_dot; }; // not used?

// and then we use built in vec3 for color



// ------- Coordinate Helper Functions

vec_cart3 add_vec(vec_cart3 v1, vec_cart3 v2) { return vec_cart3(v1.p, v1.x_dot+v2.x_dot, v1.y_dot+v2.y_dot, v1.z_dot+v2.z_dot); }
vec_cart3 subtract_vec(vec_cart3 v1, vec_cart3 v2) { return vec_cart3(v1.p, v1.x_dot-v2.x_dot, v1.y_dot-v2.y_dot, v1.z_dot-v2.z_dot); }
vec_cart3 scalar_mult(float alpha, vec_cart3 v) { return vec_cart3(v.p, alpha*v.x_dot, alpha*v.y_dot, alpha*v.z_dot); }

float dot_vec(vec_cart3 v1, vec_cart3 v2) {return v1.x_dot*v2.x_dot + v1.y_dot*v2.y_dot + v1.z_dot*v2.z_dot; }
float dot_point(point_cart3 v1, point_cart3 v2) {return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z; }

vec_cart3 cross_vec(vec_cart3 a, vec_cart3 b){
  return vec_cart3( a.p,
      a.y_dot*b.z_dot - a.z_dot*b.y_dot,
      a.z_dot*b.x_dot - a.x_dot*b.z_dot,
      a.x_dot*b.y_dot - a.y_dot*b.x_dot  );
}

point_cart3 cross_point(point_cart3 a, point_cart3 b){
  return point_cart3(
      a.y*b.z - a.z*b.y,
      a.z*b.x - a.x*b.z,
      a.x*b.y - a.y*b.x  );
}

mat3 transpose(mat3 m) {
    return mat3(
        vec3(m[0].x, m[1].x, m[2].x),
        vec3(m[0].y, m[1].y, m[2].y),
        vec3(m[0].z, m[1].z, m[2].z)
    );
}


bool points_almost_equal(point_polar3 a, point_polar3 b) {
  return 
    abs(a.r-b.r)<EPS 
    && angles_almost_equal(a.theta, b.theta) 
    && angles_almost_equal(a.phi,b.phi);
}


// -------- Coord Conversion
// conversion between points
point_cart2 convert_point_p2c(point_polar2 v) {
  return point_cart2(
    v.r*cos(v.phi),
    v.r*sin(v.phi)
  );
}
point_cart3 convert_point_p2c(point_polar3 v) {
  return point_cart3(
    v.r*sin(v.theta)*cos(v.phi),
    v.r*sin(v.theta)*sin(v.phi),
    v.r*cos(v.theta)
  );
}

point_polar2 convert_point_c2p(point_cart2 v) {
  float r = sqrt(v.x*v.x + v.y*v.y);
  float phi = atan(v.y , v.x);
  return point_polar2(r, phi);
}

point_polar3 convert_point_c2p(point_cart3 v) {
  float r = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
  float theta = atan(sqrt(v.x*v.x + v.y*v.y), v.z);
  float phi = atan(v.y , v.x);
  return point_polar3(r, theta, phi);
}

// conversion between tangent vectors

// given v \in T_pM, coordinates x^i = (x,y,z), q^a = (r,th,ph),
// polar to cart formula given by:
//  v^{x^i} = \frac{\partial x^i}{\partial q^a} v^{q^a}
//  where v^{coord} is v's coefficient of the basis vec \partial_coord
vec_cart3 convert_vec_p2c(vec_polar3 v) {
  float r = v.p.r, th = v.p.theta, ph = v.p.phi;
  float d_r = v.r_dot, d_th = v.theta_dot, d_ph = v.phi_dot;
  float // jacobian matrix
    dxdr = sin(th)*cos(ph), dxdth = r*cos(th)*cos(ph), dxdph = -r*sin(th)*sin(ph),
    dydr = sin(th)*sin(ph), dydth = r*cos(th)*sin(ph), dydph =  r*sin(th)*cos(ph),
    dzdr = cos(th),         dzdth = -r*sin(th);
  float
    x = dxdr*d_r + dxdth*d_th + dxdph*d_ph,
    y = dydr*d_r + dydth*d_th + dydph*d_ph,
    z = dzdr*d_r + dzdth*d_th;

  point_cart3 point = convert_point_p2c(v.p);

  return vec_cart3(point, x,y,z);
}


vec_polar3 convert_vec_c2p(vec_cart3 v) {
  point_polar3 point = convert_point_c2p(v.p);

  float x = v.p.x, y=v.p.y, z=v.p.z;
  float d_x = v.x_dot, d_y = v.y_dot, d_z = v.z_dot;

  float r = sqrt(x*x + y*y + z*z);
  if (r==0.) return vec_polar3(point, 0., 0., 0.); // edge!
  float rinvs = 1./r;
  
  float p = sqrt(x*x + y*y);
  if (p==0.) return vec_polar3(point, r, 0., 0.); // edge!
  float pinvs = 1./p;

  float // jacobian
    drdx  = x/r,     drdy = y/r,      drdz = z/r,
    dthdx = x*z/(r*r*p), dthdy = y*z/(r*r*p), dthdz = -p/(r*r),
    dphdx = -y/(p*p),    dphdy = x/(p*p);
  float
    r_out =     drdx*d_x  + drdy*d_y  + drdz*d_z,
    theta_out = dthdx*d_x + dthdy*d_y + dthdz*d_z,
    phi_out =   dphdx*d_x + dphdy*d_y;

  return vec_polar3(point, r_out, theta_out, phi_out);
}

point_cart3 convert_point_v2c3(vec3 p) {
  return point_cart3(p.x, p.y, p.z);
}

vec_cart3 convert_vec_v2c3(point_cart3 p, vec3 v) {
  return vec_cart3(p, v.x, v.y, v.z);
}


// --------- END Coordinates --------------------------------------------- //


// --------------- Metric and Helper Functions 

struct SC_Metric{
	float M;
};

float f(float r, SC_Metric metric) {
  return 1.- (2.*metric.M / r);
};

// distance between two points in spacetime
float distance_vec_p3(point_polar3 v1, point_polar3 v2, SC_Metric metric) {
  float
    r = 0.5*(v1.r + v2.r),
    d_r = v2.r - v1.r,
    d_th = v2.theta - v1.theta,
    d_ph = v2.phi - v1.phi,
    term_r = d_r*d_r / f(r, metric),
    term_ang = r*r*(d_th*d_th + sin(v1.theta)*sin(v2.theta)*d_ph*d_ph);
  return sqrt(term_r + term_ang);
}

// inner product
float g(vec_polar3 v1, vec_polar3 v2, SC_Metric metric) {
  //assert(points_almost_equal(v1.p, v2.p));
  float
    r = v1.p.r,
    theta = v1.p.theta,
    phi = v1.p.phi;
  return 
    ( 1./f(r, metric) ) * v1.r_dot * v2.r_dot
    + r*r * ( v1.theta_dot * v2.theta_dot 
              + sin(theta)*sin(theta) * v1.phi_dot*v2.phi_dot );
}

float squared_norm(vec_polar3 v, SC_Metric metric) {
  return g(v,v, metric);
}

vec_polar3 normalize_vecp3(vec_polar3 v, SC_Metric metric) {
  float norm_invs = 1. / sqrt(squared_norm(v, metric));
  return vec_polar3(v.p, v.r_dot*norm_invs, v.theta_dot*norm_invs, v.phi_dot*norm_invs);
}

float cos_angle_between(vec_polar3 u, vec_polar3 v, SC_Metric metric) {
  float
    g_uv = g(u,v, metric),
    g_uu = squared_norm(u, metric),
    g_vv = squared_norm(v, metric);
  return g_uv * inversesqrt(g_uu*g_vv);
}



// ---------------------------- Rotation Matrix ------------------------- //

// this matrix should take plane -> equator, 
// construct an ON basis w/ 2 vectors on the plane, then send 
// if v is a vector, Rv should take the geodesic of v to the equator
mat3 rotation_matrix(point_cart3 cam, vec_cart3 dir) {

  vec3 c = vec3(cam.x, cam.y, cam.z);
  vec3 d = vec3(dir.x_dot, dir.y_dot, dir.z_dot);

  if (length(c) < EPS) {
    return mat3(vec3(1.,0.,0.), vec3(0.,1.,0.), vec3(0.,0.,1.));
  }

  vec3 // v1 and v2 should be on the geodesic plane
    v1 = normalize(c),
    v2_candidate = d - dot(d,v1)*v1;

  // just to make sure v2 is linearly independent from v1
  if (length(v2_candidate) < EPS) {
    v2_candidate = vec3(1.,0.,0.) - dot(vec3(1.,0.,0.),v1)*v1;
    if (length(v2_candidate) < EPS) {
      v2_candidate = vec3(0.,1.,0.) - dot(vec3(0.,1.,0.),v1)*v1;
    }
  }

  vec3
    v2 = normalize(v2_candidate),
    u = normalize(cross(v1, v2));

  mat3 R_t = mat3(v1, v2, u); // this is the inverse of the matrix we want
  return transpose(R_t);
}

// ---------- END Rotation Matrix --------------------------------- //


// ---------------------- Geometry Helpers ----------------------- //

// Compute intersection of segment p0->p1 with disk in plane z=0 and radius bounds.
// Returns true and sets hit_point if the segment crosses the disk.
bool segment_hits_disk(point_cart3 p0, point_cart3 p1, out point_cart3 hit_point) {
  float dz = p1.z - p0.z;
  if (abs(dz) < EPS) return false; // segment parallel to plane
  float t = -p0.z / dz; // param where z=0
  if (t < 0.0 || t > 1.0) return false; // intersection not within segment

  hit_point = point_cart3(
      p0.x + (p1.x - p0.x) * t,
      p0.y + (p1.y - p0.y) * t,
      0.0);

  float r2 = hit_point.x*hit_point.x + hit_point.y*hit_point.y;
  float inner2 = disk_inner_radius * disk_inner_radius;
  float outer2 = disk_outer_radius * disk_outer_radius;
  return r2 >= inner2 && r2 <= outer2;
}

// -------- END Geometry Helpers --------------------------------- //

// ------------------------ Geodesic and Diff Eqs --------------------- // 

// -------------------- Structs
// meant to represent u:=1/r and du := du/dphi
struct u_vec{ float u, du; };

struct SC_Null_Geodesic{
  // input params
  SC_Metric metric;
	point_polar2 start_pos;  // mutable
  mat3 R; // matrix to convert cart3 to cart3 on the equator
  u_vec current_u;  // mutable
  float current_r;  //
  float current_phi;//
  bool left; // emission angle to the "left" means sin(psi)<0
	float h;
	int max_steps;
  float current_step;  // mutable
};
// ------- END Structs


// ------------------- Make Geodesic Support Functions

float init_b(float r, float sin_psi, SC_Metric metric) {
  return r*sin_psi / max(1e-12, sqrt(f(r, metric)));
}

// sigma here is the sign for du/dphi, not dr/lambda
float init_sigma(float cos_psi) {
  if (cos_psi > 0. ) {
    return -1.;}
  else if (cos_psi < 0. ) {
    return 1.;}
  else {
    return 0.;}
}

// -------- END Make Geodesic Support Functions



// -------------------- Diff Eq Solving

// first derivative of u wrt phi
float _du(float u, float b, float sigma, SC_Metric metric) {
  float M = metric.M;
  float du_squared = u*u*(2.*M*u - 1.) + 1./(b*b);
  return sigma*sqrt(du_squared);
}

// second derivative of u wrt phi
float _ddu(float u, SC_Metric metric) { return u*(3.*metric.M*u - 1.); }

// RK4 step for _ddu = \frac{d^2u}{d\phi^2}
// h := delta phi
u_vec rk4_step(u_vec current_u, float h, SC_Metric metric) {
  float // initial quantities
    u0 = current_u.u,
    du0 = current_u.du,
    M = metric.M;
  
  float // all the slopes n stuff
    k1_u = du0,
    k1_du = _ddu(u0, metric),

    k2_u = du0 + 0.5*h*k1_du,
    k2_du = _ddu(u0 + 0.5*h*k1_u, metric),

    k3_u = du0 + 0.5*h*k2_du,
    k3_du = _ddu(u0 + 0.5*h*k2_u, metric),
  
    k4_u = du0 + h*k3_du,
    k4_du = _ddu(u0 + h*k3_u, metric);
  
  float // use averaged slopes to find next points
    u_next = u0 + h*(k1_u + 2.*k2_u + 2.*k3_u + k4_u)/6.,
    du_next = du0 + h*(k1_du + 2.*k2_du + 2.*k3_du + k4_du)/6.;

  return u_vec(u_next, du_next);
}

// -------- END Diff Eq Solving


// ------------------------- Make Null Geodesic 
SC_Null_Geodesic make_SC_Null_Geodesic(
    SC_Metric metric,
    point_cart3 cam_pos,
    vec_cart3 emission_dir,
    float h,
    int max_steps
    ) {

  // compute psi
  point_polar3 cam_pos_pol = convert_point_c2p(cam_pos);
  vec_polar3 emission_dir_pol = convert_vec_c2p(emission_dir);
  vec_polar3 bh_dir_pol = vec_polar3(cam_pos_pol, -1., 0., 0.);
  float cos_psi = cos_angle_between(emission_dir_pol, bh_dir_pol, metric);
  float sin_psi = sqrt(1. - cos_psi*cos_psi);

  // compute the rotation matrix
  mat3 R = rotation_matrix(cam_pos, emission_dir);

  // compute rotation and convert to right basis
  // start_pos is the emission start point, in equatorial polar coords
  vec3 cam_pos_vec3 = vec3(cam_pos.x, cam_pos.y, cam_pos.z);
  vec3 start_pos_vec3 = R * cam_pos_vec3; // mat mult works natively with vec3 hence all the conversion
  point_cart2 start_pos_cart = point_cart2(start_pos_vec3.x, start_pos_vec3.y);
  point_polar2 start_pos = convert_point_c2p(start_pos_cart);

  // if the photon is angled to the left, then phi should rotate the opposite direction
  //  and we should do the entire thing but mirrored
  bool left = false;
  if (sin_psi < 0.) {
    left = true;
  }

  float r0 = start_pos.r;
  float phi0 = start_pos.phi;

  float b = init_b(start_pos.r, sin_psi, metric);
  float sigma = init_sigma(cos_psi);

  // compute du0 := initial derivative of u wrt phi
  float u0 = 1. / r0;
  float du0 = _du(u0, b, sigma, metric);
  u_vec start_u_vec = u_vec(u0, du0);

  return SC_Null_Geodesic(
      metric, 
      start_pos, 
      R, 
      start_u_vec, 
      r0, 
      phi0,
      left, 
      h, 
      max_steps, 0.);
}

// ---------- END Make Null Geodesic


// ------------------- Stepping Through Geodesic (local h, step ray)

// compute step size
float local_h(SC_Null_Geodesic ray) {

  float 
    rM = ray.current_r / ray.metric.M,
    rM0 = 4.0,  // under this, local_h = h_min
    rate = 0.1,  // the rate at which the weight increases
    h_min = ray.h,
    h_max = ray.h * 10.;

  float t = clamp(rate * (rM - rM0), 0.0, 10.0);
  float weight = t / (1.0 + t); // ranging b/w 0 and 1
  // linearly interpolate bw min and max by weight
  float h = mix(h_min, h_max, weight);  
  return h;
}

// Step through geodesic, output the next point in 3d cartesian coords
point_cart3 step_ray(inout SC_Null_Geodesic ray) {

  float h = local_h(ray);

  ray.current_u = rk4_step(ray.current_u, h, ray.metric);
  
  // update mutables in ray
  ray.current_r = 1. / ray.current_u.u;
  ray.current_phi += h;
  ray.current_step+=1.;

  float
    r = ray.current_r,
    phi = ray.current_phi;

  // points in equatorial plane
  point_polar2 current_pos_p2 = point_polar2(r, phi);
  point_cart2 current_pos_c2 = convert_point_p2c(current_pos_p2);
  point_cart3 current_pos_c3 = point_cart3(current_pos_c2.x, current_pos_c2.y, 0.);
  vec3 current_pos_v3 = vec3(current_pos_c3.x, current_pos_c3.y, current_pos_c3.z);

  // points in normal space
  // so we gotta rotate from
  vec3 point = transpose(ray.R) * current_pos_v3;
  return point_cart3(point.x, point.y, point.z);
}

// -------- END Stepping Through Geodesic


// ---- END Geodesic and Diff Eqs ------------------------------------------------ // 


// ---------------------- Object Check Functions ----------------------- //


void check_bh(point_cart3 point, point_cart3 last_point, inout float passthrough, inout vec3 color) {
  // check if in event horizon
  if (passthrough <= 0.) return;

  float dist = sqrt(point.x*point.x + point.y*point.y + point.z*point.z);

  bool in_event_horizon = dist < 3.*M; // 3M is closest stable orbit, anything within die
  if (in_event_horizon) {
    color = min(color+vec3(0.0), vec3(1.0));
    passthrough = 0.;
  }

  // thin disk: detect crossing of z=0 plane within radius bounds
  point_cart3 hit_point;
  bool crosses_disk = segment_hits_disk(last_point, point, hit_point);

  if (crosses_disk) {
    // Disk emissivity and color (simple thin-disk model)
    float r_hit = sqrt(hit_point.x*hit_point.x + hit_point.y*hit_point.y);
    float temp = disk_temperature(r_hit);
    vec3 bb = blackbody_rgb(temp);

    // Emissivity falls off ~r^-q (standard thin-disk scaling), q≈3
    float emissivity = pow(disk_inner_radius / max(r_hit, disk_inner_radius), disk_emissivity_exponent);

    // Apply color and attenuate passthrough by emissivity (clamped to [0,1])
    float opacity = clamp(emissivity, 0.0, 1.0);
    color = clamp(color + emissivity * bb, vec3(0.0), vec3(1.0));
    passthrough = clamp(passthrough - opacity, 0.0, 1.0);
  }
}



void check_test_cube(
    point_cart3 point, point_cart3 last_point,
    inout float passthrough, inout vec3 color
    ) {

  if (passthrough <= 0.) return;

  // CUBE center
  point_cart3 p = point_cart3(100., 60., 70.);
  float side_length = 20.;
  float size = side_length * 0.5;
  float opacity = 1.;

  bool // check if leading point is inside walls of CUBE
    in_x = point.x <= p.x+size && point.x >= p.x-size,
    in_y = point.y <= p.y+size && point.y >= p.y-size,
    in_z = point.z <= p.z+size && point.z >= p.z-size,
    in_cube = in_x && in_y && in_z;

  // check if past line segment intersects w/ walls of CUBE
  vec3 //define points of the CUBE
    a = vec3(p.x + size, p.y + size, p.z + size),
    b = vec3(p.x + size, p.y + size, p.z - size),
    c = vec3(p.x + size, p.y - size, p.z + size),
    d = vec3(p.x + size, p.y - size, p.z - size),
    e = vec3(p.x - size, p.y + size, p.z + size),
    f = vec3(p.x - size, p.y + size, p.z - size),
    g = vec3(p.x - size, p.y - size, p.z + size),
    h = vec3(p.x - size, p.y - size, p.z - size),
  
    // define line segment points
    r0 = vec3(point.x, point.y, point.z),
    r1 = vec3(last_point.x, last_point.y, last_point.z);

  bool // now we check if it crosses each of the 6 fucking walls
    w1 = line_crosses_parallelogram(r0, r1, h, g, d),
    w2 = line_crosses_parallelogram(r0, r1, d, c, b),
    w3 = line_crosses_parallelogram(r0, r1, b, a, f),
    w4 = line_crosses_parallelogram(r0, r1, f, e, h),
    w5 = line_crosses_parallelogram(r0, r1, g, e, c),
    w6 = line_crosses_parallelogram(r0, r1, h, f, d),
    crosses_wall = w1 || w2 || w3 || w4 || w5 || w6;
  
  if (in_cube || crosses_wall) {
    color = min( color+vec3(0.6, 0.0, 0.7), vec3(1.) );
    passthrough = clamp(passthrough - opacity, 0., 1.);
  }
}


void check_grid(point_cart3 p, inout float passthrough, inout vec3 color) {

  // if no more light, skip
  if (passthrough <= 0.) return;

  float z_axis = -150.;
  float line_space = 50.; // units
  float thickness = 1.5;
  float opacity = 0.4;
  
  bool 
    in_x = mod(p.x, line_space) <= thickness,
    in_y = mod(p.y, line_space) <= thickness,
    in_z = p.z >= z_axis-thickness && p.z <= z_axis+thickness,
    in_line = (in_x || in_y) && in_z;

  if (in_line) {
    color = min( color + vec3(0.2, 0.2, 0.2), vec3(1.));
    passthrough = clamp(passthrough - opacity , 0., 1.);
  }
}


// This one is special, and just returns true or false depending on whether or not it's out of bounds.
bool check_world_limits(point_cart3 point) {
  bool above_min = point.x>=min_bounds.x &&
                   point.y>=min_bounds.y &&
                   point.z>=min_bounds.z;
  bool below_max = point.x<=max_bounds.x &&
                   point.y<=max_bounds.y &&
                   point.z<=max_bounds.z;
  return !above_min || !below_max;
}


// -------- END Object Check Funcs -------------------------------- //


// ------------------------------------- MAIN --------------------------------- //
void main() {

  // -------------------------- Make Geodesic Object ------------------------- //
  // Converting inputs to the right forms
  point_cart3 cam_pos = convert_point_v2c3(cam_pos_IN);
  vec_cart3
    h_vec = convert_vec_v2c3(cam_pos, h_vec_IN),
    w_vec = convert_vec_v2c3(cam_pos, w_vec_IN),
    bot_left_dir = convert_vec_v2c3(cam_pos, bot_left_dir_IN);

  // make Metric
  SC_Metric metric = SC_Metric(M);


  // ------------ Compute Emission Direction
  // x is going right, y is going up
  pixel_cart2 pixel = pixel_cart2(gl_FragCoord.x, gl_FragCoord.y);

  vec_cart3 width_adjustment = scalar_mult( ((2.*pixel.x+1.)/2.) * pixel_width,  w_vec );
  vec_cart3 height_adjustment = scalar_mult( -((2.*pixel.y+1.)/2.) * pixel_height,  h_vec );

  vec_cart3 emission_dir = add_vec(add_vec( bot_left_dir, width_adjustment ), height_adjustment);
  vec3 emission_dir_v3 = normalize(vec3(emission_dir.x_dot, emission_dir.y_dot, emission_dir.z_dot));
  // ----- END Compute Emission Direction


  // make Geodesic
  SC_Null_Geodesic ray = make_SC_Null_Geodesic(
      metric,
      cam_pos,
      emission_dir,
      h,
      max_steps );

  // --------- END Make Geodesic Obejct --------------------------------------- //

  // --------- Init marching values
  vec3 color = vec3(0.);  // cummulative color of the ray
  float passthrough = 1.; // how much light in the ray is left
  bool hit_background = false; // true when we exit world bounds

  // current ray position, udpated with steps and used to check bounds
  point_cart3 ray_pos = cam_pos, last_ray_pos = ray_pos;
  vec3 exit_dir_v3; // content-wise a vec_cart3 at p = latest ray_pos
  // ------ END Init marching values

  // check that camera is not inside event horizon
  if (sqrt(cam_pos.x*cam_pos.x + cam_pos.y*cam_pos.y + cam_pos.z*cam_pos.z) >= 2.*metric.M) {

    // -------------------- Ray Marching Loop --------------------- //
    for (int step=0; step<max_steps; step++) {

      if (passthrough <= 0.) break;

      // Check Object Bounds
      //   if it returns true and breaks, that means opaque
      if (check_world_limits(ray_pos)) { hit_background = true; break; }
      check_bh(ray_pos, last_ray_pos, passthrough, color);
      check_test_cube(ray_pos, last_ray_pos, passthrough, color);
      check_grid(ray_pos, passthrough, color);


      // march the ray
      last_ray_pos = ray_pos;
      ray_pos = step_ray(ray);

      // exit dir

      exit_dir_v3 = vec3( 
        ray_pos.x - last_ray_pos.x,
        ray_pos.y - last_ray_pos.y,
        ray_pos.z - last_ray_pos.z
        );

    } // ------ END Ray March Loop -------------------------------- //

  }

  // sample starfield in remaining light
  if ((passthrough > 0.0 || hit_background) && star_exposure > 0.0) {
    vec2 uv = dir_to_uv(exit_dir_v3);
    vec3 star = texture2D(starfield, uv).rgb * star_exposure;
    float weight = hit_background ? 1.0 : passthrough;
    color = clamp(color + weight * star, 0.0, 1.0);
  }

  // set final pixel color
  gl_FragColor = vec4(color, 1.0);

} // Main
