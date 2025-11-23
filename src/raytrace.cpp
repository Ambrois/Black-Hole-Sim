// compile with:
//    g++ -O3 raytrace2.cpp -lsfml-graphics -lsfml-window -lsfml-system -o shader

#include <SFML/Graphics.hpp>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <tuple>
#include <cassert>

// TODO needed features
//  - put this shit on github!!!
//  - parallel transport camera vectors during camera movement to remove weird shifting
//  - display a picture of the stars in the background
//  - fix one of the camera FOVs given pixel ratio
//  - looking straight up or down fucks the angle, stop from looking up/down too much


using std::cos, std::sin, std::min, std::max, std::cout, std::endl;
constexpr double PI = 3.14159265358979323846;
constexpr double EPS = 1e-9;

// ------------------------------- Math Helpers ---------------------------- //

bool angles_almost_equal(double a1, double a2) {
  double diff = std::fmod(a1 - a2, 2*PI);
  return min(diff, 2*PI - diff) < EPS;
}


// ------- END Math Helpers ------------------------------------------------ //



// ------------------------------- Coordinates ---------------------------- //

// points in M w/ fixed t
// alignment:
//    phi=0 rotates from positive x to positive y axis
//    theta=0 rotates from positive z down
struct point_cart3 { double x,y,z; };
struct point_polar3 { double r,theta,phi; };

// vectors in TpM - span(\partial_t)
struct vec_cart3 {point_cart3 p; double x_dot, y_dot, z_dot; };
struct vec_polar3 {point_polar3 p; double r_dot, theta_dot, phi_dot; };

// ------- Coordinate Helper Functions

inline std::ostream& operator<<(std::ostream& os, const point_cart3& v) {
    return os << "point_cart3(x=" << v.x << ", y=" << v.y << ", z="<<v.z<< ")";
}
inline std::ostream& operator<<(std::ostream& os, const vec_cart3& v) {
    return os << "vel_cart3(x_dot=" << v.x_dot << ", y_dot=" << v.y_dot << ", z_dot="<<v.z_dot<< ")";
}

inline std::ostream& operator<<(std::ostream& os, const point_polar3& v) {
    return os << "point_polar3(r=" << v.r << ", theta=" << v.theta << ", phi="<<v.phi<< ")";
}
inline std::ostream& operator<<(std::ostream& os, const vec_polar3& v) {
    return os << "vel_polar3(r_dot=" << v.r_dot << ", theta_dot=" << v.theta_dot << ", phi_dot="<<v.phi_dot<< ") at point "<<v.p;
}

// --- math

// conversion between points
// polar to cart
point_cart3 convert_point_p2c(const point_polar3 &v) {
  return {
    v.r*sin(v.theta)*cos(v.phi),
    v.r*sin(v.theta)*sin(v.phi),
    v.r*cos(v.theta)
  };
}

point_polar3 convert_point_c2p(const point_cart3 &v) {
  double r = sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
  double theta = std::atan2(sqrt(v.x*v.x + v.y*v.y), v.z);
  double phi = std::atan2(v.y , v.x);
  return {r, theta, phi};
}


// conversion between tangent vectors

// given v \in T_pM, coordinates x^i = (x,y,z), q^a = (r,th,ph),
// polar to cart formula given by:
//  v^{x^i} = \frac{\partial x^i}{\partial q^a} v^{q^a}
//  where v^{coord} is v's coefficient of the basis vec \partial_coord
vec_cart3 convert_vec_p2c(const vec_polar3 &v) {
  double r = v.p.r, th = v.p.theta, ph = v.p.phi;
  double d_r = v.r_dot, d_th = v.theta_dot, d_ph = v.phi_dot;
  double // jacobian matrix
    dxdr = sin(th)*cos(ph), dxdth = r*cos(th)*cos(ph), dxdph = -r*sin(th)*sin(ph),
    dydr = sin(th)*sin(ph), dydth = r*cos(th)*sin(ph), dydph =  r*sin(th)*cos(ph),
    dzdr = cos(th),         dzdth = -r*sin(th);
  double
    x = dxdr*d_r + dxdth*d_th + dxdph*d_ph,
    y = dydr*d_r + dydth*d_th + dydph*d_ph,
    z = dzdr*d_r + dzdth*d_th;

  point_cart3 point = convert_point_p2c(v.p);

  return {point, x,y,z};
}

vec_polar3 convert_vec_c2p(const vec_cart3 &v) {
  point_polar3 point = convert_point_c2p(v.p);

  double x = v.p.x, y=v.p.y, z=v.p.z;
  double d_x = v.x_dot, d_y = v.y_dot, d_z = v.z_dot;

  double r = sqrt(x*x + y*y + z*z);
  if (r==0) return {point, 0., 0., 0.}; // edge!
  double rinvs = 1./r;
  
  double p = sqrt(x*x + y*y);
  if (p==0) return {point, r, 0., 0.}; // edge!
  double pinvs = 1./p;

  double // jacobian
    drdx  = x/r,     drdy = y/r,      drdz = z/r,
    dthdx = x*z/(r*r*p), dthdy = y*z/(r*r*p), dthdz = -p/(r*r),
    dphdx = -y/(p*p),    dphdy = x/(p*p);
  double
    r_out =     drdx*d_x  + drdy*d_y  + drdz*d_z,
    theta_out = dthdx*d_x + dthdy*d_y + dthdz*d_z,
    phi_out =   dphdx*d_x + dphdy*d_y;


  return {point, r_out, theta_out, phi_out};
}



bool points_almost_equal(point_polar3 a, point_polar3 b) {
  return 
    abs(a.r-b.r)<EPS 
    && angles_almost_equal(a.theta, b.theta) 
    && angles_almost_equal(a.phi,b.phi);
}

bool points_almost_equal(point_cart3 a, point_cart3 b) {
  point_polar3
    ap = convert_point_c2p(a),
    bp = convert_point_c2p(b);
  return 
    abs(ap.r-bp.r)<EPS 
    && angles_almost_equal(ap.theta, bp.theta) 
    && angles_almost_equal(ap.phi,bp.phi);
}


// Operators
vec_cart3 operator+(const vec_cart3 &v1, const vec_cart3 &v2) {
  assert(points_almost_equal(v1.p, v2.p));
  return {v1.p, v1.x_dot+v2.x_dot, v1.y_dot+v2.y_dot, v1.z_dot+v2.z_dot}; 
}
vec_cart3 operator-(const vec_cart3 &v1, const vec_cart3 &v2) {
  assert(points_almost_equal(v1.p, v2.p));
  return {v1.p, v1.x_dot-v2.x_dot, v1.y_dot-v2.y_dot, v1.z_dot-v2.z_dot}; 
}
vec_cart3 operator*(double alpha, const vec_cart3 &v) {
  return {v.p, alpha*v.x_dot, alpha*v.y_dot, alpha*v.z_dot}; 
}
vec_cart3 operator*(const vec_cart3 &v, double alpha) {
  return {v.p, alpha*v.x_dot, alpha*v.y_dot, alpha*v.z_dot}; 
}

static inline double dot(const vec_cart3 &v1, const vec_cart3 &v2) {return v1.x_dot*v2.x_dot + v1.y_dot*v2.y_dot + v1.z_dot*v2.z_dot; }

static inline vec_cart3 cross(const vec_cart3& a, const vec_cart3& b){
  assert(points_almost_equal(a.p, b.p));
  return {
    a.p,
    a.y_dot*b.z_dot - a.z_dot*b.y_dot,
    a.z_dot*b.x_dot - a.x_dot*b.z_dot,
    a.x_dot*b.y_dot - a.y_dot*b.x_dot
  };
}

// rodrigues matrix for tangent vectors
static inline vec_cart3 rotate_vector(vec_cart3 v, vec_cart3 k_unit, double theta) {
  double c=std::cos(theta), s=std::sin(theta), kd=dot(k_unit,v);
  vec_cart3 kv=cross(k_unit,v); 
  return { v.p,
           v.x_dot*c + kv.x_dot*s + k_unit.x_dot*kd*(1.-c),
           v.y_dot*c + kv.y_dot*s + k_unit.y_dot*kd*(1.-c),
           v.z_dot*c + kv.z_dot*s + k_unit.z_dot*kd*(1.-c) };
}






// -------- END Coordinates ------------------------------------------- //


// -------------------------------- Metric ----------------------------- //

// We're going to maintain signature (-+++)
struct SC_Metric{
	double M;

  double f(double r) {
    return 1.- (2.*M / r);
  }

  // inner product
  inline double g(vec_polar3 v1, vec_polar3 v2) {
    assert(points_almost_equal(v1.p, v2.p));
    double
      r = v1.p.r,
      theta = v1.p.theta,
      phi = v1.p.phi;
    return 
      ( 1./f(r) ) * v1.r_dot * v2.r_dot
      + r*r * ( v1.theta_dot * v2.theta_dot 
                + sin(theta)*sin(theta) * v1.phi_dot*v2.phi_dot );
  }

  inline double squared_norm(vec_polar3 v) { return g(v,v); }

  inline vec_polar3 normalize(vec_polar3 v) {
    double norm_invs = 1. / sqrt(squared_norm(v));
    return {v.p, v.r_dot*norm_invs, v.theta_dot*norm_invs, v.phi_dot*norm_invs};
  }

  inline vec_cart3 normalize(vec_cart3 v) {
    vec_polar3 v_pol = convert_vec_c2p(v);
    double norm_invs = 1. / sqrt(squared_norm(v_pol));
    vec_polar3 v_norm_pol = vec_polar3{
        v_pol.p,
        v_pol.r_dot*norm_invs,
        v_pol.theta_dot*norm_invs,
        v_pol.phi_dot*norm_invs};
    return convert_vec_p2c(v_norm_pol);
  }

  // DEBUGGING
  bool basis_is_orthonormal(vec_polar3 v1, vec_polar3 v2, vec_polar3 v3) {

    bool
      v1normal = abs(squared_norm(v1) -1.) < EPS,
      v2normal = abs(squared_norm(v2) -1.) < EPS,
      v3normal = abs(squared_norm(v3) - 1.) < EPS; 
    
    bool // orthogonal
      v12orth = abs( g(v1,v2) ) < EPS,
      v13orth = abs( g(v1,v3) ) < EPS,
      v23orth = abs( g(v2,v3) ) < EPS;

    return v1normal && v2normal && v3normal && v12orth && v13orth && v23orth;
  }

};




// -------------- Define Camera, Objects ------------------ //

struct Accretion_Disk {
  double inner_radius, outer_radius; 
};



  
// TODO
//  - make a function to rotate the camera a lil
//    - i guess I could also just modify the mouse handling function? no the shader not running, need to do computations beforehand
//  - let's rotate the camera's tripod with cartesian rotation
//  - then send to polar to check for orthonormal
class Camera {
  private:
    point_cart3 c; // camera position

    vec_cart3 // the orthonormal tripod
      d_vec,  // facing this dir, pointer of left hand TODO Changed to left
      w_vec, // middle finger
      h_vec; // thumb
                                
    double d{1.};                // distance to frame
    double FOV_h{1.}; 
    double FOV_w{1.};
    double wasd_sensitivity{3.};
    double turn_sensitivity{0.001};
    const double TWO_PI = PI * 2.0;
    size_t W, H; // number of pixels, width and height
    SC_Metric metric;

    vec_cart3 UP_VEC;

    
  public:

    Camera(
        double x, double y, double z,
        size_t W, size_t H,
        SC_Metric metric)
    : c{x,y,z}, W{W}, H{H}, metric{metric} {

      //cout << "Initiating Camera" << endl;

      point_polar3 polar_c = convert_point_c2p(c);

      vec_polar3 
        r_basis =     metric.normalize(vec_polar3{polar_c, 1.0, 0.0, 0.0}),
        theta_basis = metric.normalize(vec_polar3{polar_c, 0.0, 1.0, 0.0}),
        phi_basis =   metric.normalize(vec_polar3{polar_c, 0.0, 0.0, 1.0});

      // Debugging
      assert(metric.basis_is_orthonormal(r_basis, theta_basis, phi_basis));

      d_vec = convert_vec_p2c(r_basis);
      w_vec = convert_vec_p2c(phi_basis);
      h_vec = convert_vec_p2c(theta_basis);

      UP_VEC = convert_vec_p2c(
          metric.normalize(
            convert_vec_c2p(
              vec_cart3{c, 0., 0., 1.}
            )
          )
        );

      //cout << "Camera Initiated" <<endl;

    };


    // TODO
    //  - need to parallel transport all the velocities
    void move_by(vec_cart3 move) {
      c = {c.x+move.x_dot, c.y+move.y_dot, c.z+move.z_dot};
      d_vec.p = c;
      w_vec.p = c;
      h_vec.p = c;
      UP_VEC.p = c;
    }

    void manage_inputs(sf::Vector2i relative_mouse_pos) {
      // wasd
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) {
        vec_cart3 move = -1. * d_vec * wasd_sensitivity;
        move_by(move);
      };
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) {
        vec_cart3 move = -1. * w_vec * wasd_sensitivity;
        move_by(move);
      };
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) {
        vec_cart3 move = d_vec * wasd_sensitivity;
        move_by(move);
      }; 
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) {
        vec_cart3 move = w_vec * wasd_sensitivity;
        move_by(move);
      };
      // up n down
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::Space)) {
        vec_cart3 move = -1. * h_vec * wasd_sensitivity;
        move_by(move);
      };
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) {
        vec_cart3 move = h_vec * wasd_sensitivity;
        move_by(move);
      };

      // if mouse moved
      if (relative_mouse_pos.x != 0 or relative_mouse_pos.y != 0) {
        // relative_mouse_pos is in pixel coords
        double pitch = relative_mouse_pos.y * turn_sensitivity;
        double yaw = relative_mouse_pos.x * turn_sensitivity;

        // yaw
        d_vec = rotate_vector(d_vec, UP_VEC, yaw);
        w_vec = metric.normalize(cross(UP_VEC, d_vec));

        // pitch
        d_vec = rotate_vector(d_vec, w_vec, pitch);
        h_vec = metric.normalize(cross(w_vec, d_vec));

      } // mouse moved
    } // manage_inputs


    // compute vectors and stuff for shader
    std::tuple<vec_cart3,double,double> compute_botleftdir_pixwidth_pixheight() {
      double half_width = d*std::tan(FOV_w/2.);  // *d not needed
      double half_height = d*std::tan(FOV_h/2.); // *d not needed
      vec_cart3 bot_left_dir = d*d_vec/* *d not needed */ - half_width*w_vec + half_height*h_vec;
      double pixel_width = (2.*half_width)/W;
      double pixel_height = (2.*half_height)/H;
      return {bot_left_dir, pixel_width, pixel_height};
    }

    point_cart3 get_pos() {return c;}
    vec_cart3 get_dv() {return d_vec;}
    vec_cart3 get_wv() {return w_vec;}
    vec_cart3 get_hv() {return h_vec;}

};


// ---- END Define Camera, Objects --------------------------------- //


int main() {
  // -------------------- Parameters --------------------- // 
  // world bounds
  const double 
    MINX=-500., MAXX=500.,
    MINY=-500., MAXY=500.,
    MINZ=-500., MAXZ=500.;

  // which direction is up
  // UP_VEC = vec_cart3{0.0, 0.0, 1.0};

  // window size
  const size_t W=500, H=500;

  double h = 0.004;
  int max_steps = 3000;


  SC_Metric metric{10.};
  Accretion_Disk disk{30, 80};

  Camera cam{0., 300., 20., W,H, metric};
  


  // ---- END Parameters --------------------------------- //

  // --------------------- Create and configure quad, shader, window -------------- //

  double r_s = 2.*metric.M;

  sf::RectangleShape screen{ sf::Vector2f(static_cast<float>(W), static_cast<float>(H)) };
  screen.setPosition(0.f, 0.f);

  // make shader and load frag code
  if (!sf::Shader::isAvailable()) {std::cerr << "Your shit ass hardware can't support shaders\n"; return 1;}
  sf::Shader shader;
  shader.loadFromFile("raytrace.frag", sf::Shader::Fragment);

  // make state object to put shader into to apply in draw
  sf::RenderStates states;
  states.shader = &shader;

  // create window
  sf::RenderWindow win(sf::VideoMode(W,H), "Raymarching");
  win.setFramerateLimit(60);
  win.setMouseCursorGrabbed(true); // grab and hide mouse bc FPS style
  win.setMouseCursorVisible(false);
  sf::Vector2i win_center(int(win.getSize().x/2), int(win.getSize().y/2));

  // ------- END Create and Configure window stuff ---------------------------------- //

  // Set shader uniforms that don't change during game loop
  shader.setUniform("M", static_cast<float>(metric.M));
  // disk
  shader.setUniform("disk_inner_radius", static_cast<float>(disk.inner_radius));
  shader.setUniform("disk_outer_radius", static_cast<float>(disk.outer_radius));
  // world bounds
  shader.setUniform("min_bounds", sf::Glsl::Vec3(MINX, MINY, MINZ));
  shader.setUniform("max_bounds", sf::Glsl::Vec3(MAXX, MAXY, MAXZ));
  // ray marching
  shader.setUniform("h", static_cast<float>(h));
  shader.setUniform("max_steps", static_cast<int>(max_steps));


  
  // --------------------- Game Loop --------------------- //
  while (win.isOpen()) {

    //cout << "A frame happened! ----------------- " << endl;
    
    // handle queued events like closing
    sf::Event e;
    while (win.pollEvent(e)) {
        // close
        if (e.type == sf::Event::Closed)
            win.close();
        if (e.type == sf::Event::KeyPressed &&
            e.key.code == sf::Keyboard::Escape)
            win.close();
    }

    // centering mouse on window
    sf::Vector2i mouse_pos = sf::Mouse::getPosition(win);
    sf::Mouse::setPosition(win_center, win);
    sf::Vector2i relative_mouse_pos = mouse_pos - win_center;
    // handle camera stuff
    cam.manage_inputs(relative_mouse_pos);
    

    // Set shader uniforms that change during game loop
    // camera 
    point_cart3 cam_pos = cam.get_pos();
    shader.setUniform("cam_pos_IN", sf::Glsl::Vec3(cam_pos.x, cam_pos.y, cam_pos.z));
    vec_cart3 h_vec = cam.get_hv();
    shader.setUniform("h_vec_IN", sf::Glsl::Vec3(h_vec.x_dot, h_vec.y_dot, h_vec.z_dot));
    vec_cart3 w_vec = cam.get_wv();
    shader.setUniform("w_vec_IN", sf::Glsl::Vec3(w_vec.x_dot, w_vec.y_dot, w_vec.z_dot));
    // direction stuff
    auto [bld, pw, ph] = cam.compute_botleftdir_pixwidth_pixheight();
    shader.setUniform("bot_left_dir_IN", sf::Glsl::Vec3(bld.x_dot, bld.y_dot, bld.z_dot));
    shader.setUniform("pixel_width", static_cast<float>(pw));
    shader.setUniform("pixel_height", static_cast<float>(ph));

    // Drawing to window
    win.clear();
    win.draw(screen, states);
    win.display();
    
  } // ----- END Game Loop ---------------------------------- //
}
