#include <SFML/Graphics.hpp>
#include <iostream>
#include <array>
#include <cmath>

using std::cos, std::sin, std::atan2, std::sqrt, std::max, std::min, std::array, std::cout, std::endl;


const float PI = 3.1415926538;

constexpr float minx=-500,  maxx=500;
constexpr float miny=-500,  maxy=500;


// --------------------------- Coordinates ------------------------------- //

struct point_cart2 { float x,y; };
struct point_polar2 { float r, phi; }; // meant to be equatorial plane of Pol3

struct vec2 { float x,y; }; // just a general purpose 2-value structure

struct vel_polar2 { point_polar2 p; float r_dot, phi_dot; };

// ------- Coordinate Helper funcs
inline std::ostream& operator<<(std::ostream& os, const point_cart2& v) {
    return os << "point_cart2(x=" << v.x << ", y=" << v.y << ")";
}
inline std::ostream& operator<<(std::ostream& os, const point_polar2& v) {
    return os << "point_polar2(r=" << v.r << ", phi=" << v.phi << ")";
}
inline std::ostream& operator<<(std::ostream& os, const vel_polar2& v) {
    return os << "point_polar2(r_dot=" << v.r_dot << ", phi_dot=" << v.phi_dot << ") at p=" << v.p;
}

point_polar2 cart_to_pol(point_cart2 v) {
  return point_polar2{
    sqrt(v.x*v.x + v.y*v.y),
    atan2(v.x , -v.y)
  };
}

point_cart2 pol_to_cart(point_polar2 v) {
  return point_cart2{
    v.r*sin(v.phi),
    -v.r*cos(v.phi)
  };
}

// ---------- END Coordinates ---------------------------------------- // 



// ---------------------- Metric and functions ----------------------- //

struct SC_Metric{
	float M;
};

float f(float r, const SC_Metric& metric) {
  return 1.f- (2.f*metric.M / r);
};

// TODO if needed
float inner_product(vel_polar2 v1, vel_polar2 v2, const SC_Metric& metric) {
  return 0.;
}


// ------------------------ Geodesic and functions ----------------------- // 

struct SC_Null_Geodesic{
  // input params
  const SC_Metric metric;
	const vel_polar2 start_vel;
	const float h;
	const int max_steps;
  
  // computed from prior
  const float b;
  // mutable stuff
  vel_polar2 current_vel;// = start_vel; by default
  int place;
};


float init_b(float r, float psi, SC_Metric metric) {
  return r*sin(psi) / max(1e-12f, sqrt(f(r, metric)));
}

float current_sigma(SC_Null_Geodesic &ray) {
  // used to initiate sigma (current_vel should == start_vel after initialization)
  // then used to update sigma during stepping
  // TODO 
  //  - use a function rather than repeating code
  float psi = std::atan2(ray.start_vel.phi_dot , ray.start_vel.r_dot);
  // REPEATED CODE
  float sigma = (cos(psi) > 0)? 1 : -1;
  return sigma;
}



vel_polar2 rk4_step(SC_Null_Geodesic &ray) {
  // TODO
  //  - convert to computing along steps of phi

  //cout << "ray_pos = "<<ray_pos<<endl;

  point_polar2 ray_pos = ray.current_vel.p;

  float r0 = max(1e-12f, ray_pos.r),
      phi0 = ray_pos.phi,
      b = ray.b,
      h = ray.h,
      sigma = current_sigma(ray);

  cout << "r0="<<r0<<", phi0="<<phi0<< ", b=" << b<<  endl;

  // lambda=l0
  float f0 = f(r0, ray.metric);

  cout << "f0 = " << f0 << endl;

  float term_inside = 1.f - ( f0*b*b )/(r0*r0);
  cout << "term inside = "<<term_inside << endl;

  float k1_r = sigma*sqrt( max(0.f, 1.f - ( f0*b*b )/(r0*r0) ) );

  cout << "k1_r = " << k1_r << endl;

  float k1_p = b/(r0*r0);
  //cout << "k1_p = " << k1_p << endl;
  float r1 = max(1e-12f, r0 + k1_r*h/2.f);
  //cout << "r1 = " << r1 << endl;
  
  // lambda = l0 + h/2

  float f1 = f(r1, ray.metric);
  float k2_r = sigma*sqrt(max(0.f, 1.f - ( f1*b*b )/(r1*r1) ));
  cout << "k2_r = " << k2_r << endl;
  float k2_p = b/(r1*r1);
  float r2 = max(1e-12f, r0 + k2_r*h/2.f);
  
  // lambda = l0 + h/2  second time
  float f2 = f(r2, ray.metric);
  float k3_r = sigma*sqrt(max(0.f, 1.f - ( f2*b*b )/(r2*r2) ));
  cout << "k3_r = "<<k3_r<<endl;
  float k3_p = b/(r2*r2);
  //cout << "k3_p = " << k3_p << endl;
  float r3 = max(1e-12f, r0 + k3_r*h);
  //cout << "r3 = "<< r3 << endl;
  
  // lambda = l0 + h
  float f3 = f(r3, ray.metric);
  float k4_r = sigma*sqrt(max(0.f, 1.f - ( f3*b*b )/(r3*r3) ));
  cout << "k4_r = " << k4_r << endl;
  float k4_p = b/(r3*r3);

  // derivative of r wrt lambda
  float rdot = (k1_r + 2.*k2_r + 2.*k3_r + k4_r)*.166666666667;
  cout << "rdot = "<<rdot<<endl;
  float phidot = (k1_p + 2.*k2_p + 2.*k3_p + k4_p)*.166666666667;
  cout<< "phidot = "<<phidot <<endl;

  point_polar2 next_pos = point_polar2{max(1e-9f, r0 + rdot*h), phi0 + phidot*h};

  // udpate current_vel to best estimate thus far of next pos and vel
  //   k4_r and k4_p should be a better estimate than rdot,phidot
  vel_polar2 next_vel = vel_polar2{next_pos, k4_r, k4_p};
  return next_vel;
  }


  SC_Null_Geodesic make_SC_Null_Geodesic(
      SC_Metric metric,
      point_polar2 start_pos,
      float psi,
      float h,
      int max_steps
      ) {
    // TODO
    //   - initiate with psi, then compute the start_vel with derivative formula
    

    // TODO compute b
    float b = init_b(start_pos.r, psi, metric);
    // compute sigma
    // REPEATED CODE
    float sigma = (cos(psi) > 0)? 1 : -1;

    // TODO compute start_vel
    
    float r0 = start_pos.r;
    float f0 = f(r0, metric);
    // REPEATED CODE
    float k1_r = sigma*sqrt( max(0.f, 1.f - ( f0*b*b )/(r0*r0) ) );
    float k1_p = b/(r0*r0);

    vel_polar2 start_vel = {start_pos, k1_r, k1_p};
    

    SC_Null_Geodesic ray = {metric, start_vel, h, max_steps, b, start_vel, 0};

    return ray;
  }

  template<int max_steps>
  array<sf::Glsl::Vec2, max_steps> compute_SC_Null_Geodesic(SC_Null_Geodesic &ray) {
    array<sf::Glsl::Vec2, max_steps> points;
    for(int i=0; i<max_steps; i++) {

      vel_polar2 vel = rk4_step(ray);
      ray.current_vel = vel; 

      points.at(i) = sf::Glsl::Vec2(vel.p.r, vel.p.phi); // extract point from it and add to list
    }
    return points;
  }


// ------ END Step function ----------------------------------- //


int main() {
  const unsigned W = static_cast<unsigned>(maxx - minx);
  const unsigned H = static_cast<unsigned>(maxy - miny);

  // MAGIC NUMBERS

  float drawing_radius = 3.;

  SC_Metric metric{25.};

  // ------------ step through a geodesic ---------- //
  
  // for a single ray 
  // TODO
  //  - make a SC_Null_Geodesic
  //  - stop when inside r_s

    
  point_polar2 start_pos = {200.f , 0.f};
  float psi = 0.6*PI;
  const int max_steps = 512;
  const float h = 1.;
  cout << "Initial ray start position = "<<start_pos << ", with psi angle = "<<psi<<endl;

  SC_Null_Geodesic ray = make_SC_Null_Geodesic(metric, start_pos, psi, h, max_steps);

  array<sf::Glsl::Vec2, max_steps> points = compute_SC_Null_Geodesic<max_steps>(ray);

  // ------- END geodesic stepthrough


  sf::RenderWindow win(sf::VideoMode(W,H), "2d thang");
  win.setFramerateLimit(60);
  sf::RectangleShape screen{ sf::Vector2f(static_cast<float>(W), static_cast<float>(H)) };
  screen.setPosition(0.f, 0.f);
  if (!sf::Shader::isAvailable()) { std::cerr << "No shader support\n"; return 1; }
  sf::Shader shader;
  if (!shader.loadFromFile("main2.frag", sf::Shader::Fragment)) return 1;

  // Set Uniforms
  shader.setUniform("boundsMin",  sf::Glsl::Vec2(minx, miny));
  shader.setUniform("boundsMax",  sf::Glsl::Vec2(maxx, maxy));
  shader.setUniformArray("points", points.data(), points.size());
  shader.setUniform("drawing_radius", drawing_radius);
  shader.setUniform("r_s", 2.f*metric.M);

  sf::RenderStates states;     // use RenderStates to attach shader
  states.shader = &shader; // we pass the shader object into the RenderStates object for some reason

  // main loop
  while (win.isOpen()) {
    for (sf::Event e; win.pollEvent(e); ) {
      if (e.type == sf::Event::Closed) win.close();
      if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Escape) win.close();
    }

    win.clear();
    win.draw(screen, states);
    win.display();
  }
}

