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

//struct vel_polar2 { point_polar2 p; float r_dot, phi_dot; };

// ------- Coordinate Helper funcs
inline std::ostream& operator<<(std::ostream& os, const point_cart2& v) {
    return os << "point_cart2(x=" << v.x << ", y=" << v.y << ")";
}
inline std::ostream& operator<<(std::ostream& os, const point_polar2& v) {
    return os << "point_polar2(r=" << v.r << ", phi=" << v.phi << ")";
}
//inline std::ostream& operator<<(std::ostream& os, const vel_polar2& v) {
//    return os << "point_polar2(r_dot=" << v.r_dot << ", phi_dot=" << v.phi_dot << ") at p=" << v.p;
//}

//point_polar2 cart_to_pol(point_cart2 v) {
//  return point_polar2{
//    sqrt(v.x*v.x + v.y*v.y),
//    atan2(v.x , -v.y)
//  };
//}

//point_cart2 pol_to_cart(point_polar2 v) {
//  return point_cart2{
//    v.r*sin(v.phi),
//    -v.r*cos(v.phi)
//  };
//}

// ---------- END Coordinates ---------------------------------------- // 



// ---------------------- Metric and functions ----------------------- //

struct SC_Metric{
	float M;
};

float f(float r, const SC_Metric& metric) {
  return 1.f- (2.f*metric.M / r);
};


// ------------------------ Geodesic and functions ----------------------- // 

// meant to represent u:=1/r and du := du/dphi
struct u_vec{ float u, du; };

struct SC_Null_Geodesic{
  // input params
  const SC_Metric metric;
	const point_polar2 start_pos;
  const u_vec start_u_vec;
  const bool left; // emission angle to the "left" means sin(psi)<0
	const float h;
	const int max_steps;
  // mutable stuff
  int place;
};


float init_b(float r, float psi, SC_Metric metric) {
  return r*sin(psi) / max(1e-12f, sqrt(f(r, metric)));
}


// sigma here is the sign for du/dphi
int init_sigma(float psi) {
  cout << "init sigma called, sigma = ";
  if (cos(psi) >0 ) {
    cout << -1 << endl;
    return -1;}
  else if (cos(psi) <0 ) {
    cout << 1 << endl;
    return 1;}
  else {
    cout << 0 << endl;
    return 0;}
}


// first derivative of u wrt phi
float _du(float u, float b, int sigma, SC_Metric metric) {
  float M = metric.M;
  float du_squared = u*u*(2.f*M*u - 1.f) + 1/(b*b);
  return sigma*sqrt(du_squared);
}

// second derivative of u wrt phi
float _ddu(float u, SC_Metric metric) { return u*(3.f*metric.M*u - 1.f); }


// rk4 step for _ddu = d^2u/dphi^2
// h := delta phi
u_vec rk4_step(u_vec start_u, float h, SC_Metric metric) {
  float u0 = start_u.u;
  float du0 = start_u.du;
  float M = metric.M;
  
  float k1_u = du0;
  float k1_du = _ddu(u0, metric);

  float k2_u = du0 + 0.5f*h*k1_du;
  float k2_du = _ddu(u0 + 0.5f*h*k1_u, metric);

  float k3_u = du0 + 0.5f*h*k2_du;
  float k3_du = _ddu(u0 + 0.5f*h*k2_u, metric);
  
  float k4_u = du0 + h*k3_du;
  float k4_du = _ddu(u0 + h*k3_u, metric);
  
  float u_next = u0 + h*(k1_u + 2.f*k2_u + 2.f*k3_u + k4_u)/6.f;
  float du_next = du0 + h*(k1_du + 2.f*k2_du + 2.f*k3_du + k4_du)/6.f;

  return u_vec{u_next, du_next};
}


SC_Null_Geodesic make_SC_Null_Geodesic(
    SC_Metric metric,
    point_polar2 start_pos,
    float psi,
    float h,
    int max_steps
    ) {

  // if the photon is angled to the left, then phi should rotate the opposite direction
  //  and we should do the entire thing but mirrored
  bool left = false;
  cout << "sin(psi) = " << sin(psi) << endl;
  if (sin(psi) < 0) {
    psi = 2.f*PI - psi; // adding 2pi just keeps it positive, but shouldn't be needed
    left = true;
  }

  float r0 = start_pos.r;

  float b = init_b(start_pos.r, psi, metric);
  int sigma = init_sigma(psi);

  // compute du0 := initial derivative of u wrt phi
  float u0 = 1.f / r0;
  float du0 = _du(u0, b, sigma, metric);
  u_vec start_u_vec = {u0, du0};


  cout << "start_u_vec: u0="<<u0<<" , du0="<<du0 << endl;

  SC_Null_Geodesic ray = {metric, start_pos, start_u_vec, left, h, max_steps, 0};
  return ray;
}


// Function to compute the points polar2(r,phi) along a Null Geodesic
// outputs points in Glsl vec2 type for sending to shader
template<int max_steps>
array<sf::Glsl::Vec2, max_steps> compute_SC_Null_Geodesic(SC_Null_Geodesic &ray) {

  int flip_sign = ray.left ? -1 : 1;
  float h = ray.h;
  float phi0 = ray.start_pos.phi;

  cout << "Starting compute_SC_Null_Geodesic with "<<
    "h="<<h<< ", u0="<< ray.start_u_vec.u << ", du0="<< ray.start_u_vec.du << endl;

  u_vec current_u_vec = ray.start_u_vec;

  array<sf::Glsl::Vec2, max_steps> points;

  for(int i=0; i<max_steps; i++) {

    current_u_vec = rk4_step(current_u_vec, h, ray.metric);

    float u = current_u_vec.u;

    float r = 1.f / u;
    float phi = phi0 + flip_sign*h*(i+1);

    points.at(i) = sf::Glsl::Vec2(r, phi);

    // check if u is almost 0, which means r goes off to infinity and the photon escapes
    if (u<1e-9f) { break; }

    // DEBUGGING printing first few
    if (i<5) {
      cout << "step "<<i<< ":\n" <<
        "  du=" << current_u_vec.du << "\n" <<
        "  position after:  r="<<r<<" and phi="<<phi<<endl;
    }

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

  // ------------------- step through a geodesic ----------------- //
  
  point_polar2 start_pos = {200.f , 1.f};
  float psi1 = 0.4*PI;
  float psi2 = 1.2*PI;
  const int max_steps = 512/2;
  const float h = 0.02;
  //cout << "Initial ray start position = "<<start_pos << ", with psi angle = "<<psi<<endl;

  cout << "##############################################################\n"<<
    "# Starting ray 1" << endl;
  SC_Null_Geodesic ray1 = make_SC_Null_Geodesic(metric, start_pos, psi1, h, max_steps);
  array<sf::Glsl::Vec2, max_steps> points1 = compute_SC_Null_Geodesic<max_steps>(ray1);

  cout << "##############################################################\n"<<
    "# Starting ray 2" << endl;
  SC_Null_Geodesic ray2 = make_SC_Null_Geodesic(metric, start_pos, psi2, h, max_steps);
  array<sf::Glsl::Vec2, max_steps> points2 = compute_SC_Null_Geodesic<max_steps>(ray2);

  array<sf::Glsl::Vec2, max_steps*2> points;
  for (int i=0; i<max_steps*2; i++) {
    if ( i<max_steps ) points.at(i) = points1.at(i);
    else points.at(i) = points2.at(i-max_steps);
  }

  // ------- END geodesic stepthrough ---------------------------- //



  

  // ---------------------------- Rendering Stuff -------------------------------------- //
  // 

  sf::RenderWindow win(sf::VideoMode(W,H), "2d thang");
  win.setFramerateLimit(60);
  sf::RectangleShape screen{ sf::Vector2f(static_cast<float>(W), static_cast<float>(H)) };
  screen.setPosition(0.f, 0.f);
  if (!sf::Shader::isAvailable()) { std::cerr << "No shader support\n"; return 1; }
  sf::Shader shader;
  if (!shader.loadFromFile("main2.frag", sf::Shader::Fragment)) return 1;

  // Set Uniforms
  shader.setUniform("boundsMin",  sf::Glsl::Vec2(minx, miny));
  //shader.setUniform("boundsMax",  sf::Glsl::Vec2(maxx, maxy));
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

