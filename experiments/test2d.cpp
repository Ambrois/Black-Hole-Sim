#include <SFML/Graphics.hpp>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>


constexpr double PI = 3.14159265358979323846;

// -------------- Camera, Objects ------------------ //

// world bounds
constexpr double 
  MINX=0, MAXX=1000,
  MINY=0, MAXY=1000,
  minz=0, maxz=1000;


struct _3Vec {
  double x, y, z;
  double norm() {return std::sqrt(x*x+y*y+z*z); }
};
_3Vec operator+(_3Vec v1, _3Vec v2) { return {v1.x+v2.x, v1.y+v2.y, v1.z+v2.z}; }
_3Vec operator-(_3Vec v1, _3Vec v2) { return {v1.x-v2.x, v1.y-v2.y, v1.z-v2.z}; }
_3Vec operator*(double alpha, _3Vec v) { return {alpha*v.x, alpha*v.y, alpha*v.z}; }
_3Vec operator*(_3Vec v, double alpha) { return {alpha*v.x, alpha*v.y, alpha*v.z}; }
std::ostream& operator<<(std::ostream& os, _3Vec v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}




class Camera {
  // TODO
  //  - add turn function
  //    - in what form are mouse inputs?
  //  - add method to compute directions
  //
  private:
    _3Vec c;  // camera coords x(left/right) y(forward/back) horizontal, z vertical
    _3Vec d_vec{0.0, 1.0, 0.0}; // facing this dir, pointer of right hand
    _3Vec w_vec{-1.0, 0.0, 0.0}; // middle finger
    _3Vec h_vec{0.0, 0.0, 1.0}; // thumb
    double d{1.};
    double FOV_h{1.};
    double FOV_w{1.};
    double ray_speed{1.};
    double wasd_sensitivity{5.};
    double turn_sensitivity{0.002};
    const double TWO_PI = PI * 2.0;
    
  public:

    Camera(double x, double y, double z)
    : c{x,y,z} {};

    void wasd(char key) {
      if (key=='w')      { c = c + d_vec * wasd_sensitivity; }
      else if (key=='a') { c = c + w_vec * wasd_sensitivity; }
      else if (key=='s') { c = c - d_vec * wasd_sensitivity; }
      else if (key=='d') { c = c - w_vec * wasd_sensitivity; }
    }

    void manage_inputs() {
      // TODO add turning
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) wasd('w');
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) wasd('a');
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) wasd('s'); 
      if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) wasd('d');
    }

    _3Vec get_coords() {return c;}

};

// --------------- END Camera, Objects ---------------- //





// --------------- Drawing and Rendering -------------- //

// TODO move to frag
void draw_x_y(sf::Image &img, Camera &camera) {
  // we'll just use world coords as pixels
  // so we need to normalize them so pixels go from 0-n
  size_t H = MAXY - MINY;
  size_t W = MAXX - MINX;

  // func to transition from world coords to pixel coords
  auto to_pixel = [H](_3Vec world_coords) -> std::pair<int,int> {
    return {
      static_cast<int>( world_coords.x - MINX ),
      static_cast<int>( H - world_coords.y + MINY )};
  };


  std::pair<int,int> cam_loc_p = to_pixel(camera.get_coords());

  for (size_t yp=0; yp<H; ++yp) { //p for pixel
    for (size_t xp=0; xp<W; ++xp) {

      // print a circle for the camera, TODO draw line for direction

      int xp_dist = (int)xp - cam_loc_p.first;
      int yp_dist = (int)yp - cam_loc_p.second;
      int square_proximity_to_cam_p = xp_dist*xp_dist + yp_dist*yp_dist ;

      sf::Color color(0,0,0);
      
      // camera circle
      if (square_proximity_to_cam_p<=20*20) {
        color.r = 255; color.g = 255; color.b = 255;
      }

      img.setPixel(xp,yp,color);
    }
  }

}

// --------------- END Drawing and Rendering -------------- //

void manage_events(
    sf::Event &e,
    sf::RenderWindow &win
    ) {
    while (win.pollEvent(e)) {
        // close
        if (e.type == sf::Event::Closed)
            win.close();
        if (e.type == sf::Event::KeyPressed &&
            e.key.code == sf::Keyboard::Escape)
            win.close();
    }
}


//  ----------------------- OLD CODE -------------------------- //

int main() {
    
  const size_t H = MAXY - MINY;
  const size_t W = MAXX - MINX;

  // make the quad which we're going to cover the screen w/ and apply the state to
  sf::RectangleShape screen{ sf::Vector2f{static_cast<float>(W), static_cast<float>(H)} };
  screen.setPosition(0.f,0.f);


  // make a shader object to store the frag code
  if (!sf::Shader::isAvailable()) {std::cerr << "Your shit has no shader!\n"; return 1;}
  sf::Shader shader;
  // load frag code
  shader.loadFromFile("test2d.frag", sf::Shader::Fragment);

  // make state object to put shader into to apply in draw
  sf::RenderStates states;
  states.shader = &shader;

  // init camera
  Camera camera(100,100,100);

  // create window
  sf::RenderWindow win(sf::VideoMode(W,H), "2d Camera Position");
  win.setFramerateLimit(60);
  

  while (win.isOpen()) {
    
    sf::Event e;
    manage_events(e, win);
    camera.manage_inputs();

    // set uniform for updated cam position
    _3Vec cam_world_pos = camera.get_coords();
    float cam_win_pos_x = cam_world_pos.x - MINX;
    float cam_win_pos_y = H - (MAXY - cam_world_pos.y);

    shader.setUniform("camera_window_pos", sf::Glsl::Vec2{cam_win_pos_x, cam_win_pos_y});

    win.clear();
    win.draw(screen, states); // rectangle and renderstate
    win.display();

  }
}


