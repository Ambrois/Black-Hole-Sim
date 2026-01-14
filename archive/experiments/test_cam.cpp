#include <iostream>
#include <SFML/Graphics.hpp>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <array>
#include <cmath>

using std::array, std::tan, std::endl;

struct _3Vec{ double x,y,z; };
// addition and scalar mult on _3Vec's
_3Vec operator+(const _3Vec &v1, const _3Vec &v2) {
  return {v1.x+v2.x, v1.y+v2.y, v1.z+v2.z};
}
_3Vec operator-(const _3Vec &v1, const _3Vec &v2) {
  return {v1.x-v2.x, v1.y-v2.y, v1.z-v2.z};
}
_3Vec operator*(double alpha, const _3Vec &v) {
  return {alpha*v.x, alpha*v.y, alpha*v.z};
}
std::ostream& operator<<(std::ostream& os, const _3Vec& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}


// normally I'd check the FOV radians make sense, but it's going to be turned into a class method called on the camera which should already be checked
// TODO
//  - 
template <size_t H, size_t W>
array<array<_3Vec, W>, H> compute_velocities(
    _3Vec c,  // camera coordinates
    _3Vec d_vec, // unit vec pointing towards pane from camera
    _3Vec h_vec, // unit vec pointing up from cam
    _3Vec w_vec, // unit vec pointing left from cam
    double d, // distance from camera to pane, not actually needed?
    double FOV_h, // horizontal FOV
    double FOV_w,
    double ray_speed
    ) {
  // computes velocities of light rays projecting from the camera into the center of each pixel of the frame
  // indexing the output is [height][width].coord, height and width from top left down/right
  array<array<_3Vec, W>, H> pixel_velocities;

  // compute top left point on frame
  double half_width = d*tan(FOV_w/2.); // don't have to compute *d bc norm cancels it out,
                                    // so these descriptions should all have "per unit distance" at the end
  double half_height = d*tan(FOV_h/2.);// *d not needed
  _3Vec top_left_dir = d*d_vec/* *d not needed */ + half_width*w_vec + half_height*h_vec;
  double pixel_width = (2.*half_width)/W;
  double pixel_height = (2.*half_height)/H;


  // loop over pixels
  for (size_t j=0; j<H; j++) {
    for (size_t i=0; i<W; i++) {
      
      // pixel center direction (pixel center minus camera)
      _3Vec dir = top_left_dir - ((2.*i+1.)/2.) * pixel_width * w_vec - ((2.*j+1.)/2.) * pixel_height * h_vec;

      double dir_norm = std::sqrt(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
      
      pixel_velocities[j][i] = (ray_speed / dir_norm) * dir;
    }
  }
  return pixel_velocities;
}


uint8_t to_u8(double x) {
  // x should be between [-1,1]
  // returns value proportional to squared magnitude
  return (x*x)*255;
}



int main() {
  const int W=500, H=500;
  sf::Image img; img.create(H, W);
  

  // make the example velocity directions
  array<array<_3Vec, W>, H> frame_velocities = compute_velocities<H,W>(
    _3Vec{0.0, 0.0, 0.0}, // camera coord
    _3Vec{1.0, 0.0, 0.0}, // d_vec
    _3Vec{0.0, 0.0, 1.0}, // h_hat
    _3Vec{0.0, -1.0, 0.0}, // w_hat
    1.0, // distance
    2.0, // FOV_h
    2.0, // FOV_w
    1.0 // speed of light
    );

  // VISUALIZE STUFF

  auto visualize_frame = [&](sf::Image &im, const array<array<_3Vec, W>, H> &pixel_vels) {
    for (size_t j=0; j<H; ++j) {
      for (size_t i=0; i<W; ++i) {

        _3Vec pixel_vel = pixel_vels[j][i];

        sf::Color c(to_u8(pixel_vel.x), to_u8(pixel_vel.y), to_u8(pixel_vel.z));

        im.setPixel(i,j,c);
      }
    }
  };

  visualize_frame(img, frame_velocities);
  sf::Texture tex;
  tex.loadFromImage(img);
  sf::Sprite spr(tex);

  sf::RenderWindow win(sf::VideoMode(W,H), "test frame");
  win.setFramerateLimit(60);

  while(win.isOpen()) {
    sf::Event e;
    while (win.pollEvent(e)) {
      if (e.type==sf::Event::Closed) win.close();
    }

    win.clear();
    win.draw(spr);
    win.display();
  }

}

