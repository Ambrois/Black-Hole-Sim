#include <SFML/Graphics.hpp>
#include <iostream>

constexpr float minx=0,  maxx=1000;
constexpr float miny=0,  maxy=1000;

struct Vec3 { float x,y,z; };// renamed to Vec3 rather than _3Vec, ig this is fine

class Camera {
  Vec3 c;
public:
  Camera(float x,float y,float z): c{x,y,z} {}
  void wasd() {
    const float step = 10.f;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::W)) c.y += step;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::S)) c.y -= step;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::A)) c.x -= step;
    if (sf::Keyboard::isKeyPressed(sf::Keyboard::D)) c.x += step;
  }
  Vec3 pos() const { return c; }
};

// ok, in the main function, let's go over what each line does
int main() {
  // we make the pixel width and height bounds from the world bound size
  const unsigned W = static_cast<unsigned>(maxx - minx);
  const unsigned H = static_cast<unsigned>(maxy - miny);

  // make a render window again
  sf::RenderWindow win(sf::VideoMode(W,H), "Shader demo");
  win.setFramerateLimit(60);

  // Fullscreen quad (brace init avoids most-vexing parse)
  // we make a rectangle shape, which is meant to be the screen.
  //  TODO questions
  //    - how does this get used?
  //    - how will it be positioned into the world?
  // it's initialized with a vector of 2 floats, so it's an abstract rectangle of fixed size w/ no particular position or orientation in the world or the pixel space.
  sf::RectangleShape screen{ sf::Vector2f(static_cast<float>(W), static_cast<float>(H)) };
  // we set the position, one of the corners?
  screen.setPosition(0.f, 0.f);

  // Shader
  // if shader not available error
  if (!sf::Shader::isAvailable()) { std::cerr << "No shader support\n"; return 1; }
  sf::Shader shader;
  // if shader is available, load sf::Shader::Fragment from the rt.frag file, else error.
  if (!shader.loadFromFile("rt.frag", sf::Shader::Fragment)) { std::cerr << "Load rt.frag failed\n"; return 1; }

  // Camera
  Camera cam(100.f,100.f,100.f);

  // Static uniforms
  // what is this? are uniforms constant variables or something that are accessible by every instance of the shader/frag/whatever? why do they have to be created in such a weird way? why not like declare a constant and pass it along to the frag like its a lambda function, idk I'll keep reading.
  shader.setUniform("resolution", sf::Glsl::Vec2(float(W), float(H)));
  shader.setUniform("boundsMin",  sf::Glsl::Vec2(minx, miny));
  shader.setUniform("boundsMax",  sf::Glsl::Vec2(maxx, maxy));
  shader.setUniform("radius", 20.f);

  // what is states used for? what is meant by attaching the shader?
  sf::RenderStates states;     // use RenderStates to attach shader
  states.shader = &shader; // we pass the shader object into the RenderStates object for some reason

  // main loop
  while (win.isOpen()) {
    for (sf::Event e; win.pollEvent(e); ) {
      if (e.type == sf::Event::Closed) win.close();
      if (e.type == sf::Event::KeyPressed && e.key.code == sf::Keyboard::Escape) win.close();
    }

    // handle cam wasd, smart to build this into the method
    cam.wasd();

    // get cam position
    auto p = cam.pos();

    // make another uniform for the current camera position
    shader.setUniform("camPos", sf::Glsl::Vec3(p.x, p.y, p.z));

    win.clear();
    // states is used here to draw, so the object lets us render... states? idk what its doing here. I assume this line is where the shader is called and a frag is computed for each pixel, and then compiled by something into the final product.
    win.draw(screen, states);  // correct draw call
    win.display();
  }
}

