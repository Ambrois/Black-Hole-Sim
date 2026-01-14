#include<iostream>
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
int main() {
  std::cout << sf::Shader::isAvailable() << std::endl;
}
