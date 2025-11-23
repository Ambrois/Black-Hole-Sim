#include <SFML/Graphics.hpp>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <iostream>






int main() {
    
    const int W=800, H=600;
    sf::Image img; img.create(W,H);

    // function to write radial gradient to image
    auto fill_gradient = [&](sf::Image& im, int green=0){

        for (int y=0; y<H; ++y)
            for (int x=0; x<W; ++x) {

                float u = x - W/2.0f, v = y - H/2.0f;
                float r = std::sqrt(u*u+v*v);

                sf::Color c(static_cast<sf::Uint8>(std::min(255.f, r/2.f)), 
                            static_cast<sf::Uint8>(std::min(255, green)), 0);

                im.setPixel(x,y,c);
            }
    };
    
    fill_gradient(img);

    
    // send to gpu
    sf::Texture tex;
    tex.loadFromImage(img);

    // turn texture to sprite
    sf::Sprite spr(tex);



    sf::RenderWindow win(sf::VideoMode(W,H), "BH Sim"); // create window
    win.setFramerateLimit(60);

    for (int i=0; win.isOpen(); i= (i+1)%250) {

        
        sf::Event e;
        while (win.pollEvent(e)) {            // poll = fetch next event if any
            if (e.type == sf::Event::Closed)  // user clicked the X
                win.close();
            if (e.type == sf::Event::KeyPressed &&
                e.key.code == sf::Keyboard::Escape) // user used keyboard and pressed escape
                win.close();


            if (e.type == sf::Event::KeyPressed &&
                e.key.code == sf::Keyboard::S)
                img.saveToFile("frame.png");

        }

        fill_gradient(img, i);
        tex.update(img);
        
        win.clear();
        win.draw(spr);
        win.display();


    }
}






