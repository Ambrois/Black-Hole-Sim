#include "rk4.hpp"
#include <iostream>
#include <functional>
#include <cmath>

int main() {
    std::function<double(double, const double&)> f =
        [](double /*t*/, const double& y) { return y; }; // dy/dt = y

    double y = 1.0;
    double t = 0.0;
    double dt = 0.1;
    for (int i = 0; i < 10; ++i) {
        y = physics::rk4_step(f, t, y, dt);
        t += dt;
    }
    std::cout << "Approx y(1) = " << y << "\n";
    std::cout << "Exact  e^1 = " << std::exp(1.0) << "\n";
}

