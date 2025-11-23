#pragma once

#include <array>
#include <cstddef>
#include <cstddef>

namespace physics {

template <class F>
double rk4_step_scalar(F&& f, double t0, double y0, double h) {

  double k1 = f( t0, y0 );
  double k2 = f( t0+h/2. , y0 + k1*h/2. );
  double k3 = f( t0+h/2. , y0 + k2*h/2. );
  double k4 = f( t0+h , y0 + k3*h );

  return y0 + (h/6)*(k1 + 2*k2 + 2*k3 + k4);
}



// Array versions of addition and scalar mult
template <size_t N>
std::array<double, N> operator+(const std::array<double,N> &v1, const std::array<double,N> &v2) {
  std::array<double, N> sum;
  for (size_t i=0; i<N; i++)  sum[i] = v1[i] + v2[i];
  return sum;
}

template <size_t N>
std::array<double, N> operator*(double scalar, const std::array<double,N> &v) {
  std::array<double, N> prod;
  for (size_t i=0; i<N; i++)  prod[i] = scalar * v[i];
  return prod;
}


// Vectorized RK4 step for systems of DEs
template <size_t N, class F>
std::array<double,N> rk4_step( F &&f_v, double t0, const std::array<double,N> &y0_v, double h) {

  std::array<double,N> k1_v = f_v( t0, y0_v );
  std::array<double,N> k2_v = f_v( t0+h/2,  y0_v + (h/2)*k1_v );
  std::array<double,N> k3_v = f_v( t0+h/2,  y0_v + (h/2)*k2_v );
  std::array<double,N> k4_v = f_v( t0+h,  y0_v + h*k3_v );

  return y0_v + (h/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v);
}


} // namespace physics
