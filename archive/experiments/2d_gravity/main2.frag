#version 120

#define MAX_STEPS 512
// TODO let's draw a line around the hole at 1.5*r_s
uniform vec2 boundsMin;    // (minx,miny)
uniform vec2 points[MAX_STEPS];
uniform float drawing_radius;
uniform float r_s;

struct Cart2 { float x,y; };
struct Cart2w { float x,y; }; // cartesian vec window coords
struct Pol2 { float r, phi; }; // meant to be equatorial plane of Pol3



// CONVERSION FUNCS
Pol2 vec2_2_pol(vec2 v) {
  return Pol2(v.x, v.y);
};

Cart2w cart_2_window(Cart2 v) {
  return Cart2w(
      v.x - boundsMin.x,
      v.y - boundsMin.y
      );
};

Cart2 pol_2_cart(Pol2 v) {
  return Cart2(
      v.r*sin(v.phi),
      -v.r*cos(v.phi)
      );
};

Cart2w pol_2_window(Pol2 v) {
  return cart_2_window(pol_2_cart(v));
};





void main() {
  Pol2 bh_center = Pol2(0.0, 0.0);


  vec3 color = vec3(0.0);

  // loop through array and check distances for each, set to 1 if any are within radius

  for (int i=0; i<MAX_STEPS; i++) {
    // distance of i'th element of array (vec2 version of pol2)

    Pol2 point_ = vec2_2_pol(points[i]);

    Cart2w point = pol_2_window(point_);

    float dist = distance(gl_FragCoord.xy, vec2(point.x, point.y));

    if (dist <= drawing_radius) {color = vec3(1.0);}

  }


  // draw bh, need to put bh into window coords
  Cart2w bh = pol_2_window(bh_center);
  float bh_dist = distance(gl_FragCoord.xy, vec2(bh.x, bh.y));
  if (bh_dist <= r_s) {color = vec3(0.0, 1.0, 1.0);} 

  // draw 3M line
  // convert r_s to window distance
  Cart2 rs_temp1 = Cart2(r_s, 0.);
  Cart2w rs_temp2 = cart_2_window(rs_temp1);
  float rs_temp3 = rs_temp2.x;


  if (abs(bh_dist - 1.5*r_s) <= 1.) {color = vec3(0.0, 1.0, 0.0);}

  gl_FragColor = vec4(color, 1.0);
}

