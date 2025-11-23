uniform vec2 camera_window_pos;

void main() {

  float dist_from_cam = distance(camera_window_pos, gl_FragCoord.xy);

  vec3 color;
  if (dist_from_cam <= 20.f) {color = vec3(1.0);}
  else {color = vec3(0.0);}

  gl_FragColor = vec4(color, 1.0);
}
