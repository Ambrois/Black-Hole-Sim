# From Scratch Black Hole Visualization

### Screenshots

![screenshot](/screenshots/screenshot.jpg?raw=true "screenshot1")


TODO:
- the center of the bh has trouble rendering bc parameterizing with phi, make step size also adjust for phi
- if you move, your camera turns because I've yet to implement parallel transport of the camera directions
- accretion disk
  - should be flared out bc intertia
  - needs to use intersections fully, the ray skipped segments are noticable
 
Lessons learned:
- I didn't know about compute shaders so I made the computation happen in a fragment shader, which could be more efficient
- For confusing math, use lots of structs to maintain crystal clear mathematical clarity, as long as performance is good enough
- 
