# From Scratch Black Hole Visualization

### Progress Screenshots

- Added background stars
- Made disk color more realistic: calculate temp -> approximate black body radiation -> color
![Most Recent Screenshot](screenshots/Screenshot%20from%202025-11-22%2023-17-13.png?raw=true "screenshot")

- Improved object bounds rendering
![Older Screenshot](screenshots/Screenshot%20from%202025-11-17%2017-32-40.png?raw=true "screenshot")

- Finally got light to curve right in 3D
![Older Screenshot](screenshots/Screenshot%20from%202025-11-15%2015-33-02.png?raw=true "screenshot")

Figuring out null geodesic equations
![Older Screenshot](screenshots/Screenshot%20from%202025-11-04%2018-38-16.png?raw=true "screenshot")

Building euclidean engine first
![Older Screenshot](screenshots/Screenshot%20from%202025-10-21%2019-07-02.png?raw=true "screenshot")


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
