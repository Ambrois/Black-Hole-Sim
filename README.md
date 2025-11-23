# Black Hole Visualization From Scratch

More detailed writeup WIP.
- Using C++, SFML, and GLSL
- The simulation marches a grid of light rays (one per pixel) from the camera and numerically approximates the null geodesic equations for the Schwarzschild metric. The light rays curve and hit objects, which determine the color of the pixel.
- For speed of computation, and possible because of spherical symmetry, null geodesics are computed on the equator and rotated into place

### Milestone Screenshots

- Added background stars
- Made disk color more realistic: calculate temp -> approximate black body radiation -> color
![Most Recent Screenshot](screenshots/Screenshot%20from%202025-11-22%2023-17-13.png?raw=true "screenshot")

- Object bounds rendering
![Older Screenshot](screenshots/Screenshot%20from%202025-11-17%2017-32-40.png?raw=true "screenshot")

- Got light to curve coherently in 3D
![Older Screenshot](screenshots/Screenshot%20from%202025-11-15%2015-33-02.png?raw=true "screenshot")

- Figured out null geodesic equations
![Older Screenshot](screenshots/Screenshot%20from%202025-11-04%2018-38-16.png?raw=true "screenshot")

- Built euclidean engine first
![Older Screenshot](screenshots/Screenshot%20from%202025-10-21%2019-07-02.png?raw=true "screenshot")

- Drew a circle!
![Older Screenshot](screenshots/Screenshot%20from%202025-10-18%2022-30-28.png?raw=true "screenshot")


TODO:
- the center of the bh has trouble rendering bc parameterizing with phi, make step size also adjust for phi
- if you move, your camera turns because I've yet to implement parallel transport of the camera directions
- accretion disk
  - should be flared out bc intertia
  - needs to use intersections fully, the ray skipped segments are noticable

