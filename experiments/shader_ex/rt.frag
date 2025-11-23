#version 120

// what determines which pixel this frag is drawn on?
// what does the frag doa dn return?
// what is a .frag file, and where are these functions and objects like gl_FragCoord and distance coming from? maybe they're handled by somethng like the shader or maybe "RenderStates" object? also these classes like vec2 vec3
// so this is called once per pixel, and the resulting color is written to this gl_FragColor object which idk what it is but it handles the color of that pixel. is the only possible output of a frag object to write to this pixel color? I can't think of anything else it could do



// here in the frag, these uniforms are passed along from the shader object whenever the frag is called
uniform vec2  resolution;   // (W,H)
uniform vec2  boundsMin;    // (minx,miny)
uniform vec2  boundsMax;    // (maxx,maxy)
uniform vec3  camPos;       // camera in world coords
uniform float radius;       // circle radius in pixels

void main() {
    // Map world->pixel: x -> x - minx; y -> H - (y - miny)
    vec2 camPix = vec2(
        camPos.x - boundsMin.x,
        resolution.y - (camPos.y - boundsMin.y)
    );

    // gl_FragCoord.xy is pixel center in window coords. is this the pixel which an instance of the frag is run on? a pixel denoted by gl_FragCoord.xy? also what's the distance function, presumably some very optimized L2 norm function?
    float d = distance(gl_FragCoord.xy, camPix);

    // color vector is set to a vector of all 1s if distance is within camera and black if not? why is it from like 0 to 1 rather than 0 to 255?
    vec3 color = (d <= radius) ? vec3(1.0) : vec3(0.0);
    gl_FragColor = vec4(color, 1.0);
}

