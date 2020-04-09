#version 130

uniform vec2 iResolution;
uniform float iProgress;

out vec4 gl_FragColor;

const float pi = acos(-1.);
const vec3 c = vec3(1.,0.,-1.);
const float ssize = .6;


void rand(in vec2 x, out float n);
void lfnoise(in vec2 t, out float n);
void smoothmin(in float a, in float b, in float k, out float dst);
void dsmoothvoronoi(in vec2 x, in float sm, out float d, out vec2 z);
void dbox3(in vec3 x, in vec3 b, out float d);
void zextrude(in float z, in float d2d, in float h, out float d);
void add(in vec2 sda, in vec2 sdb, out vec2 sdf);

void main_scene(in vec3 x, out vec2 sdf)
{
    vec2 na;
    lfnoise(12.*x.xy, na.x);
    lfnoise(12.*x.xy-1337., na.y);
    na = .5+.5*na;
    
    // Floor structure
    float v;
    vec2 vi;
    dsmoothvoronoi(12.*x.xy-.8*na, .3, v, vi);
    v /= 12.;
    v = abs(v) - .0001;
    zextrude(x.z+.01, v, .02, v);
    
    float d = x.z+.01;
    smoothmin(d, v, .06*na.x, d);
    
    // Floor fine structure
    dsmoothvoronoi(32.*x.xy-2.4*na, .3, v, vi);
    v /= 32.;
    v = abs(v) - .0001;
    zextrude(x.z+.01+d, v, .001, v);
    
    smoothmin(d, v, .018*na.y, d);
    
    // Scene bounds (floor and wall)
    sdf = vec2(abs(d-.01)-.001,1.);
    add(sdf, vec2(d, 3.), sdf);
    add(sdf, vec2(abs(x.z-1.)-.001, 1.), sdf);
    add(sdf, vec2(-x.z+1.01, 3.), sdf);
    
    // Loading bar
    dbox3(x-vec3(0.,-.1,.05), vec3(.5,.05,.05), d);
    d = abs(d)-.001;
    smoothmin(d, sdf.x, .001,  sdf.x);
    add(sdf, vec2(d,2.), sdf);
    
    dbox3(x-vec3(-.5+.5*iProgress,-.1,.05), vec3(.5*iProgress, .045,.045), d);
    d = abs(d)-.001;
    add(sdf, vec2(d,0.), sdf);
    
    dbox3(x-vec3(.5-.5*(1.-iProgress),-.1,.05), vec3(.5*(1.-iProgress), .045,.045), d);
    d = abs(d)-.001;
    add(sdf, vec2(d,4.), sdf);
}

#define normal(o, t)void o(in vec3 x, out vec3 n, in float dx){vec2 s, na;t(x,s);t(x+dx*c.xyy, na);n.x = na.x;t(x+dx*c.yxy, na);n.y = na.x;t(x+dx*c.yyx, na);n.z = na.x;n = normalize(n-s.x);} 
// FIXME #CRLF  
normal(main_normal, main_scene)

#define march(id, sid)void id(out vec3 x, in vec3 o, inout float d, in vec3 dir, in int N, out int i, out vec2 s){for(i = 0; i<N; ++i){x=o+d*dir;sid(x,s);if(s.x < 1.e-4) return;d+=min(s.x,1.e-2);}}
// FIXME #CRLF
march(march_main, main_scene)

float sm(in float d)
{
    return smoothstep(1.5/iResolution.y, -1.5/iResolution.y, d);
}

void floor_texture(in vec2 uv, out vec3 col)
{
    vec2 na;
    lfnoise(12.*uv, na.x);
    lfnoise(12.*uv-1337., na.y);
    na = .5+.5*na;
    
    // Floor structure
    float v;
    vec2 vi;
    dsmoothvoronoi(12.*uv-.8*na, .3, v, vi);
    v /= 12.;
    v = abs(v) - .0001;

    float d = v;
    
    // Floor fine structure
    dsmoothvoronoi(32.*uv-2.4*na, .3, v, vi);
    v /= 32.;
    v = abs(v) - .0001;
    
    smoothmin(d, v, .018*na.y, d);
    
    col = mix(c.yyy, 2.*vec3(0.83,0.02,0.02), sm(d/2.5));
    col = mix(col, 2.*vec3(0.83,0.82,0.02), sm(d));

}

void illuminate(in vec3 x, in vec3 n, in vec3 dir, in vec3 l, inout vec3 col, in vec2 s)
{
    if(s.y == 0.) // Progress inside
    {
        col = 1.2*mix(c.xxy, c.xyy, .3);
        col = 1.1*col
            + 1.1*col*dot(l, n)
            + 1.1*col*pow(abs(dot(reflect(l,n),dir)),2.);
    }
    else if(s.y == 1.) // Refracting floor surface
    {
        floor_texture(x.xy, col);
        col = .1*col
            + .1*col*dot(l, n)
            + 1.1*col*pow(abs(dot(reflect(l,n),dir)),2.);
    }
    else if(s.y == 2.) // Foreground material
    {
        col = vec3(0.83,0.00,0.02);
        col = 1.*col
            + 1.1*col*dot(l, n)
            + 1.1*col*pow(abs(dot(reflect(l,n),dir)),2.);
    }
    else if(s.y == 3.) // Reflecting floor
    {
        col = c.xxx;
        col = .1*col
            + .1*col*dot(l, n)
            + 1.1*col*pow(abs(dot(reflect(l,n),dir)),2.);
    }
    if(s.y == 4.) // Progress inside
    {
        col = .2*c.xxx;
        col = 1.1*col
            + 1.1*col*dot(l, n)
            + 1.1*col*pow(abs(dot(reflect(l,n),dir)),2.);
    }
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 uv = (fragCoord-.5*iResolution.xy)/iResolution.y,
        s, ss,
        s0;
    
    vec3 col = c.yyy,
        o = c.yzx,
        o0 = o,
        r = c.xyy,
        t = c.yyy, 
        u = cross(normalize(t-o),-r),
        dir,
        n, 
        x,
        c1 = c.yyy,
        l,
        dir0,
        x0,
        c2,
        n0;
    int N = 250,
        i;
    float d = 0., d0;
    
    t += uv.x * r + uv.y * u;
    dir = normalize(t-o);
    
    march_main(x, o, d, dir, N, i, s);
    
    l = normalize(c.xzx);
    
    vec3 y = vec3(mod(x.xy, ssize)-.5*ssize, x.z),
        yi = round((x-y)/ssize+.5);
    
    if(i<N)
    {
        main_normal(x, n, 5.e-5);
        illuminate(x, n, dir, normalize(l), col, s);
        
        s0 = s;
        x0 = x;
        o0 = o;
        d0 = d;
        dir0 = dir;
        n0 = n;
        
        o = x;
        d = 1.e-2;
        ss = s;
        
        for(int j = 0; j < 5; ++j)
        {
            if(s.y == 1.) // Transparent floor surface
                dir = refract(dir, n, .95);
            else if(s.y == 2.) // Transparent objects
                dir = refract(dir, n, .95);
            else if(s.y == 3.) // Reflection on floor
                dir = reflect(dir, n);

            march_main(x, o, d, dir, N, i, s);

            if(i<N)
            {
				main_normal(x, n, 5.e-5);
                illuminate(x, n, dir, normalize(l), c1, s);
                
                if(s0.y == 1.) col = mix(col, c1, .1);
                else if(s0.y == 2.) col = mix(col, c1, .7);
                else col = c1;
            }
            
            o = x;
            d = 1.e-2;
            ss = s;
        }
    } else d0 = -1.;

    o = o0;
    dir = dir0;
    x = x0;
    s = s0;
    
    // Soft shadow
    {
        o = x;
        dir = normalize(l-o);
        d = 1.e-2;
        
        march_main(x, o, d, dir, N, i, s);
        
        if(i<N)
        {
            if(s.y == 2.) col *= .6;
        }
    }

    // Ambient
    col *= (1.+1./length(x-l));
    
    // Gamma
    col *= 2.*col;
    
    fragColor = vec4(clamp(col,0.,1.),1.);
}

void main()
{
    mainImage(gl_FragColor, gl_FragCoord.xy);
}
