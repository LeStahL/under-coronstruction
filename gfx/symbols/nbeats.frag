#version 130

uniform float iTime;

void nbeats(out float s, in int index)
{
    if(index ==  0 )
    {
        s = round(iTime/ 0.535675 );
    }
    if(index ==  1 )
    {
        s = round(iTime/ 0.35715 );
    }
}
