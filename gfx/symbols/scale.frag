#version 130
uniform float iTime;
void scale(out float s)
{
//     if(iTime >=  0.0  && iTime <  34.2857 )
//     {
//         s = mod(iTime+.3- 0.0 , 0.535675 )- 0.2678375 ;
//         s = smoothstep( -0.04463958333333334 ,0.,s)*(1.-smoothstep(0., 0.13391875 ,s));
//     }
    if(iTime >=  34.2857  && iTime <  98.5714 )
    {
        s = mod(iTime+.3- 34.2857 , 0.35715 )- 0.178575 ;
        s = smoothstep( -0.0297625 ,0.,s)*(1.-smoothstep(0., 0.0892875 ,s));
    }
}

