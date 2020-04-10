#version 130
const float PI = radians(180.);
const float TAU = 2.*PI;
float clip(float a) { return clamp(a,-1.,1.); }
float smstep(float a, float b, float x) {return smoothstep(a, b, clamp(x, a, b));}
float theta(float x) { return smstep(0.,1e-3,x); }
float _sin(float a) { return sin(TAU * mod(a,1.)); }
float _sin_(float a, float p) { return sin(TAU * mod(a,1.) + p); }
float _sq_(float a,float pwm) { return sign(2.*fract(a) - 1. + pwm); }
float _tri(float a) { return (4.*abs(fract(a)-.5) - 1.); }
float freqC1(float note){ return 32.7 * exp2(note/12.); }
float minus1hochN(int n) { return (1. - 2.*float(n % 2)); }
float minus1hochNminus1halbe(int n) { return sin(.5*PI*float(n)); }
float pseudorandom(float x) { return fract(sin(dot(vec2(x),vec2(12.9898,78.233))) * 43758.5453); }
float fhelp(float x) { return 1. + .333*x; } // 1. + .33333*x + .1*x*x + .02381*x*x*x + .00463*x*x*x*x;
float linmix(float x, float a, float b, float y0, float y1) { return mix(y0,y1,clamp(a*x+b,0.,1.)); }
float s_atan(float a) { return .636 * atan(a); }
float s_moothmin(float a, float k) {
    float ha = max(1.-2.*abs(abs(a)-1.), 0.);
    return a >= 0. ? min(a, 1.) - .5/6.*ha*ha*ha : max(a, -1.) + .5/6.*ha*ha*ha;
}
float s_moothmin(float a) { return s_moothmin(a,.5); }

#define SONGLENGTH 95.2403
#define NTIME 3
const float pos_B[3] = float[3](0.,16.,60.);
const float pos_t[3] = float[3](0.,34.2857,94.2857);
const float pos_BPS[2] = float[2](.4667,.7333);
const float pos_SPB[2] = float[2](2.1427,1.3637);
float BPS, SPB, BT;

float Tsample;

#define filterthreshold 1.e-3

//TEXCODE

float lpnoise(float t, float fq)
{
    t *= fq;
    float tt = fract(t);
    float tn = t - tt;
    return mix(pseudorandom(floor(tn) / fq), pseudorandom(floor(tn + 1.0) / fq), smstep(0.0, 1.0, tt));
}

float env_AHDSR(float x, float L, float A, float H, float D, float S, float R)
{
    return (x<A ? x/A : x<A+H ? 1. : x<A+H+D ? (1. - (1.-S)*(x-H-A)/D) : x<=L-R ? S : x<=L ? S*(L-x)/R : 0.);
}

float env_AHDSRexp(float x, float L, float A, float H, float D, float S, float R)
{
    float att = pow(x/A,8.);
    float dec = S + (1.-S) * exp(-(x-H-A)/D);
    float rel = (x <= L-R) ? 1. : pow((L-x)/R,4.);
    return (x < A ? att : x < A+H ? 1. : dec) * rel;
}

float waveshape(float s, float amt, float A, float B, float C, float D, float E)
{
    float w;
    float m = sign(s);
    s = abs(s);

    if(s<A) w = B * smstep(0.,A,s);
    else if(s<C) w = C + (B-C) * smstep(C,A,s);
    else if(s<=D) w = s;
    else if(s<=1.)
    {
        float _s = (s-D)/(1.-D);
        w = D + (E-D) * (1.5*_s*(1.-.33*_s*_s));
    }
    else return 1.;

    return m*mix(s,w,amt);
}

float sinshape(float x, float amt, float parts)
{
    return (1.-amt) * x + amt * sign(x) * 0.5 * (1. - cos(parts*PI*x));
}

float comp_SAW(int N, float inv_N, float PW) {return inv_N * (1. - _sin(float(N)*PW));}
float comp_TRI(int N, float inv_N, float PW) {return N % 2 == 0 ? .1 * inv_N * _sin(float(N)*PW) : inv_N * inv_N * (1. - _sin(float(N)*PW));}
float comp_SQU(int N, float inv_N, float PW) {return inv_N * (minus1hochN(N) * _sin(.5*float(N)*PW + .25) - 1.);}
float comp_HAE(int N, float inv_N, float PW) {return N % 2 == 0 ? 0. : inv_N * (1. - minus1hochNminus1halbe(N))*_sin(PW);}
float comp_OBO(int N, float inv_N, float PW) {return sqrt(inv_N) * (1. + _sin(float(N)*(1.5+PW)+.5*PI));}

float MADD(float t, float f, float p0, int NMAX, int NINC, float MIX, float CO, float NDECAY, float RES, float RES_Q, float DET, float PW, float LOWCUT, float keyF)
{
    float ret = 0.;
    float f_ = keyF > .99 ? 1. : (keyF < 1.e-3 ? f : pow(f, 1.-keyF));
    float INR = f_/CO;
    float IRESQ = 1./(RES_Q*f_);

    float p = f*t;
    float float_N, inv_N, comp_mix, filter_N;
    for(int N = 1 + int(LOWCUT/f - 1.e-3); N<=NMAX; N+=NINC)
    {
        float_N = float(N);
        inv_N = 1./float_N;
        comp_mix = MIX < -1. ? (MIX+2.) * comp_SAW(N,inv_N,PW)  - (MIX+1.) * comp_OBO(N,inv_N,PW)
                 : MIX <  0. ? (MIX+1.) * comp_TRI(N,inv_N,PW)  -     MIX  * comp_SAW(N,inv_N,PW)
                 : MIX < 1. ? (1.-MIX) * comp_TRI(N,inv_N,PW)  +     MIX  * comp_SQU(N,inv_N,PW)
                            : (MIX-1.) * comp_HAE(N,inv_N,PW)  + (2.-MIX) * comp_SQU(N,inv_N,PW);

        if(abs(comp_mix) < 1e-4) continue;

        filter_N = pow(1. + pow(float_N*INR,NDECAY),-.5) + RES * exp(-pow((float_N*f-CO)*IRESQ,2.));

        ret += comp_mix * filter_N * (_sin_(float_N * p, p0) + _sin_(float_N * p * (1.+DET), p0));
    }
    return s_moothmin(ret);
}

float MADD(float t, float f, float p0, int NMAX, int NINC, float MIX, float CO, float NDECAY, float RES, float RES_Q, float DET, float PW, int keyF)
{
    return MADD(t, f, p0, NMAX, NINC, MIX, CO, NDECAY, RES, RES_Q, DET, PW, 0., keyF);
}

float QFM_FB(float PH, float FB) // my guessing of feedback coefficients, FB>0 'saw', FB<0 'sq'
{
    if(FB > 0.) return abs(FB) * .8*_sin(PH + .35*_sin(PH));
    else return abs(FB) * _sin(PH + .5*PI);
}

float QFM(float t, float f, float phase, float LV1, float LV2, float LV3, float LV4, float FR1, float FR2, float FR3, float FR4, float FB1, float FB2, float FB3, float FB4, float ALGO)
{
    int iALGO = int(ALGO);
    float PH1 = FR1 * f * t + phase;
    float PH2 = FR2 * f * t + phase;
    float PH3 = FR3 * f * t + phase;
    float PH4 = FR4 * f * t + phase;

    float LINK41 = 0., LINK42 = 0., LINK43 = 0., LINK32 = 0., LINK31 = 0., LINK21 = 0.;
    if(iALGO == 1)       {LINK43 = 1.; LINK32 = 1.; LINK21 = 1.;}
    else if(iALGO == 2)  {LINK42 = 1.; LINK32 = 1.; LINK21 = 1.;}
    else if(iALGO == 3)  {LINK41 = 1.; LINK32 = 1.; LINK21 = 1.;}
    else if(iALGO == 4)  {LINK42 = 1.; LINK43 = 1.; LINK31 = 1.; LINK21 = 1.;}
    else if(iALGO == 5)  {LINK41 = 1.; LINK31 = 1.; LINK21 = 1.;}
    else if(iALGO == 6)  {LINK43 = 1.; LINK32 = 1.;}
    else if(iALGO == 7)  {LINK43 = 1.; LINK32 = 1.; LINK31 = 1.;}
    else if(iALGO == 8)  {LINK21 = 1.; LINK43 = 1.;}
    else if(iALGO == 9)  {LINK43 = 1.; LINK42 = 1.; LINK41 = 1.;}
    else if(iALGO == 10) {LINK43 = 1.; LINK42 = 1.;}
    else if(iALGO == 11) {LINK43 = 1.;}

    float OP4 = LV4 * _sin(PH4 + QFM_FB(PH4, FB4));
    float OP3 = LV3 * _sin(PH3 + QFM_FB(PH3, FB3) + LINK43*OP4);
    float OP2 = LV2 * _sin(PH2 + QFM_FB(PH2, FB2) + LINK42*OP4 + LINK32*OP3);
    float OP1 = LV1 * _sin(PH1 + QFM_FB(PH1, FB1) + LINK41*OP4 + LINK31*OP3 + LINK21*OP2);

    float sum = OP1;
    if(LINK21 > 0.) sum += OP2;
    if(LINK31 + LINK32 > 0.) sum += OP3;
    if(LINK41 + LINK42 + LINK43 > 0.) sum += OP4;

    return s_moothmin(sum);
}

float sfqm_vol(float _BEAT)
{
    return _BEAT<0 ? 0. : 1.;
}
float maceboss_vol(float B)
{
    return B<0. ? 0. : (B>=0. && B<4.) ? 0. : (B>=4. && B<6.) ? linmix(B, .5, -2., 0.0, 0.2) : (B>=6. && B<8.) ? linmix(B, .5, -3., 0.2, 0.4) : (B>=8. && B<12.) ? linmix(B, .25, -2., 0.4, 1.0) : 1.;
}
float pluck7short_vol(float B)
{
    return B<0. ? 0. : (B>=0. && B<1.5) ? .02 : (B>=1.5 && B<8.) ? linmix(B, .1538, -.2308, 0.02, 0.3) : (B>=8. && B<16.) ? linmix(B, .125, -1., 0.3, 0.7) : 1.;
}
float SUBvol(float B)
{
    return B<0. ? 0. : (B>=0. && B<1.) ? linmix(B, 1., 0., 16.0, 20.0) : 1.;
}

uniform float iBlockOffset;
uniform float iSampleRate;
uniform float iTexSize;
uniform sampler2D iSequence;
uniform float iSequenceWidth;

// Read short value from texture at index off
float rshort(in float off)
{
    float hilo = mod(off, 2.);
    off = .5*off;
    vec2 ind = vec2(mod(off, iSequenceWidth), floor(off/iSequenceWidth));
    vec4 block = texelFetch(iSequence, ivec2(ind), 0);
    vec2 data = mix(block.rg, block.ba, hilo);
    return round(dot(vec2(255., 65280.), data));
}

// Read float value from texture at index off
float rfloat(int off)
{
    float d = rshort(float(off));
    float sign = floor(d/32768.),
        exponent = floor(d*9.765625e-4 - sign*32.),
        significand = d-sign*32768.-exponent*1024.;

    if(exponent == 0.)
         return mix(1., -1., sign) * 5.960464477539063e-08 * significand;
    return mix(1., -1., sign) * (1. + significand * 9.765625e-4) * pow(2.,exponent-15.);
}

#define NTRK 8
#define NMOD 30
#define NPTN 11
#define NNOT 923
#define NDRM 51

int trk_sep(int index)      {return int(rfloat(index));}
int trk_syn(int index)      {return int(rfloat(index+1+1*NTRK));}
float trk_norm(int index)   {return     rfloat(index+1+2*NTRK);}
float trk_rel(int index)    {return     rfloat(index+1+3*NTRK);}
float trk_pre(int index)    {return     rfloat(index+1+4*NTRK);}
float trk_slide(int index)  {return     rfloat(index+1+5*NTRK);} // idea for future: change to individual note_slide_time
float mod_on(int index)     {return     rfloat(index+1+6*NTRK);}
float mod_off(int index)    {return     rfloat(index+1+6*NTRK+1*NMOD);}
int mod_ptn(int index)      {return int(rfloat(index+1+6*NTRK+2*NMOD));}
float mod_transp(int index) {return     rfloat(index+1+6*NTRK+3*NMOD);}
int ptn_sep(int index)      {return int(rfloat(index+1+6*NTRK+4*NMOD));}
float note_on(int index)    {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN);}
float note_off(int index)   {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+1*NNOT);}
float note_pitch(int index) {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+2*NNOT);}
float note_pan(int index)   {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+3*NNOT);}
float note_vel(int index)   {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+4*NNOT);}
float note_slide(int index) {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+5*NNOT);}
float note_aux(int index)   {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+6*NNOT);}
float drum_rel(int index)   {return     rfloat(index+2+6*NTRK+4*NMOD+NPTN+7*NNOT);}

vec2 mainSynth(float time)
{
    float sL = 0.;
    float sR = 0.;
    float dL = 0.;
    float dR = 0.;

    if (time > SONGLENGTH) return vec2(0.);
    
    int _it;
    for(_it = 0; _it < NTIME - 2 && pos_t[_it + 1] < time; _it++);
    BPS = pos_BPS[_it];
    SPB = pos_SPB[_it];
    BT = pos_B[_it] + (time - pos_t[_it]) * BPS;

    float time2 = time - .0002;
    float sidechain = 1.;

    float amaysynL, amaysynR, amaydrumL, amaydrumR, B, Bon, Boff, Bprog, Bproc, L, tL, _t, _t2, vel, rel, pre, f, amtL, amtR, env, slide, aux;
    int tsep0, tsep1, _modU, _modL, ptn, psep0, psep1, _noteU, _noteL, syn, drum;

    for(int trk = 0; trk < NTRK; trk++)
    {
        tsep0 = trk_sep(trk);
        tsep1 = trk_sep(trk + 1);

        syn = trk_syn(trk);
        rel = trk_rel(trk) + 1.e-3;
        pre = trk_pre(trk);

        for(_modU = tsep0; (_modU < tsep1 - 1) && (BT > mod_on(_modU + 1) - pre); _modU++);
        for(_modL = tsep0; (_modL < tsep1 - 1) && (BT >= mod_off(_modL) + rel); _modL++);

        for(int _mod = _modL; _mod <= _modU; _mod++)
        {
            B = BT - mod_on(_mod) + pre;

            ptn   = mod_ptn(_mod);
            psep0 = ptn_sep(ptn);
            psep1 = ptn_sep(ptn + 1);

            for(_noteU = psep0; (_noteU < psep1 - 1) && (B > note_on(_noteU + 1)); _noteU++);
            for(_noteL = psep0; (_noteL < psep1 - 1) && (B >= note_off(_noteL) + rel); _noteL++);

            for(int _note = _noteL; _note <= _noteU; _note++)
            {
                if(syn == 139)
                {
                    drum = int(note_pitch(_note));
                    rel = drum_rel(drum) + 1.e-3;
                }

                amaysynL  = 0.;
                amaysynR  = 0.;
                amaydrumL = 0.;
                amaydrumR = 0.;

                Bon   = note_on(_note);
                Boff  = note_off(_note) + rel;
                L     = Boff - Bon;
                tL    = L * SPB;
                Bprog = max(0., B - Bon); // I DO NOT GET THIS WEIRD FIX, but Revision is approaching
                Bproc = Bprog / L;
                _t    = Bprog * SPB; 
                _t2   = _t - .0002; // this is on purpose not max(0., _t - .0002), because I hope future-QM is clever
                vel   = note_vel(_note);
                amtL  = clamp(1. - note_pan(_note), 0., 1.);
                amtR  = clamp(1. + note_pan(_note), 0., 1.);
                slide = note_slide(_note);
                aux   = note_aux(_note);

                if(syn == 139)
                {
                    env = trk_norm(trk) * theta(Bprog) * theta(L - Bprog);
                    if(drum == 0) { sidechain = min(sidechain, 1. - vel * (clamp(1.e4 * Bprog,0.,1.) - pow(Bprog/(L-rel),8.)));}
                    
                    if(drum > 0)
                    {
                        dL += amtL * s_moothmin(env * amaydrumL);
                        dR += amtR * s_moothmin(env * amaydrumR);
                    }
                }
                else
                {
                    f = freqC1(note_pitch(_note) + mod_transp(_mod));

                    if(abs(slide) > 1e-3) // THIS IS SLIDEY BIZ
                    {
                        float Bslide = trk_slide(trk);
                        float fac = slide * log(2.)/12.;
                        if (Bprog <= Bslide)
                        {
                            float help = 1. - Bprog/Bslide;
                            f *= Bslide * (fhelp(fac) - help * fhelp(fac*help*help)) / Bprog;
                        }
                        else
                        {
                            f *= 1. + (Bslide * (fhelp(fac)-1.)) / Bprog;
                        }
                    }

                    env = theta(Bprog) * (1. - smstep(Boff-rel, Boff, B));
                    if(syn == 0){amaysynL = _sin(f*_t); amaysynR = _sin(f*_t2);}
                    else if(syn == 10){
                        
                        amaysynL = .8*env_AHDSRexp(Bprog,L,.001,.3,.1,1.,.3)*(waveshape(clip(1.6*QFM((_t-0.0*(1.+2.*_sin(.15*_t))),f,0.,.00787*127.*pow(vel,12.*7.87e-3),.00787*112.*pow(vel,63.*7.87e-3),.00787*127.*pow(vel,26.*7.87e-3),.00787*96.*pow(vel,120.*7.87e-3),.5,1.,1.5,1.,.00787*0.,.00787*0.,.00787*0.,.00787*50.,8.)),.3,.2,.8,.4,.8,.8)
      +waveshape(clip(1.6*QFM((_t-2.0e-03*(1.+2.*_sin(.15*_t))),f,0.,.00787*127.*pow(vel,12.*7.87e-3),.00787*112.*pow(vel,63.*7.87e-3),.00787*127.*pow(vel,26.*7.87e-3),.00787*96.*pow(vel,120.*7.87e-3),.5,1.,1.5,1.,.00787*0.,.00787*0.,.00787*0.,.00787*50.,8.)),.3,.2,.8,.4,.8,.8)
      +waveshape(clip(1.6*QFM((_t-4.0e-03*(1.+2.*_sin(.15*_t))),f,0.,.00787*127.*pow(vel,12.*7.87e-3),.00787*112.*pow(vel,63.*7.87e-3),.00787*127.*pow(vel,26.*7.87e-3),.00787*96.*pow(vel,120.*7.87e-3),.5,1.,1.5,1.,.00787*0.,.00787*0.,.00787*0.,.00787*50.,8.)),.3,.2,.8,.4,.8,.8));
                        amaysynR = .8*env_AHDSRexp(Bprog,L,.001,.3,.1,1.,.3)*(waveshape(clip(1.6*QFM((_t2-0.0*(1.+2.*_sin(.15*_t2))),f,0.,.00787*127.*pow(vel,12.*7.87e-3),.00787*112.*pow(vel,63.*7.87e-3),.00787*127.*pow(vel,26.*7.87e-3),.00787*96.*pow(vel,120.*7.87e-3),.5,1.,1.5,1.,.00787*0.,.00787*0.,.00787*0.,.00787*50.,8.)),.3,.2,.8,.4,.8,.8)
      +waveshape(clip(1.6*QFM((_t2-2.0e-03*(1.+2.*_sin(.15*_t2))),f,0.,.00787*127.*pow(vel,12.*7.87e-3),.00787*112.*pow(vel,63.*7.87e-3),.00787*127.*pow(vel,26.*7.87e-3),.00787*96.*pow(vel,120.*7.87e-3),.5,1.,1.5,1.,.00787*0.,.00787*0.,.00787*0.,.00787*50.,8.)),.3,.2,.8,.4,.8,.8)
      +waveshape(clip(1.6*QFM((_t2-4.0e-03*(1.+2.*_sin(.15*_t2))),f,0.,.00787*127.*pow(vel,12.*7.87e-3),.00787*112.*pow(vel,63.*7.87e-3),.00787*127.*pow(vel,26.*7.87e-3),.00787*96.*pow(vel,120.*7.87e-3),.5,1.,1.5,1.,.00787*0.,.00787*0.,.00787*0.,.00787*50.,8.)),.3,.2,.8,.4,.8,.8));
                    }
                    else if(syn == 22){
                        
                        amaysynL = exp(-11.*Bprog)*env_AHDSRexp(Bprog,L,.01,0.,.1+.5*vel,.01,.4)*sinshape(clip((1.+exp(-11.*Bprog))*_tri(f*_t+.2*env_AHDSRexp(Bprog,L,.5,1.,.1,1.,.1)*clip((1.+3.)*_sq_(1.99*f*_t,.3+2.*vel+.2*(2.*fract(3.97*f*_t)-1.)))+.2*vel*env_AHDSRexp(Bprog,L,.325,1.,.1,1.,.3)*(2.*fract(3.97*f*_t)-1.))),.01*aux*exp(-11.*Bprog),5.)
      +.2*env_AHDSRexp(Bprog,L,.001,0.,.05,0.,.8)*lpnoise(_t + 0.,6000.+200.*note_pitch(_note))
      +.4*exp(-11.*Bprog)*env_AHDSRexp(Bprog,L,.325,1.,.1,1.,.3)*clip((1.+3.)*_sq_(1.99*f*_t,.3+2.*vel+.2*(2.*fract(3.97*f*_t)-1.)))*env_AHDSRexp(Bprog,L,.001,0.,.2+.2*vel,.01,.4);
                        amaysynR = exp(-11.*Bprog)*env_AHDSRexp(Bprog,L,.01,0.,.1+.5*vel,.01,.4)*sinshape(clip((1.+exp(-11.*Bprog))*_tri(f*_t2+.2*env_AHDSRexp(Bprog,L,.5,1.,.1,1.,.1)*clip((1.+3.)*_sq_(1.99*f*_t2,.3+2.*vel+.2*(2.*fract(3.97*f*_t2)-1.)))+.2*vel*env_AHDSRexp(Bprog,L,.325,1.,.1,1.,.3)*(2.*fract(3.97*f*_t2)-1.))),.01*aux*exp(-11.*Bprog),5.)
      +.2*env_AHDSRexp(Bprog,L,.001,0.,.05,0.,.8)*lpnoise(_t2 + 0.,6000.+200.*note_pitch(_note))
      +.4*exp(-11.*Bprog)*env_AHDSRexp(Bprog,L,.325,1.,.1,1.,.3)*clip((1.+3.)*_sq_(1.99*f*_t2,.3+2.*vel+.2*(2.*fract(3.97*f*_t2)-1.)))*env_AHDSRexp(Bprog,L,.001,0.,.2+.2*vel,.01,.4);
                        env = theta(Bprog)*pow(1.-smstep(Boff-rel, Boff, B),6.);
                    }
                    else if(syn == 109){
                        time2 = time-.1; _t2 = _t-.1;
                        amaysynL = pluck7short_vol(BT)*((vel*sinshape(QFM(_t,f,0.,.00787*125.,.00787*env_AHDSR(Bprog,L,.0001,.047,.01,.404,0.)*20.,.00787*env_AHDSR(Bprog,L,.0001,.151,.071,.069,0.)*110.,.00787*env_AHDSR(Bprog,L,.0001,.232,.08,.003,0.)*65.,.999,1.,1.+.0799*(.5+(.5*_sin(.18*Bprog))),2.,.00787*109.,.00787*21.,.00787*94.,.00787*0.,11.),.03*aux,3.)*env_AHDSR(Bprog,L,.0001,.03,.167,.796,.114))+_sq_(.501*f*_t,.4+.3*(.5+(.5*_sin(.8*Bprog))))+.6*clip((1.+.2*aux)*_sin(.25*f*_t)));
                        amaysynR = pluck7short_vol(BT)*((vel*sinshape(QFM(_t2,f,0.,.00787*125.,.00787*env_AHDSR(Bprog,L,.0001,.047,.01,.404,0.)*20.,.00787*env_AHDSR(Bprog,L,.0001,.151,.071,.069,0.)*110.,.00787*env_AHDSR(Bprog,L,.0001,.232,.08,.003,0.)*65.,.999,1.,1.+.0799*(.5+(.5*_sin(.18*Bprog))),2.,.00787*109.,.00787*21.,.00787*94.,.00787*0.,11.),.03*aux,3.)*env_AHDSR(Bprog,L,.0001,.03,.167,.796,.114))+_sq_(.501*f*_t2,.4+.3*(.5+(.5*_sin(.8*Bprog))))+.6*clip((1.+.2*aux)*_sin(.25*f*_t2)));
                    }
                    else if(syn == 119){
                        time2 = time-0.02; _t2 = _t-0.02;
                        amaysynL = (MADD(_t,f,0.,12,1,.92,2513.*env_AHDSRexp(Bprog,L,.31,0.,.1,1.,0.3),65.1,1.,3.,.008,.25,0.,1)*env_AHDSRexp(Bprog,L,.145+aux,.13,.38,.27,0.3));
                        amaysynR = (MADD(_t2,f,0.,12,1,.92,2513.*env_AHDSRexp(Bprog,L,.31,0.,.1,1.,0.3),65.1,1.,3.,.008,.25,0.,1)*env_AHDSRexp(Bprog,L,.145+aux,.13,.38,.27,0.3));
                        env = theta(Bprog)*pow(1.-smstep(Boff-rel, Boff, B),4.);
                    }
                    else if(syn == 133){
                        
                        amaysynL = ((QFM((_t-0.0*(1.+3.*_sin(.1*_t))),f,0.,.00787*110.,.00787*env_AHDSRexp(Bprog,L,.195,.003,.035,.383,0.)*78.,.00787*env_AHDSRexp(Bprog,L,.144,.062,.01,.425,0.)*47.,.00787*env_AHDSRexp(Bprog,L,.003,.018,.18,.126,0.)*70.,.5,1.,1.001,1.,.00787*118.,.00787*10.,.00787*103.,.00787*83.,5.)+QFM((_t-4.0e-03*(1.+3.*_sin(.1*_t))),f,0.,.00787*110.,.00787*env_AHDSRexp(Bprog,L,.195,.003,.035,.383,0.)*78.,.00787*env_AHDSRexp(Bprog,L,.144,.062,.01,.425,0.)*47.,.00787*env_AHDSRexp(Bprog,L,.003,.018,.18,.126,0.)*70.,.5,1.,1.001,1.,.00787*118.,.00787*10.,.00787*103.,.00787*83.,5.)+QFM((_t-8.0e-03*(1.+3.*_sin(.1*_t))),f,0.,.00787*110.,.00787*env_AHDSRexp(Bprog,L,.195,.003,.035,.383,0.)*78.,.00787*env_AHDSRexp(Bprog,L,.144,.062,.01,.425,0.)*47.,.00787*env_AHDSRexp(Bprog,L,.003,.018,.18,.126,0.)*70.,.5,1.,1.001,1.,.00787*118.,.00787*10.,.00787*103.,.00787*83.,5.))*env_AHDSRexp(Bprog,L,.001,.034,.148,.677,.094));
                        amaysynR = ((QFM((_t2-0.0*(1.+3.*_sin(.1*_t2))),f,0.,.00787*110.,.00787*env_AHDSRexp(Bprog,L,.195,.003,.035,.383,0.)*78.,.00787*env_AHDSRexp(Bprog,L,.144,.062,.01,.425,0.)*47.,.00787*env_AHDSRexp(Bprog,L,.003,.018,.18,.126,0.)*70.,.5,1.,1.001,1.,.00787*118.,.00787*10.,.00787*103.,.00787*83.,5.)+QFM((_t2-4.0e-03*(1.+3.*_sin(.1*_t2))),f,0.,.00787*110.,.00787*env_AHDSRexp(Bprog,L,.195,.003,.035,.383,0.)*78.,.00787*env_AHDSRexp(Bprog,L,.144,.062,.01,.425,0.)*47.,.00787*env_AHDSRexp(Bprog,L,.003,.018,.18,.126,0.)*70.,.5,1.,1.001,1.,.00787*118.,.00787*10.,.00787*103.,.00787*83.,5.)+QFM((_t2-8.0e-03*(1.+3.*_sin(.1*_t2))),f,0.,.00787*110.,.00787*env_AHDSRexp(Bprog,L,.195,.003,.035,.383,0.)*78.,.00787*env_AHDSRexp(Bprog,L,.144,.062,.01,.425,0.)*47.,.00787*env_AHDSRexp(Bprog,L,.003,.018,.18,.126,0.)*70.,.5,1.,1.001,1.,.00787*118.,.00787*10.,.00787*103.,.00787*83.,5.))*env_AHDSRexp(Bprog,L,.001,.034,.148,.677,.094));
                    }
                    else if(syn == 135){
                        time2 = time-.5; _t2 = _t-.5;
                        amaysynL = .4*env_AHDSRexp(Bprog,L,.02,0.,.01,.9,.01)*(_sin(.5*f*floor(24333.*(_t-0.0*(1.+8.5*_sin(.2*_t)))+.5)/24333.)+_sin(.499*f*floor(13400.*(_t-0.0*(1.+8.5*_sin(.2*_t)))+.5)/13400.)
      +_sin(.5*f*floor(24333.*(_t-1.0e-03*(1.+8.5*_sin(.2*_t)))+.5)/24333.)+_sin(.499*f*floor(13400.*(_t-1.0e-03*(1.+8.5*_sin(.2*_t)))+.5)/13400.))*SUBvol(BT);
                        amaysynR = .4*env_AHDSRexp(Bprog,L,.02,0.,.01,.9,.01)*(_sin(.5*f*floor(24333.*(_t2-0.0*(1.+8.5*_sin(.2*_t2)))+.5)/24333.)+_sin(.499*f*floor(13400.*(_t2-0.0*(1.+8.5*_sin(.2*_t2)))+.5)/13400.)
      +_sin(.5*f*floor(24333.*(_t2-1.0e-03*(1.+8.5*_sin(.2*_t2)))+.5)/24333.)+_sin(.499*f*floor(13400.*(_t2-1.0e-03*(1.+8.5*_sin(.2*_t2)))+.5)/13400.))*SUBvol(BT);
                    }
                    else if(syn == 137){
                        time2 = time-.005; _t2 = _t-.005;
                        amaysynL = sfqm_vol(BT)*env_AHDSRexp(Bprog,L,.02,0.,.01,.95,.001)*s_atan(s_atan(5.*QFM(_t,f,0.,.00787*127.,.00787*50.*env_AHDSRexp(Bprog,L,.185,0.,.1,1.,.001),.00787*60.,.00787*0.,.25+.25*.04*env_AHDSRexp(Bprog,L,.225,0.,.1,1.,.001)*(.5+(.001*_sin(1.*Bprog))),.5+.04*env_AHDSRexp(Bprog,L,.225,0.,.1,1.,.001)*(.5+(.001*_sin(1.*Bprog))),.4999,1.,.00787*-255.,.00787*0.,.00787*0.,.00787*0.,9.))+s_atan(5.*QFM(_t,f,0.,.00787*127.,.00787*50.*env_AHDSRexp(Bprog,L,.185,0.,.1,1.,.001),.00787*60.,.00787*0.,.25+.25*.04*env_AHDSRexp(Bprog,L,.225,0.,.1,1.,.001)*(.5+(.001*_sin(1.*Bprog))),.5+.04*env_AHDSRexp(Bprog,L,.225,0.,.1,1.,.001)*(.5+(.001*_sin(1.*Bprog))),.4999,1.,.00787*-255.,.00787*0.,.00787*0.,.00787*0.,9.)));
                        amaysynR = sfqm_vol(BT)*env_AHDSRexp(Bprog,L,.02,0.,.01,.95,.001)*s_atan(s_atan(5.*QFM(_t2,f,0.,.00787*127.,.00787*50.*env_AHDSRexp(Bprog,L,.185,0.,.1,1.,.001),.00787*60.,.00787*0.,.25+.25*.04*env_AHDSRexp(Bprog,L,.225,0.,.1,1.,.001)*(.5+(.001*_sin(1.*Bprog))),.5+.04*env_AHDSRexp(Bprog,L,.225,0.,.1,1.,.001)*(.5+(.001*_sin(1.*Bprog))),.4999,1.,.00787*-255.,.00787*0.,.00787*0.,.00787*0.,9.))+s_atan(5.*QFM(_t2,f,0.,.00787*127.,.00787*50.*env_AHDSRexp(Bprog,L,.185,0.,.1,1.,.001),.00787*60.,.00787*0.,.25+.25*.04*env_AHDSRexp(Bprog,L,.225,0.,.1,1.,.001)*(.5+(.001*_sin(1.*Bprog))),.5+.04*env_AHDSRexp(Bprog,L,.225,0.,.1,1.,.001)*(.5+(.001*_sin(1.*Bprog))),.4999,1.,.00787*-255.,.00787*0.,.00787*0.,.00787*0.,9.)));
                        env = theta(Bprog)*pow(1.-smstep(Boff-rel, Boff, B),3.);
                    }
                    else if(syn == 138){
                        
                        amaysynL = (env_AHDSR(Bprog,L,.007,0.,.01,1.,.01)*(sinshape(MADD(_t,.5*f,0.,256,2,1.+.9*(.55+(.4*clip((1.+1.)*_sin(4.*BT)))),(1088.+(863.*_sin_(2.*BT,.4))),10.,5.37,3.85,.005,.4*(.55+(.4*clip((1.+1.)*_sin(4.*BT)))),0.,0.),1.,3.)+.8*clip((1.+.5)*_sin(.499*f*_t))+.4*_sq_(1.01*f*_t,.95))*maceboss_vol(BT));
                        amaysynR = (env_AHDSR(Bprog,L,.007,0.,.01,1.,.01)*(sinshape(MADD(_t2,.5*f,0.,256,2,1.+.9*(.55+(.4*clip((1.+1.)*_sin(4.*BT)))),(1088.+(863.*_sin_(2.*BT,.4))),10.,5.37,3.85,.005,.4*(.55+(.4*clip((1.+1.)*_sin(4.*BT)))),0.,0.),1.,3.)+.8*clip((1.+.5)*_sin(.499*f*_t2))+.4*_sq_(1.01*f*_t2,.95))*maceboss_vol(BT));
                    }
                    
                    sL += amtL * trk_norm(trk) * s_moothmin(clamp(env,0.,1.) * amaysynL);
                    sR += amtR * trk_norm(trk) * s_moothmin(clamp(env,0.,1.) * amaysynR);
                }
            }
            
        }
    }
    float masterL = .1 * sidechain * sL + .67 * dL;
    float masterR = .1 * sidechain * sR + .67 * dR;
    return vec2(
        masterL,
        masterR);
}

void main()
{
    Tsample = 1./iSampleRate;
    float t = (iBlockOffset + gl_FragCoord.x + gl_FragCoord.y*iTexSize) * Tsample;
    vec2 s = mainSynth(t);
    vec2 v  = floor((0.5+0.5*s)*65535.0);
    vec2 vl = mod(v,256.0)/255.0;
    vec2 vh = floor(v/256.0)/255.0;
    gl_FragColor = vec4(vl.x,vh.x,vl.y,vh.y);
}
