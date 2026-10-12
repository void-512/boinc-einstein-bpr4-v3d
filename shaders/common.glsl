layout(local_size_x_id=0) in;
layout(push_constant) uniform Parameters {
    uint n, unpadded, steps, logn;
    float tau, omega, psi, dt, step_inv, s0;
    uint fundamental, harmonic, window, stage;
    float threshold[5];
    uint capacity;
} p;
layout(std430,binding=0) readonly buffer Input { float input_data[]; };
layout(std430,binding=1) buffer Raw { float raw[]; };
layout(std430,binding=2) buffer FFT { vec2 a[]; };
layout(std430,binding=3) readonly buffer Twiddles { vec2 tw[]; };
layout(std430,binding=4) buffer Power { float power[]; };
layout(std430,binding=5) buffer Results { uint count, overflow; uvec4 candidates[]; };
layout(std430,binding=6) buffer Sums { float partial[]; };
layout(std430,binding=7) buffer Scratch { vec2 scratch[]; };
vec2 multiply(vec2 x,vec2 y) { return vec2(x.x*y.x-x.y*y.y,x.x*y.y+x.y*y.x); }
vec2 twiddle(uint width,uint j) {
#if BRP_STRIDED_TWIDDLES
    return tw[j<<(p.logn-uint(findMSB(width)))];
#else
    return tw[(width>>1)-1+j];
#endif
}
float sine(float x) {
    precise float xt=(1.0/6.283185)*x;
    xt=xt-trunc(xt); if(xt<0) xt+=1;
    int i=int(xt*64.0+0.5);
    precise float d=6.283185*(xt-(1.0/64.0)*float(i));
    precise float d2=d*(0.5*d);
    float s=input_data[p.unpadded+2*i],c=input_data[p.unpadded+2*i+1];
    precise float v=s+d*c-d2*s;
    return v;
}
