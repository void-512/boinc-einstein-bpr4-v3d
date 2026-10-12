layout(local_size_x_id=0) in;
layout(constant_id=3) const uint ARITHMETIC=0;
layout(constant_id=4) const uint FUSION=0;
layout(push_constant) uniform Parameters {
    uint n, unpadded, steps, logn;
    float tau, omega, psi, dt, step_inv, s0;
    uint fundamental, harmonic, window, stage;
    float threshold[5];
    uint capacity;
    uint fft_n, fft_base, fft_radix, flags, twiddle3, twiddle_real;
    uint result_slot;
} p;
layout(std430,binding=0) readonly buffer Input { float input_data[]; };
layout(std430,binding=1) buffer Raw { float raw[]; };
layout(std430,binding=2) buffer FFT { vec2 a[]; };
layout(std430,binding=3) readonly buffer Twiddles { vec2 tw[]; };
#ifdef BRP_ATOMIC_POWER
layout(std430,binding=4) buffer Power { uint power_bits[]; };
#else
layout(std430,binding=4) buffer Power { float power[]; };
#endif
layout(std430,binding=5) buffer Results { uint result_words[]; };
uint result_base() {return p.result_slot*(4+4*p.capacity);}
#define count result_words[result_base()]
#define overflow result_words[result_base()+1]
void write_candidate(uint slot,uvec4 value) {
    uint base=result_base()+4+slot*4;
    result_words[base]=value.x;result_words[base+1]=value.y;
    result_words[base+2]=value.z;result_words[base+3]=value.w;
}
layout(std430,binding=6) buffer Sums { float partial[]; };
layout(std430,binding=7) buffer Scratch { vec2 scratch[]; };
uint invocation() { return gl_GlobalInvocationID.x+gl_WorkGroupID.z*gl_NumWorkGroups.x*gl_WorkGroupSize.x; }
uint group_index() { return gl_WorkGroupID.x+gl_WorkGroupID.z*gl_NumWorkGroups.x; }
vec2 multiply(vec2 x,vec2 y) { return vec2(x.x*y.x-x.y*y.y,x.x*y.y+x.y*y.x); }
vec2 twiddle(uint width,uint j) {
#if BRP_STRIDED_TWIDDLES
    return tw[j<<(p.logn-uint(findMSB(width)))];
#else
    return tw[(width>>1)-1+j];
#endif
}
float fused_madd(float x,float y,float z) {
    if((ARITHMETIC&4)==0) return fma(x,y,z);
    // V3D lowers GLSL fma to separately rounded multiply/add. Recover the low
    // product and sum parts before the final rounding (normal finite domain).
    precise float high_x=uintBitsToFloat(floatBitsToUint(x)&0xfffff000u);
    precise float high_y=uintBitsToFloat(floatBitsToUint(y)&0xfffff000u);
    precise float low_x=x-high_x,low_y=y-high_y;
    precise float product=x*y;
    precise float error=high_x*high_y-product;
    error+=high_x*low_y;error+=low_x*high_y;error+=low_x*low_y;
    precise float sum=product+z,recovered=sum-product;
    precise float sum_error=(product-(sum-recovered))+(z-recovered);
    precise float correction=error+sum_error;
    precise float result=sum+correction;
    return result;
}
float sine(float x) {
    precise float xt=(1.0/6.283185)*x;
    xt=xt-trunc(xt); if(xt<0) xt+=1;
    int i=int(xt*64.0+0.5); // Multiplication by 64 is exact in this bounded domain.
    precise float d=6.283185*(xt-(1.0/64.0)*float(i));
    precise float d2=d*(0.5*d);
    float s=input_data[p.unpadded+2*i],c=input_data[p.unpadded+2*i+1];
    precise float v=(ARITHMETIC&2)!=0?fused_madd(-d2,s,fused_madd(d,c,s)):s+d*c-d2*s;
    return v;
}
