// The historical dyadic ordering matters: preserve each separately rounded sum.
vec4 dyadic_sums(uint i) {
    precise float s=READ_POWER(i);
    s+=READ_POWER((i*8+8)>>4);float s2=s;
    s+=READ_POWER((i*12+8)>>4)+READ_POWER((i*4+8)>>4);float s4=s;
    s+=READ_POWER((i*14+8)>>4)+READ_POWER((i*10+8)>>4)+READ_POWER((i*6+8)>>4)+READ_POWER((i*2+8)>>4);float s8=s;
    s+=READ_POWER((i*15+8)>>4)+READ_POWER((i*13+8)>>4)+READ_POWER((i*11+8)>>4)+READ_POWER((i*9+8)>>4)+
       READ_POWER((i*7+8)>>4)+READ_POWER((i*5+8)>>4)+READ_POWER((i*3+8)>>4)+READ_POWER((i+8)>>4);
    return vec4(s2,s4,s8,s);
}
