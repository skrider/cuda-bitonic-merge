int ilog2(int x) {
    return sizeof(int)*8 - 1 - __builtin_clz(x);
}
