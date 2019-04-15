__kernel void edge_padding(__global float *X){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int batch = get_global_id(2);


    if(x < P@shape[3] || x > (X@shape[3] - P@shape[3] - 1) || y < P@shape[2] || y > (X@shape[2] - P@shape[2] - 1)){
        for(int c = 0; c < X@shape[1]; c++){
            const int ix = min(X@shape[3] - P@shape[3] - 1, max(P@shape[3], x));
            const int iy = min(X@shape[2] - P@shape[2] - 1, max(P@shape[2], y));

            X@[batch, c, y, x] = X@[batch, c, iy, ix];
        }
    }

}