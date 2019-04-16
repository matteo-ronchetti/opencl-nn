__kernel void convolution(__global const float *X, __constant const float *W, __global float *Y){
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    const int batch = get_global_id(2);

    const int n_filters = W@shape[0];
    const int filter_channels = W@shape[1];
    const int filter_h = W@shape[2];
    const int filter_w = W@shape[3];

    __local float filter[W@shape[2] * W@shape[3]];


    for(int d = 0; d < n_filters; d++){
        float tmp = 0;

        for(int c = 0; c < filter_channels; c++){
            if(get_local_id(0) < filter_h && get_local_id(1) < filter_w){
                filter[get_local_id(0)*W@shape[3]+get_local_id(1)] = W@[d, c, get_local_id(0), get_local_id(1)];
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            for(int i = 0; i < filter_h; i++){
                for(int j = 0; j < filter_w; j++){
                    tmp += $conv_op{X@[batch, c, y+i, x+j], filter[i*W@shape[3]+j]};
                }
            }
        }
        Y@padded[batch, d, y, x] = $final_op{tmp};
    }

}