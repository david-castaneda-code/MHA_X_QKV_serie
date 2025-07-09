#include "X_QKV.h"
#include <cstdint>
#include <hls_stream.h>
#include <stdint.h>
#include "hls_task.h"

#define MAX_M 64              //Máximo de tokens a gestionar
#define MAX_L 768             //Embeddings a gestionar
#define MAX_N 768             //Columnas matrices de peso

#define COL_W 256             //Bloques de columnas en los que se va a separar W

#define HEADS 12              //Matrices en las que se dividen Q, K  y V.

#define MAX_A 64              //Máximo de tokens a gestionar
#define MAX_B 768/HEADS       //Embeddings a gestionar cabezas
#define MAX_C 64              //Columnas matriz Kt

const int DIM_TILE = 16;      //Dimensiones tiles que se van a realizar
const int DIM_TILE_HEADS = 8; //Dimensiones tiles cabezas

#define data_t_in int8_t
#define data_t_out int32_t

//Función para multiplicar X(input) * W(matrices de peso) optimizada por tiles y columnas
void  mat_tile_mul(
    const data_t_in *X,                    //Input
    const data_t_in *W,                    //Matriz de pesos
    hls::stream<data_t_out> &O_stream,     //Salida
    int M,                                 //Filas X
    int L,                                 //Dimensión común
    int N,                                 //Columnas W  
    const int update_A)                    //Actualizar X                  
{   
    #pragma HLS INTERFACE m_axi port = X offset = slave bundle = gmemX depth = MAX_M * MAX_L
    #pragma HLS INTERFACE m_axi port = W offset = slave bundle = gmemW depth = MAX_L * MAX_N

    #pragma HLS INTERFACE axis port = O_stream

    #pragma HLS INTERFACE s_axilite port = X bundle = control
    #pragma HLS INTERFACE s_axilite port = W bundle = control
    #pragma HLS INTERFACE s_axilite port = M bundle = control
    #pragma HLS INTERFACE s_axilite port = L bundle = control
    #pragma HLS INTERFACE s_axilite port = N bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    //Recepción input stream en memoria bram
    static  data_t_in X_bram[MAX_M][MAX_L];
    #pragma HLS BIND_STORAGE variable=X_bram type=ram_2p impl=bram

    if (update_A)
    {
        gestion_X:
        for (int i = 0; i < M; i++){
            for (int l = 0; l < L; l++){
                #pragma HLS PIPELINE II=1
                X_bram[i][l] = X[i * L + l];
            }
        }
    }
    //Recepción y gestión matriz B en trozos de columna
    bucle_general:
    for  (int columnas = 0; columnas < N; columnas += COL_W){
        int current_block_N = ((columnas + COL_W) <= N) ? COL_W : (N - columnas); //Por si el bloque a cargar supera las dimensiones de la matriz

        data_t_in W_bram[MAX_L][COL_W];
        #pragma HLS BIND_STORAGE variable = W_bram type = ram_2p impl = bram

        gestion_W:
        for (int l = 0; l < L; l++){
            for (int j = 0; j < current_block_N; j++){
                #pragma HLS PIPELINE II = 1
                W_bram[l][j] = W[l * N + columnas + j];
            }
        }       

        //Matrices X y W recorridas por parches (a cachos de filas y columnas)
        filas_tile:
        for (int i = 0; i < M; i += DIM_TILE){
            columnas_tile:
            for (int j = 0; j < current_block_N; j += DIM_TILE){

                data_t_out localO[DIM_TILE][DIM_TILE];
                #pragma HLS ARRAY_PARTITION variable=localO dim=0 type=complete
                init_output:
                    for (int ti = 0; ti < DIM_TILE; ti++) {
                        #pragma HLS UNROLL
                        for (int tj = 0; tj < DIM_TILE; tj++) {
                            #pragma HLS UNROLL
                            localO[ti][tj] = 0;
                        }
                    }    
                data_t_in localX[DIM_TILE][DIM_TILE];
                data_t_in localW[DIM_TILE][DIM_TILE];
                #pragma HLS ARRAY_PARTITION variable=localX dim=0 type=complete
                #pragma HLS ARRAY_PARTITION variable=localW dim=0 type=complete
        
                //Recorremos la dimensión común de las matrices a multiplicar (out_loop)
                dim_comun_tile:
                for (int l = 0; l < L; l += DIM_TILE){                   
                    
                    //Conseguimos tiles de la matriz input X: X_tile
                    tile_X:
                    for (int ti = 0; ti < DIM_TILE; ti++){
                        for(int tl = 0; tl < DIM_TILE; tl++){
                            #pragma HLS PIPELINE II=1       
                            int fila_x = i + ti;
                            int col_x  = l + tl;
                            if (fila_x < M && col_x < L)                 
                                localX[ti][tl] = X_bram[fila_x][col_x];
                            else
                                localX[ti][tl] = 0;
                        }
                    }

                    //Conseguimos tiles de las matrices de pesos W: W_tile
                    tile_W:
                    for (int tl = 0; tl < DIM_TILE; tl++){
                        for(int tj = 0; tj < DIM_TILE; tj++){
                            #pragma HLS PIPELINE II=1
                            int fila_w = l + tl;
                            int col_w  = j + tj;
                            if (fila_w < L && col_w < current_block_N) 
                                localW[tl][tj] = W_bram[fila_w][col_w];
                            else
                                localW[tl][tj] = 0;
                        }
                    }

                    calculo_matmul:
                    for (int tl = 0; tl < DIM_TILE; tl++){
                        #pragma HLS PIPELINE II=1
                        for (int ti = 0; ti < DIM_TILE; ti++){
                            #pragma HLS UNROLL
                            data_t_out x = static_cast<data_t_out>(localX[ti][tl]);
                            for (int tj = 0; tj < DIM_TILE; tj++){
                                #pragma HLS UNROLL
                                data_t_out w = static_cast<data_t_out>(localW[tl][tj]);
                                localO[ti][tj] += x * w;
                            }
                        }
                    }  
                }             

                //Pasamos lo calculado al stream de salida
                // Emitimos al stream en orden fila-columna global
                O_out_stream:
                for (int ti = 0; ti < DIM_TILE; ti++) {
                    for (int tj = 0; tj < DIM_TILE; tj++) {
                        #pragma HLS PIPELINE II=1
                        int fila_o = i + ti;
                        int col_o  = j + tj + columnas;
                        if (fila_o < M && col_o < N)
                            O_stream.write(localO[ti][tj]);
                    }
                }
            }
        }
    }
}

//Función para multiplicar Q_heads * K_heads = QKt heads optimizada por tiles
void  mat_heads_tile_mul(
    const data_t_out Q[MAX_A*MAX_B],       // Input local
    const data_t_out Kt[MAX_B*MAX_C],      // Peso local
    data_t_out QKt[MAX_A*MAX_C],           //Salida
    int A,                                 //Filas X
    int B,                                 //Dimensión común
    int C)                                 //Columnas W            
{   
    #pragma HLS INTERFACE m_axi port = Q offset = slave bundle = gmemQ depth = MAX_A * MAX_B
    #pragma HLS INTERFACE m_axi port = Kt offset = slave bundle = gmemK depth = MAX_B * MAX_C
    #pragma HLS INTERFACE m_axi port = QKt offset = slave bundle = gmemQKt depth = MAX_A * MAX_C

    #pragma HLS INTERFACE s_axilite port = Q bundle = control
    #pragma HLS INTERFACE s_axilite port = Kt bundle = control
    #pragma HLS INTERFACE s_axilite port = QKt bundle = control
    #pragma HLS INTERFACE s_axilite port = A bundle = control
    #pragma HLS INTERFACE s_axilite port = B bundle = control
    #pragma HLS INTERFACE s_axilite port = C bundle = control
    #pragma HLS INTERFACE s_axilite port = return bundle = control

    //Recepción input stream en memoria bram
    static data_t_out Q_bram[MAX_A][MAX_B];
    #pragma HLS BIND_STORAGE variable=Q_bram type=ram_2p impl=bram

    gestion_Q:
    for (int i = 0; i < A; i++){
        for (int l = 0; l < B; l++){
            #pragma HLS PIPELINE II=1
            Q_bram[i][l] = Q[i * B + l];
        }
    }

    //Recepción y gestión matriz Kt
    static data_t_out Kt_bram[MAX_B][MAX_C];
    #pragma HLS BIND_STORAGE variable = Kt_bram type = ram_2p impl = bram

    gestion_Kt:
    for (int l = 0; l < B; l++){
        for (int j = 0; j < C; j++){
            #pragma HLS PIPELINE II = 1
            Kt_bram[l][j] = Kt[l * C + j];
        }
    }       
    bucle_general_heads:
    //Matrices X y W recorridas por parches (a cachos de filas y columnas)
    filas_tile:
    for (int i = 0; i < A; i += DIM_TILE_HEADS){
        columnas_tile:
        for (int j = 0; j < C; j += DIM_TILE_HEADS){

            data_t_out localQKt[DIM_TILE_HEADS][DIM_TILE_HEADS];
            #pragma HLS ARRAY_PARTITION variable=localQKt dim=0 type=complete
            init_output:
                for (int ti = 0; ti < DIM_TILE_HEADS; ti++) {
                    #pragma HLS UNROLL
                    for (int tj = 0; tj < DIM_TILE_HEADS; tj++) {
                        #pragma HLS UNROLL
                        localQKt[ti][tj] = 0;
                    }
                }    
            data_t_out localQ[DIM_TILE_HEADS][DIM_TILE_HEADS];
            data_t_out localKt[DIM_TILE_HEADS][DIM_TILE_HEADS];
            #pragma HLS ARRAY_PARTITION variable=localQ dim=0 type=complete
            #pragma HLS ARRAY_PARTITION variable=localKt dim=0 type=complete
    
            //Recorremos la dimensión común de las matrices a multiplicar (out_loop)
            dim_comun_tile_heads:
            for (int l = 0; l < B; l += DIM_TILE_HEADS){                   
                
                //Conseguimos tiles de la matriz input X: X_tile
                tile_Q:
                for (int ti = 0; ti < DIM_TILE_HEADS; ti++){
                    for(int tl = 0; tl < DIM_TILE_HEADS; tl++){
                        #pragma HLS PIPELINE II=1       
                        int fila_x = i + ti;
                        int col_x  = l + tl;
                        if (fila_x < A && col_x < B)                 
                            localQ[ti][tl] = Q_bram[fila_x][col_x];
                        else
                            localQ[ti][tl] = 0;
                    }
                }

                //Conseguimos tiles de las matrices de pesos W: W_tile
                tile_Kt:
                for (int tl = 0; tl < DIM_TILE_HEADS; tl++){
                    for(int tj = 0; tj < DIM_TILE_HEADS; tj++){
                        #pragma HLS PIPELINE II=1
                        int fila_w = l + tl;
                        int col_w  = j + tj;
                        if (fila_w < B && col_w < C) 
                            localKt[tl][tj] = Kt_bram[fila_w][col_w];
                        else
                            localKt[tl][tj] = 0;
                    }
                }

                calculo_matmul_heads:
                for (int tl = 0; tl < DIM_TILE_HEADS; tl++){
                    #pragma HLS PIPELINE II=1
                    for (int ti = 0; ti < DIM_TILE_HEADS; ti++){
                        #pragma HLS UNROLL
                        data_t_out q = localQ[ti][tl];
                        for (int tj = 0; tj < DIM_TILE_HEADS; tj++){
                            #pragma HLS UNROLL
                            data_t_out kt = localKt[tl][tj];
                            localQKt[ti][tj] += q * kt;
                        }
                    }
                }  
            }             

            //Pasamos lo calculado al stream de salida
            QKt_out_heads:
            for (int ti = 0; ti < DIM_TILE_HEADS; ti++) {
                for (int tj = 0; tj < DIM_TILE_HEADS; tj++) {
                    #pragma HLS PIPELINE II=1
                    int fila_o = i + ti;
                    int col_o  = j + tj;
                    if (fila_o < A && col_o < C)
                        QKt[fila_o * C + col_o] = localQKt[ti][tj];
                }
            }
        }
    }
}

//Función que recibe stream de mat_tile_mul -> Q, K y V; y los separa en heads.
template<bool traspuesta>
void tile_stream_read(
    hls::stream<data_t_out> &O_stream,
    data_t_out heads[HEADS][MAX_M * (MAX_N / HEADS)],
    const int M,
    const int N)
{
    #pragma HLS INTERFACE axis port=O_stream

    const int N_head = N/HEADS;
    separar_filas:
    for (int i = 0; i < M; i++) {
        separar_columnas:
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            data_t_out dato = O_stream.read();
            int h = j/N_head;

            if (traspuesta == false){
                heads[h][i * N_head + (j % N_head)] = dato;
            } 
            else {
                heads[h][(j % N_head) * M + i] = dato;
            }
        }
    }
}

//Proceso MHA
void process_QKV_serial(
    const int8_t *X,
    const int8_t *Wq,
    const int8_t *Wk,
    const int8_t *Wv,
    int32_t Q_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t K_heads_out[HEADS][(MAX_N / HEADS) * MAX_M], // Transpuesta
    int32_t V_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t QKt_heads_out[HEADS][MAX_M * MAX_M], // Resultado atención sin softmax
    const int M, const int L, const int N)
{
    const int D = N / HEADS;
    {
        hls::stream<data_t_out> stream_k("stream_k");
        #pragma HLS STREAM variable=stream_k depth=64

        mat_tile_mul(X, Wk, stream_k, M, L, N, 1); 
        tile_stream_read<true>(stream_k, K_heads_out, M, N);
    }
    {
        hls::stream<data_t_out> stream_q("stream_q");
        #pragma HLS STREAM variable=stream_q depth=64

        mat_tile_mul(X, Wq, stream_q, M, L, N, 0);
        tile_stream_read<false>(stream_q, Q_heads_out, M, N);
    }
    {
        hls::stream<data_t_out> stream_v("stream_v");
        #pragma HLS STREAM variable=stream_v depth=64

        mat_tile_mul(X, Wv, stream_v, M, L, N, 0); 
        tile_stream_read<false>(stream_v, V_heads_out, M, N);
    }
    calculo_QKt_heads:
    for (int h = 0; h < HEADS; h++) {
        mat_heads_tile_mul(Q_heads_out[h],K_heads_out[h],QKt_heads_out[h],M, D, M); 
    }
}

void calcula_X_QKV(
    const int8_t *X,
    const int8_t *Wq,
    const int8_t *Wk,
    const int8_t *Wv,
    int32_t Q_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t K_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t V_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t QKt[HEADS][MAX_A*MAX_C])
{
    process_QKV_serial(X, Wq, Wk, Wv, Q_heads_out, K_heads_out, V_heads_out,QKt,64,768,768);
}
