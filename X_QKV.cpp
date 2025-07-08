#include "X_QKV.h"
#include <cstdint>
#include <hls_stream.h>
#include <stdint.h>
#include "hls_task.h"

#define MAX_M 64 //Máximo de tokens a gestionar
#define MAX_L 768 //Embeddings a gestionar
#define MAX_N 768 //Columnas matrices de peso

#define COL_W 256 //Bloques de columnas en los que se va a separar W

#define HEADS 12  //Matrices en las que se dividen Q, K  y V.

const int DIM_TILE = 32; //Dimensiones tiles que se van a realizar

#define data_t_in int8_t
#define data_t_out int32_t

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

void process_QKV_serial(
    const int8_t *X,
    const int8_t *Wq,
    const int8_t *Wk,
    const int8_t *Wv,
    int32_t Q_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t K_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t V_heads_out[HEADS][MAX_M * (MAX_N / HEADS)])
{

    hls::stream<data_t_out> stream("O_stream");
    #pragma HLS STREAM variable=stream depth=32
  
    mat_tile_mul(X, Wk, stream, 64, 768, 768, 1);
    tile_stream_read<true>(stream, K_heads_out, 64, 768);

    mat_tile_mul(X, Wq, stream, 64, 768, 768, 0);
    tile_stream_read<false>(stream, Q_heads_out, 64, 768);
    
    mat_tile_mul(X, Wv, stream, 64, 768, 768, 0);
    tile_stream_read<false>(stream, V_heads_out, 64, 768);
}

void calcula_X_QKV(
    const int8_t *X,
    const int8_t *Wq,
    const int8_t *Wk,
    const int8_t *Wv,
    int32_t Q_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t K_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t V_heads_out[HEADS][MAX_M * (MAX_N / HEADS)])
{
    process_QKV_serial(X, Wq, Wk, Wv, Q_heads_out, K_heads_out, V_heads_out);
}
