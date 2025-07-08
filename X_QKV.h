#ifndef X_QKV
#define X_QKV

#include <ap_fixed.h>
#include <cstdint>
#include <stdint.h>
#include <hls_stream.h>

#define HEADS 12
#define MAX_M 64
#define MAX_N 768

void mat_tile_mul(
    const int8_t *X,                      //Input
    const int8_t *W,                      //Matriz de pesos
    hls::stream<int32_t> &O_stream,       //Salida
    int M,                                //Filas X
    int L,                                //Dimensión común
    int N,                                //Columnas W  
    const int update_A);                  //Cambiar input

template<bool traspuesta>
void tile_stream_read(
    hls::stream<int32_t> &O_stream,
    int32_t heads[HEADS][MAX_M * (MAX_N / HEADS)],
    const int M,
    const int N);

void process_QKV_serial(
    const int8_t *X,
    const int8_t *Wq,
    const int8_t *Wk,
    const int8_t *Wv,
    int32_t Q_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t K_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t V_heads_out[HEADS][MAX_M * (MAX_N / HEADS)]);

// Función principal (sin templates)
void calcula_X_QKV(
    const int8_t *X,
    const int8_t *Wq,
    const int8_t *Wk,
    const int8_t *Wv,
    int32_t Q_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t K_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t V_heads_out[HEADS][MAX_M * (MAX_N / HEADS)]);

#endif
