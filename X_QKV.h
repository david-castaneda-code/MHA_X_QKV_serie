#ifndef X_QKV
#define X_QKV

#include <ap_fixed.h>
#include <cstdint>
#include <stdint.h>
#include <hls_stream.h>

#define HEADS 12
#define MAX_M 64
#define MAX_N 768
#define MAX_A 64
#define MAX_B MAX_N/HEADS
#define MAX_C 64

void mat_tile_mul(
    const int8_t *X,                      //Input
    const int8_t *W,                      //Matriz de pesos
    hls::stream<int32_t> &O_stream,       //Salida
    int M,                                //Filas X
    int L,                                //Dimensión común
    int N,                                //Columnas W  
    const int update_A);                  //Cambiar input

void  mat_heads_tile_mul(
    const int32_t Q[MAX_A * MAX_B],       // Input local
    const int32_t Kt[MAX_B * MAX_C],      // Peso local
    int32_t QKt[MAX_A * MAX_C],           //Salida
    int A,                                //Filas X
    int B,                                //Dimensión común
    int C);                               //Columnas W       

template<bool traspuesta>
void tile_stream_read(
    hls::stream<int32_t> &O_stream,
    int32_t heads[HEADS][MAX_M * (MAX_N / HEADS)],
    const int M,
    const int N);

void softmax(
    const int32_t QKt[MAX_M * MAX_M],
    int32_t SV[MAX_M * MAX_M],
    const int M,
    const int N);

void process_QKV_serial(
    const int8_t *X,
    const int8_t *Wq,
    const int8_t *Wk,
    const int8_t *Wv,
    int16_t Q_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int16_t K_heads_out[HEADS][(MAX_N / HEADS) * MAX_M], // Transpuesta
    int16_t V_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t QKt_heads_out[HEADS][MAX_M * MAX_M], // Resultado atención sin softmax
    int16_t softmax_heads[HEADS][MAX_M * MAX_M],
    int32_t SV_heads[HEADS][MAX_M * (MAX_N/HEADS)],
    const int M, const int L, const int N);

// Función principal (sin templates)
void calcula_X_QKV(
    const int8_t *X,
    const int8_t *Wq,
    const int8_t *Wk,
    const int8_t *Wv,
    int16_t Q_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int16_t K_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int16_t V_heads_out[HEADS][MAX_M * (MAX_N / HEADS)],
    int32_t QKt_heads[HEADS][MAX_A*MAX_C],
    int16_t softmax_heads[HEADS][MAX_M * MAX_M],
    int32_t SV_heads[HEADS][MAX_M * (MAX_N/HEADS)]);

#endif
