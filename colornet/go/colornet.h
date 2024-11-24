#ifndef COLORNET_H
#define COLORNET_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

// Buffer structure to hold data and its size
typedef struct {
    const unsigned char* data;
    size_t size;
} Buffer;

// Function to load a model with given model and parameter paths
void* load_model(const char* model_path, const char* param_path);

// Function to unload the model
void unload_model(void* net_ptr);

// Function to perform inference, returns a pointer to a Buffer containing the result
Buffer* infer(Buffer* input_buffer, void* net_ptr, const char* format);

#ifdef __cplusplus
}
#endif

#endif // COLORNET_H
