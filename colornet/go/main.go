package main

/*
#cgo LDFLAGS: -L. -lcolornet
#include "colornet.h"
#include <stdlib.h>
*/
import "C"
import (
    "fmt"
    "os"
    "unsafe"
)

func main() {
    // Load the model
    modelPath := C.CString("/home/asparkoffire/CLionProjects/ncnn/colornet/models/siggraph17_color_sim.bin")
    paramPath := C.CString("/home/asparkoffire/CLionProjects/ncnn/colornet/models/siggraph17_color_sim.param")
    defer C.free(unsafe.Pointer(modelPath))
    defer C.free(unsafe.Pointer(paramPath))

    net := C.load_model(modelPath, paramPath)
    if net == nil {
        fmt.Println("Failed to load model")
        return
    }
    defer C.unload_model(net)

    // Load input image
    inputBuffer, err := os.ReadFile("stalin.jpg")
    if err != nil || len(inputBuffer) == 0 {
        fmt.Println("Failed to read input image or image is empty:", err)
        return
    }


    // Prepare input buffer
    inputSize := C.size_t(len(inputBuffer))
    inputBufferC := C.CBytes(inputBuffer)
    defer C.free(inputBufferC)

    input := C.Buffer{
        data: (*C.uchar)(inputBufferC),
        size: inputSize,
    }

    // Set the format for output encoding (e.g., "jpg")
    format := C.CString("jpg") // Change the format to "jpg"
    defer C.free(unsafe.Pointer(format))

    // Perform inference
    result := C.infer(&input, net, format)
    if result.data == nil {
        fmt.Println("Inference failed")
        return
    }
    defer C.free(unsafe.Pointer(result.data)) // Free the data in result

    // Access the data in the result buffer
    outputImage := C.GoBytes(unsafe.Pointer(result.data), C.int(result.size))

    // Save the output image to a file
    outputImagePath := "./stalin_out.jpg"
    err = os.WriteFile(outputImagePath, outputImage, 0644)
    if err != nil {
        fmt.Println("Failed to save output image:", err)
        return
    }

    fmt.Println("Inference completed successfully. Output saved to:", outputImagePath)
}
