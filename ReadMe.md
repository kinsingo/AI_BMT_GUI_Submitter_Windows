> **Last Updated:** 2026-03-03 (Version 2.3)
## Environment
1. ISA(Instruction Set Architecture) : AMD64(x86_64)
2. OS : Windows 10
    
## Project Description
1. Implement AI_BMT_Interface to operate with the intended AI Processing Unit (e.g., CPU, GPU, NPU).
2. Various task example codes are provided. Use these example codes as a reference to implement the interface for the AI Processing Unit.

## Submitter User Guide Steps
Step1) Build System Set-up  
Step2) Interface Implementation  
Step3) Build and Start BMT

## Step 1) Build System Set-up (Installation Guide for Windows)
### 1. Current Project Settings (Do Not Modify)
  - ISO C++17 standard (/std:c++17) is used (C++17 or higher is required).
  - References:
     - Header files: `snu_bmt_gui_caller.h`, `snu_bmt_interface.h`, `label_type.h` (located in the `include` folder).
     - Library: `SNU_AI_BMT_GUI_Library.lib` (located in the `lib` folder).
  - Most files in the `Release/Debug` folder where the executable is generated should not be deleted.  
     - Exceptions: OpenCV/ONNXRuntime-related DLLs can be deleted if unnecessary.
### 2. Current Project Settings (Modifiable)
  - OpenCV 3.416 version has been included (headers/lib/DLL).
     - Headers: `\include\opencv3416`
     - Library: `\lib\opencv3416`
     - DLLs: `opencv_world3416.dll` (Release), `opencv_world3416d.dll` (Debug)
  - ONNXRuntime has been included (headers/lib/DLL).
     - Headers: `\include\onnxruntime`
     - Library: `\lib\onnxruntime`
     - DLLs: `onnxruntime.dll`, `onnxruntime_providers_shared.dll`

## Step2) Interface Implementation
- Implement the `SNU_BMT_Interface` interface.
- Ensure that these functions operate correctly on the intended computing unit (e.g., CPU, GPU, NPU).

```cpp
#ifndef SNU_BMT_INTERFACE_H
#define SNU_BMT_INTERFACE_H
#include "label_type.h"
using namespace std;

class EXPORT_SYMBOL AI_BMT_Interface
{
public:
   virtual ~AI_BMT_Interface(){}

    // Optional: override to provide system metadata.
    // Returned values will be stored in the database (used for benchmarking context).
   virtual Optional_Data getOptionalData();

   // return the implemented interface task type. 
   virtual InterfaceType getInterfaceType() = 0;

   // This initialize(..) function is guaranteed to be called before preprocess(..) and infer(..) are executed.
   // The submitter can load the model using the provided modelPath
   virtual void initialize(string modelPath) = 0;
   
    // Power measurement selection (default: do not measure)
   virtual PowerDeviceType getPowerDeviceType() { return PowerDeviceType::None; }

   // Vision tasks: preprocessing & inference
   // - preprocessVisionData: convert raw image file into model input format
   // - inferVision: run inference on preprocessed data and return results
   virtual VariantType preprocessVisionData(const string& imagePath) {throw runtime_error("preprocessVisionData(..) should be implemented for vision task");}
   virtual vector<BMTVisionResult> inferVision(const vector<VariantType>& data) {throw runtime_error("inferVision(..) should be implemented for vision task");}

   // LLM tasks: preprocessing & inference
   // - preprocessLLMData: convert raw text input into model input format
   // - inferLLM: run inference on preprocessed data and return results
   virtual VariantType preprocessLLMData(const LLMPreprocessedInput& llmData) {throw runtime_error("LLMPreprocessedInput(..) should be implemented for llm task");}
   virtual vector<BMTLLMResult> inferLLM(const vector<VariantType>& data) {throw runtime_error("inferLLM(..) should be implemented for llm task");}
   
   // LLM MMLU tasks: first token generation for TTFT measurement
   // - inferFirstToken: generate only the first token (AI-BMT will measure the time internally)
   // - Returns void (we only measure TTFT, don't care about the actual first token output)
   // - Only used for MMLU tasks that require TTFT measurement
   virtual void inferFirstToken(const VariantType& data) {throw runtime_error("inferFirstToken(..) should be implemented for MMLU task");}

};

#endif // AI_BMT_INTERFACE_H
```

## Step3) Build and Start BMT
: It's recommended to use Visual Studio 2022 for this step.
