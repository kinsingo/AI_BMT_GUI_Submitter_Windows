> **Last Updated:** 2025-04-23 — Object Detection is now supported (example code included)
## Environment
1. ISA(Instruction Set Architecture) : AMD64(x86_64)
2. OS : Windows 10
    
## Project Description
1. Implement the overridden functions in the `Virtual_Submitter_Implementation` class within `main.cpp`.  
   - Ensure these functions are correctly implemented to operate with the intended AI Processing Unit (e.g., CPU, GPU, NPU).
2. A classification example code is provided. Use this example code as a reference to implement the interface for the AI Processing Unit.

## Submitter User Guide Steps
Step1) Build System Set-up  
Step2) Interface Implementation  
Step3) Build and Start BMT

## Step 1) Build System Set-up (Installation Guide for Windows)
### 1. Current Project Settings (Do Not Modify)
  - ISO C++17 standard (/std:c++17) is used (C++17 or higher is required).
  - References:
     - Header files: `snu_bmt_gui_caller.h`, `snu_bmt_interface.h` (located in the `include` folder).
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
- Implement the overridden functions in the `Virtual_Submitter_Implementation` class, which inherits from the `SNU_BMT_Interface` interface, within `main.cpp`.
- Ensure that these functions operate correctly on the intended computing unit (e.g., CPU, GPU, NPU).
![SNU_BMT_Interface_Diagram_For_README](https://github.com/user-attachments/assets/4d863c75-14df-4af1-98e0-2c623753b98c)

```cpp
#ifndef SNU_BMT_INTERFACE_H
#define SNU_BMT_INTERFACE_H
#include "label_type.h"
using namespace std;

// Represents the result of the inference process for a single query.
// Fields:
// - Classification_ImageNet_PredictedIndex_0_to_999: An integer representing the predicted class index (0-999) for the ImageNet dataset.
struct EXPORT_SYMBOL BMTResult
{
    // Output scores for 1000 ImageNet classes from the classification model.
    // Each element represents the probability or confidence score for a class.
    vector<float> classProbabilities;

    // Output tensor from object detection model.
    // The vector stores raw model outputs for 25200 detection candidates.
    // Each candidate includes 85 values: [x, y, w, h, objectness, 80 class scores].
    // Total size must be exactly 25200 * 85 = 2,142,000 elements.
    vector<float> objectDetectionResult;
};

// Stores optional system configuration data provided by the Submitter.
// These details will be uploaded to the database along with the performance data.
struct EXPORT_SYMBOL Optional_Data
{
    string cpu_type; // e.g., Intel i7-9750HF
    string accelerator_type; // e.g., DeepX M1(NPU)
    string submitter; // e.g., DeepX
    string cpu_core_count; // e.g., 16
    string cpu_ram_capacity; // e.g., 32GB
    string cooling; // e.g., Air, Liquid, Passive
    string cooling_option; // e.g., Active, Passive (Active = with fan/pump, Passive = without fan)
    string cpu_accelerator_interconnect_interface; // e.g., PCIe Gen5 x16
    string benchmark_model; // e.g., ResNet-50
    string operating_system; // e.g., Ubuntu 20.04.5 LTS
};

// A variant can store and manage values only from a fixed set of types determined at compile time.
// Since variant manages types statically, it can be used with minimal runtime type-checking overhead.
// std::get<DataType>(variant) checks if the requested type matches the stored type and returns the value if they match.
using VariantType = variant<uint8_t*, uint16_t*, uint32_t*,
                            int8_t*,int16_t*,int32_t*,
                            float*, // Define variant pointer types
                            vector<uint8_t>, vector<uint16_t>, vector<uint32_t>,
                            vector<int8_t>, vector<int16_t>, vector<int32_t>,
                            vector<float>>; // Define variant vector types

class EXPORT_SYMBOL SNU_BMT_Interface
{
public:
   virtual ~SNU_BMT_Interface(){}

   // This is not mandatory but can be implemented if needed.
   // The virtual function getOptionalData() returns an Optional_Data object,
   // which includes fields like CPU_Type and Accelerator_Type.
   // By default, these fields are initialized as empty strings.
   virtual Optional_Data getOptionalData()
   {
       Optional_Data data;
       data.cpu_type = ""; // e.g., Intel i7-9750HF
       data.accelerator_type = ""; // e.g., DeepX M1(NPU)
       data.submitter = ""; // e.g., DeepX
       data.cpu_core_count = ""; // e.g., 16
       data.cpu_ram_capacity = ""; // e.g., 32GB
       data.cooling = ""; // e.g., Air, Liquid, Passive
       data.cooling_option = ""; // e.g., Active, Passive (Active = with fan/pump, Passive = without fan)
       data.cpu_accelerator_interconnect_interface = ""; // e.g., PCIe Gen5 x16
       data.benchmark_model = ""; // e.g., ResNet-50
       data.operating_system = ""; // e.g., Ubuntu 20.04.5 LTS
       return data;
   }

   // It is recommended to use this instead of a constructor,
   // as it allows handling additional errors that cannot be managed within the constructor.
   // The Initialize function is guaranteed to be called before convertToData and runInference are executed.
   virtual void Initialize() = 0;

   // Performs preprocessing before AI inference to convert data into the format required by the AI Processing Unit.
   // This method prepares model input data and is excluded from latency/throughput performance measurements.
   // The converted data is loaded into RAM prior to invoking the runInference(..) method.
   virtual VariantType convertToPreprocessedDataForInference(const string& imagePath) = 0;

   // Returns the final BMTResult value of the batch required for performance evaluation in the App.
   virtual vector<BMTResult> runInference(const vector<VariantType>& data) = 0;
};
#endif // SNU_BMT_INTERFACE_H
```

## Step3) Build and Start BMT
: It's recommended to use Visual Studio 2022 for this step.
