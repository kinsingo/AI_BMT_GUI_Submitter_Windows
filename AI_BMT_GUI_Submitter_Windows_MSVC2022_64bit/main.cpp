#include "ai_bmt_gui_caller.h"
#include "ai_bmt_interface.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <onnxruntime_cxx_api.h>
#include <filesystem>
#include "ImageClassification_Implementaion.cpp"
#include "ImageClassification_CustomDataset_Interface_Implementation.cpp"
#include "ObjectDetection_Implementation.cpp"
#include "Segmentation_Implementation.cpp"
#include "llm_Implementation.cpp"

int main(int argc, char* argv[])
{
    try
    {
		// -- For Single Task --
        shared_ptr<AI_BMT_Interface> interface = make_shared<ImageClassification_Interface_Implementation>(); 
        //shared_ptr<AI_BMT_Interface> interface = make_shared<ImageClassification_CustomDataset_Interface_Implementation>();
        //shared_ptr<AI_BMT_Interface> interface = make_shared<ObjectDetection_Interface_Implementation>(); 
        //shared_ptr<AI_BMT_Interface> interface = make_shared<ObjectDetection_CustomDataset_Interface_Implementation>();
        //shared_ptr<AI_BMT_Interface> interface = make_shared<Segmentation_Interface_Implementation>(); 
        //shared_ptr<AI_BMT_Interface> interface = make_shared<Segmentation_CustomDataset_Interface_Implementation>(); 
        //shared_ptr<AI_BMT_Interface> interface = make_shared<LLM_Interface_Implementation>();
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Single_Task(argc, argv, interface);

		// -- For Multi-Domain Tasks --
        /*
        vector<shared_ptr<AI_BMT_Interface>> interfaceVector =
        {
            make_shared<ImageClassification_Interface_Implementation>(),
            make_shared<ObjectDetection_Interface_Implementation>(),
            make_shared<Segmentation_Interface_Implementation>(),
		};
        return AI_BMT_GUI_CALLER::call_BMT_GUI_For_Multiple_Tasks(argc, argv, interfaceVector);
        */
    }
    catch (const exception& ex)
    {
        cout << ex.what() << endl;
    }
}
