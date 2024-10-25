# GestLLM

This is project for researching about VLM/LLM capabilities in gesture detection with adding some RAG-like features, based on Mediapipe

dataset_reformater.py is for reformatting dataset from Hagrid format to form appropriate for LLAMA-Factory finetuning tool 
geom_helper.py is for some geometry functions
gesture_vectorization.py is for unfinished feature for gesture vectorization(which allows application of clasterization methods and much more)
hand2descrition.py is for creating from raw Mediapipe data something JSON-like and then use it to create text description of the gesture
langchain_test.py is for testing some VLM capabilities
main.py - main file to bring every part of project to something working

instruction for main.py:
0) add folder models and models for MediaPipe part
1) make sure that your cam number for opencv is right, and add API keys/base url for endpoint(and launch it if it is local)
2) launch
3) right hand is controlling test workflow, and left hand showing gestures
4) turn right hand to fist to start recording gesture
5) turn right hand to palm to end recording gesture and sending everything to endpoint
6) enjoy results