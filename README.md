# Safecity-Ai
fine tuning  a transfer lreaning model o yolov5 to help us find no helmet and no seatbelt violation in video and photo feed of cctv camera on road side

## Current Engineering Bottlenecks & Debugging Roadmap

The core frame-processing logic and model integration are functional, but the prototype currently experiences structural issues under specific edge cases. 

### Known Issues:
1. One and only issue occuring in the model is that it fail when its subjected to video which are taken from heights as i aint able to upload data here, only thing wrong from what i think is that my data was more like camera taking photos from low heights and thats why i think its failing when applied to indian traffic videos , and its failing majorly at pointing no helmet cases, if its used on uploading photo it isnt throughing error and is actually good at detecting, i dont know why it contius to fail on video thing

### Next Steps for Optimization:
* Implement asynchronous frame batching to decouple the video stream input from the model inference loop.
* Set up proper exception handling blocks within `server.py` to prevent server crashes during unaligned data inputs.

###One note for every reader

* I hadnt being able to push my all the file including data folder and other mentioned in my notebook due to come distrubance on my laptop so please go with it, server.py is good to go, and i had temporarly uploaded the weights of my final fine tuned model into this model folder as best.pt and made changes in the server.py so you can test run it anytime
