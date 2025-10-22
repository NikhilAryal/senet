
To run the project:
1. Create a local env using conda, pyenv or any other tool
2. Run 'pip install -r requirements.txt'
3. Run scripts.sh
4. Note: The settings for epoch is set to 100, but it is recommended to run it for less than 50 epochs to see effectiveness of model

API and GUI:
As the project's scope was to finalize paper implementation part for Midterm, the GUI/API part, associated with inference time verification,
is still ongoing, and has not been released yet.


Status of project:
Senet model implemented with 50K ReLu budget. 
Masking and training PR model, using AR model's eval values and distillation loss yields 
loss = 0.739
accuracy = 75%
(base model accuracy = 67%)