
To run the project:
1. Create a local env using conda, pyenv or any other tool
2. Run 'pip install -r requirements.txt' (may encounter dependencies issues installing crypten)
3. Run scripts.sh > trains substitute model
4. Note: The settings for epoch is set to 100, but it is recommended to run it for less than 50 epochs to see effectiveness of model
5. Create folder model/model_pr in the root of the repo to store substitute and parent models
6. Run python3 senet-mpc/main.py > gives inference time for each model
