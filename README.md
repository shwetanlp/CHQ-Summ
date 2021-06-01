# Reinforcement Learning for Abstractive Question Summarization with Question-aware Semantic Rewards



The code requires **Python 3** and please install the Python dependencies with the command:
```bash
pip install -r requirements.txt
```


### Running the code 
1. Please make sure the pre-trained question-type identification and question-focus recognition models from [here](https://drive.google.com/drive/folders/1ePtuMPR20rZSgZbarSnno4-sqazLJVn0?usp=sharing) and 
    place it in the current directory.

2. Train MLE Model
    ```
     python main.py --train_mode mle --model_dir /path/to/save/the/mle/model
    ```

3. Train MLE + RL Model
    ```
    python main.py --train_mode rl --trained_model_path /path/to/the/trained/mle/model
    ```

4. Test Model
    ```
    python main.py --model test --trained_model_path /path/to/the/saved/model

    ```
