# INSTRUCTERC-MBTI

Model for textual multiclass emotion recognition in conversation.
Based on the InstructionERC model: https://github.com/LIN-SHANG/InstructERC with personality type categorization and reduction of video memory requirements additions.

Used datasets:
* MELD (https://paperswithcode.com/dataset/meld)
* DailyDialog (https://paperswithcode.com/dataset/dailydialog)

# Run

InstructERC Setup Environment:
```
cd ./InstructERC/envs/
pip3 install -r requirements.txt
```

To reproduce the results, run the ***train_and_inference_Uni.sh***

```
bash train_and_inference_Uni.sh
```

The hyperparameters you need setting: 
```
1.MODEL_NAME

2.Experiments_setting

3.dataset

4.historical_window

5.accumulations

6.graphics_card
Notes: batch size = graphics_card * accumulations

7.data_percent
```

***data_process.py*** contains different prompts for LLM ***at lines 107-147***
