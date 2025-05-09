#!/bin/bash

if [ "$1" == "dip" ]; then
    echo "Finetuning on DIP..." 
    [ -d "mobileposer/checkpoints/$2/finetuned_dip" ] && rm -r "mobileposer/checkpoints/$2/finetuned_dip"
    python mobileposer/train.py --module joints --init-from mobileposer/checkpoints/$2/joints --finetune dip
    python mobileposer/train.py --module poser --init-from mobileposer/checkpoints/$2/poser --finetune dip
elif [ "$1" == "imuposer" ]; then 
    echo "Finetuning on IMUPoser..." 
    [ -d "mobileposer/checkpoints/$2/finetuned_imuposer" ] && rm -r "mobileposer/checkpoints/$2/finetuned_imuposer"
    python mobileposer/train.py --module joints --init-from mobileposer/checkpoints/$2/finetuned_dip/joints --finetune imuposer
    python mobileposer/train.py --module poser --init-from mobileposer/checkpoints/$2/finetuned_dip/poser --finetune imuposer
else
    echo "Invalid argument. Please specify 'dip' or 'imuposer'"
fi
