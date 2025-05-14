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
elif [ "$1" == "nymeria" ]; then
    echo "Finetuning on Nymeria..." 
    [ -d "mobileposer/checkpoints/$2/finetuned_nymeria" ] && rm -r "mobileposer/checkpoints/$2/finetuned_nymeria"
    python mobileposer/train.py --module joints --init-from mobileposer/checkpoints/$2/finetuned_imuposer/joints --finetune nymeria
    python mobileposer/train.py --module poser --init-from mobileposer/checkpoints/$2/finetuned_imuposer/poser --finetune nymeria
else
    echo "Invalid argument. Please specify 'dip' or 'imuposer' or 'nymeria'"
fi
